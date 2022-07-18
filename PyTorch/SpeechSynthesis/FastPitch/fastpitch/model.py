# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import ConvReLUNorm
from common.utils import mask_from_lens
from fastpitch.alignment import b_mas, mas_width1
from fastpitch.attention import ConvAttention
from fastpitch.transformer import FFTransformer


def regulate_len(durations, enc_out, pace: float = 1.0,
                 mel_max_len: Optional[int] = None):
    """If target=None, then predicted durations are applied"""
    #up sampling the frame
    dtype = enc_out.dtype
    reps = durations.float() / pace
    reps = (reps + 0.5).long()
    dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    reps_cumsum = torch.cumsum(F.pad(reps, (1, 0, 0, 0), value=0.0),
                               dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype)

    range_ = torch.arange(max_len).to(enc_out.device)[None, :, None]
    mult = ((reps_cumsum[:, :, :-1] <= range_) &
            (reps_cumsum[:, :, 1:] > range_))
    mult = mult.to(dtype)
    enc_rep = torch.matmul(mult, enc_out)

    if mel_max_len is not None:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)
    return enc_rep, dec_lens


def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = F.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = F.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce)
                  - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce)
                    - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems,
                            pitch_sums / pitch_nelems)
    return pitch_avg


class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""
    # for pitch, energy and duration (maybe spectral tilt)
    def __init__(self, input_size, filter_size, kernel_size, dropout,
                 n_layers=2, n_predictions=1):
        super(TemporalPredictor, self).__init__()
        # (in_fft_output_size=384, filter_size=256, kernel_size=3, dropout=0.1, n_layers=2, n_predictions=1)

        self.layers = nn.Sequential(*[
            ConvReLUNorm(input_size if i == 0 else filter_size, filter_size,
                         kernel_size=kernel_size, dropout=dropout)
            for i in range(n_layers)]
        )
        self.n_predictions = n_predictions
        self.fc = nn.Linear(filter_size, self.n_predictions, bias=True)

    def forward(self, enc_out, enc_out_mask):
        out = enc_out * enc_out_mask
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out) * enc_out_mask
        return out


class BinaryClassification(nn.Module):
    """Predicts a categorical label per each temporal location"""
    # for categorical cwt labels
    def __init__(self, input_size, filter_size, kernel_size, dropout,
                 n_layers=2, n_predictions=1):
        super(BinaryClassification, self).__init__()
        self.layers = nn.Sequential(*[
            ConvReLUNorm(input_size if i == 0 else filter_size, filter_size,
                         kernel_size=kernel_size, dropout=dropout)
            for i in range(n_layers)]
        )  # in_channels, out_channels, kernel_size=1, dropout=0.0
        self.n_predictions = n_predictions
        self.sigmoid = nn.sigmoid()
        self.fc = nn.Linear(filter_size, self.n_predictions, bias=True)
        # self.lastcnn = ConvReLUNorm(filter_size, self.n_predictions, kernel_size=filter_size)

    def forward(self, enc_out, enc_out_mask):
        out = enc_out * enc_out_mask
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.sigmoid(self.fc(out)) * enc_out_mask
        # out = self.sigmoid(self.lastcnn(out)) * enc_out_mask
        return out


class FastPitch(nn.Module):
    def __init__(self, n_mel_channels, n_symbols, padding_idx,
                 symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads,
                 in_fft_d_head,
                 in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size,
                 in_fft_output_size,
                 p_in_fft_dropout, p_in_fft_dropatt, p_in_fft_dropemb,
                 out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
                 out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
                 out_fft_output_size,
                 p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb,
                 dur_predictor_kernel_size, dur_predictor_filter_size,
                 p_dur_predictor_dropout, dur_predictor_n_layers,
                 pitch_conditioning,
                 pitch_predictor_kernel_size, pitch_predictor_filter_size,
                 p_pitch_predictor_dropout, pitch_predictor_n_layers,
                 pitch_embedding_kernel_size,
                 energy_conditioning,
                 energy_predictor_kernel_size, energy_predictor_filter_size,
                 p_energy_predictor_dropout, energy_predictor_n_layers,
                 energy_embedding_kernel_size,
                 cwt_conditioning,
                 cwt_predictor_kernel_size, cwt_predictor_filter_size,
                 p_cwt_predictor_dropout, cwt_predictor_n_layers,
                 cwt_embedding_kernel_size, cwt_continuous,
                 n_speakers, speaker_emb_weight, pitch_conditioning_formants=1):
        super(FastPitch, self).__init__()

        self.encoder = FFTransformer(
            n_layer=in_fft_n_layers, n_head=in_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=in_fft_d_head,
            d_inner=in_fft_conv1d_filter_size,
            kernel_size=in_fft_conv1d_kernel_size,
            dropout=p_in_fft_dropout,
            dropatt=p_in_fft_dropatt,
            dropemb=p_in_fft_dropemb,
            embed_input=True,
            d_embed=symbols_embedding_dim,
            n_embed=n_symbols,
            padding_idx=padding_idx)

        if n_speakers > 1:
            self.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim)
        else:
            self.speaker_emb = None
        self.speaker_emb_weight = speaker_emb_weight

        self.duration_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=dur_predictor_filter_size,
            kernel_size=dur_predictor_kernel_size,
            dropout=p_dur_predictor_dropout, n_layers=dur_predictor_n_layers
        )

        self.decoder = FFTransformer(
            n_layer=out_fft_n_layers, n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim
        )

        # pitch conditioning --------modified---------
        self.pitch_conditioning = pitch_conditioning
        if pitch_conditioning:
            print("Pitch Predictor")
            self.pitch_predictor = TemporalPredictor(
                in_fft_output_size,
                filter_size=pitch_predictor_filter_size,
                kernel_size=pitch_predictor_kernel_size,
                dropout=p_pitch_predictor_dropout, n_layers=pitch_predictor_n_layers,
                n_predictions=pitch_conditioning_formants
            )
            # (in_fft_output_size=384, filter_size=256, kernel_size=3, dropout=0.1, n_layers=2, n_predictions=1)

            self.pitch_emb = nn.Conv1d(
                pitch_conditioning_formants, symbols_embedding_dim,
                kernel_size=pitch_embedding_kernel_size,
                padding=int((pitch_embedding_kernel_size - 1) / 2))
            # (1, symbols_embedding_dim=384, kernel_size=3, padding=1)

        # Store values precomputed for training data within the model
        self.register_buffer('pitch_mean', torch.zeros(1))
        self.register_buffer('pitch_std', torch.zeros(1))

        # -------------modified--------------
        self.cwt_conditioning = cwt_conditioning
        self.cwt_continuous = cwt_continuous
        if cwt_conditioning:
            print("Prominence Predictor")
            if cwt_continuous:
                self.cwt_predictor = TemporalPredictor(
                    in_fft_output_size,
                    filter_size=cwt_predictor_filter_size,
                    kernel_size=cwt_predictor_kernel_size,
                    dropout=p_cwt_predictor_dropout,
                    n_layers=cwt_predictor_n_layers,
                    n_predictions=1)
                print("cwt embedding")

                self.cwt_emb = nn.Conv1d(
                    1, symbols_embedding_dim,
                    kernel_size=cwt_embedding_kernel_size,
                    padding=int((cwt_embedding_kernel_size - 1) / 2))  # for continuous label
                # symbols_embedding_dim=384, filter_size=256, kernel_size=3

            # self.cwt_emb = nn.Embedding(1, symbols_embedding_dim)  # for categorical label

        self.energy_conditioning = energy_conditioning
        if energy_conditioning:
            print("Energy Predictor")
            self.energy_predictor = TemporalPredictor(
                in_fft_output_size,
                filter_size=energy_predictor_filter_size,
                kernel_size=energy_predictor_kernel_size,
                dropout=p_energy_predictor_dropout,
                n_layers=energy_predictor_n_layers,
                n_predictions=1
            )

            self.energy_emb = nn.Conv1d(
                1, symbols_embedding_dim,
                kernel_size=energy_embedding_kernel_size,
                padding=int((energy_embedding_kernel_size - 1) / 2))

        self.proj = nn.Linear(out_fft_output_size, n_mel_channels, bias=True)

        self.attention = ConvAttention(
            n_mel_channels, 0, symbols_embedding_dim,
            use_query_proj=True, align_query_enc_type='3xconv')

        print("init done")

    def binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer receive a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = torch.zeros_like(attn)
            for ind in range(b_size):
                hard_attn = mas_width1(
                    attn_cpu[ind, 0, :out_lens[ind], :in_lens[ind]])
                attn_out[ind, 0, :out_lens[ind], :in_lens[ind]] = torch.tensor(
                    hard_attn, device=attn.get_device())
        return attn_out

    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(),
                             out_lens.cpu().numpy(), width=1)
        return torch.from_numpy(attn_out).to(attn.get_device())

    def forward(self, inputs, use_gt_pitch=True, use_gt_cwt=True, pace=1.0, max_duration=75):

        (inputs, input_lens, mel_tgt, mel_lens, pitch_dense, energy_dense,
         speaker, attn_prior, audiopaths, cwt_tgt) = inputs  # --------modified--------
        # print(f'inputs shape: {inputs.shape}')
        # print(f'cwt_tgt shape: {cwt_tgt.shape}')
        print(f'mel_tgt shape: {mel_tgt.shape}')

        # inputs: [16, 140] [batch_size, text_len]
        # mel_tgt: [batch_size, mel-channel, mel_len]
        # spk_emb: None
        # enc_out: [16, 140, 384] [batch_size, text_len, output_dim]
        # enc_mask: [16, 140, 1]
        # text_emb: [16, 140, 384] [batch_size, text_len, output_dim]
        # log_dur_pred: [16, 140] [batch_size, text_len]
        # pitch_pred: [16, 1, 140] [batch_size, num_formants, text_len]
        # energy_pred: [16, 140] [batch_size, text_len]

        mel_max_len = mel_tgt.size(2)

        # Calculate speaker embedding
        if self.speaker_emb is None:
            spk_emb = 0
        else:
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)  # forward() in transformer.py
        # print(f'enc_out shape: {enc_out.shape}')
        # print(f'enc_mask shape: {enc_mask.shape}')

        # Alignment
        text_emb = self.encoder.word_emb(inputs)
        # print(f'text_emb shape: {text_emb.shape}')

        # make sure to do the alignments before folding
        attn_mask = mask_from_lens(input_lens)[..., None] == 0
        # attn_mask should be 1 for unused timesteps in the text_enc_w_spkvec tensor

        attn_soft, attn_logprob = self.attention(
            mel_tgt, text_emb.permute(0, 2, 1), mel_lens, attn_mask,
            key_lens=input_lens, keys_encoded=enc_out, attn_prior=attn_prior)
        # torch.permute() > alter the dimension of tensor
        # forward(self, queries, keys, query_lens, mask=None, key_lens=None, keys_encoded=None, attn_prior=None)
        # queries=mel_tgt, keys=text_emb.permute(0, 2, 1), query_lens=mel_lens, mask=attn_mask

        attn_hard = self.binarize_attention_parallel(
            attn_soft, input_lens, mel_lens)

        # Viterbi --> durations
        attn_hard_dur = attn_hard.sum(2)[:, 0, :] # 2 refers to the axis
        dur_tgt = attn_hard_dur

        assert torch.all(torch.eq(dur_tgt.sum(dim=1), mel_lens))

        # Predict prominence -------------modified--------------
        if self.cwt_conditioning:
            print("cwt")
            cwt_pred = self.cwt_predictor(enc_out, enc_mask).permute(0, 2, 1)
            print(f'cwt_pred shape: {cwt_pred.shape}')  # [batch_size, 1, text_len], predicting continuous number now
            # print(cwt_pred)
            if use_gt_cwt and cwt_tgt is not None:
                cwt_tgt = cwt_tgt.unsqueeze(1)  # [batch_size, 1, text_len]
                # print(f'cwt_tgt shape: {cwt_tgt.shape}')
                cwt_emb = self.cwt_emb(cwt_tgt)
            else:
                cwt_emb = self.cwt_emb(cwt_pred)
            enc_out = enc_out + cwt_emb.transpose(1, 2)
        else:
            cwt_pred = None

        # Predict durations
        log_dur_pred = self.duration_predictor(enc_out, enc_mask).squeeze(-1)
        print(f'log_dur_pred shape: {log_dur_pred.shape}')
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        # Predict pitch (add pitch conditioning)
        if self.pitch_conditioning:
            print("Model is predicting pitch")
            pitch_pred = self.pitch_predictor(enc_out, enc_mask).permute(0, 2, 1)
            print(f'pitch_pred shape: {pitch_pred.shape}')  # [batch_size, num_formants, text_len]

            # Average pitch over characters
            pitch_tgt = average_pitch(pitch_dense, dur_tgt)

            if use_gt_pitch and pitch_tgt is not None:
                pitch_emb = self.pitch_emb(pitch_tgt)
            else:
                pitch_emb = self.pitch_emb(pitch_pred)
            enc_out = enc_out + pitch_emb.transpose(1, 2)  # [batch_size, mel_len, num_formants]
        else:
            pitch_pred = None
            pitch_tgt = None

        # Predict energy
        if self.energy_conditioning:
            energy_pred = self.energy_predictor(enc_out, enc_mask).squeeze(-1)
            print(f'energy_pred shape: {energy_pred.shape}')  # [batch_size, mel_len]
            # Average energy over characters
            energy_tgt = average_pitch(energy_dense.unsqueeze(1), dur_tgt)
            energy_tgt = torch.log(1.0 + energy_tgt)
            print(f'energy_tgt shape {energy_tgt.shape}')
            energy_emb = self.energy_emb(energy_tgt)
            energy_tgt = energy_tgt.squeeze(1)
            enc_out = enc_out + energy_emb.transpose(1, 2)
        else:
            energy_pred = None
            energy_tgt = None

        len_regulated, dec_lens = regulate_len(
            dur_tgt, enc_out, pace, mel_max_len)

        # Output FFT
        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)  # dec for decoder
        mel_out = self.proj(dec_out)
        # print(f'mel_out shape: {mel_out.shape}')
        return (mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred,
                pitch_tgt, energy_pred, energy_tgt, attn_soft, attn_hard,
                attn_hard_dur, attn_logprob, cwt_pred, cwt_tgt)  # add cwt_tgt and cwt_pred

    def infer(self, inputs, pace=1.0, cwt_tgt=None, dur_tgt=None, pitch_tgt=None,
              energy_tgt=None, pitch_transform=None, max_duration=75,
              speaker=0):  # add cwt_tgt=None

        if self.speaker_emb is None:
            spk_emb = 0
        else:
            speaker = (torch.ones(inputs.size(0)).long().to(inputs.device)
                       * speaker)
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

        # Predict cwt
        if self.cwt_conditioning:
            if cwt_tgt is None:
                cwt_pred = self.cwt_predictor(enc_out, enc_mask).permute(0, 2, 1)
                cwt_emb = self.cwt_emb(cwt_pred)
            else:
                cwt_tgt = cwt_tgt.unsqueeze(1)
                cwt_emb = self.cwt_emb(cwt_tgt)
            enc_out = enc_out + cwt_emb.transpose(1, 2)
        else:
            cwt_pred = None

        # Predict durations
        log_dur_pred = self.duration_predictor(enc_out, enc_mask).squeeze(-1)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        # Pitch over chars -------modified-------
        if self.pitch_conditioning:

            pitch_pred = self.pitch_predictor(enc_out, enc_mask).permute(0, 2, 1)

            if pitch_transform is not None:
                if self.pitch_std[0] == 0.0:
                    # XXX LJSpeech-1.1 defaults
                    mean, std = 218.14, 67.24
                else:
                    mean, std = self.pitch_mean[0], self.pitch_std[0]
                pitch_pred = pitch_transform(pitch_pred, enc_mask.sum(dim=(1,2)),
                                             mean, std)
            if pitch_tgt is None:
                pitch_emb = self.pitch_emb(pitch_pred).transpose(1, 2)
            else:
                pitch_emb = self.pitch_emb(pitch_tgt).transpose(1, 2)
            enc_out = enc_out + pitch_emb
        else:
            pitch_pred = None

        # Predict energy
        if self.energy_conditioning:

            if energy_tgt is None:
                energy_pred = self.energy_predictor(enc_out, enc_mask).squeeze(-1)
                energy_emb = self.energy_emb(energy_pred.unsqueeze(1)).transpose(1, 2)
            else:
                energy_emb = self.energy_emb(energy_tgt).transpose(1, 2)

            enc_out = enc_out + energy_emb
        else:
            energy_pred = None

        len_regulated, dec_lens = regulate_len(
            dur_pred if dur_tgt is None else dur_tgt,
            enc_out, pace, mel_max_len=None)

        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)
        # mel_lens = dec_mask.squeeze(2).sum(axis=1).long()
        mel_out = mel_out.permute(0, 2, 1)  # For inference.py
        return mel_out, dec_lens, dur_pred, pitch_pred, energy_pred, cwt_pred
