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

import functools
import json
import re
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import betabinom

import common.layers as layers
from common.text.text_processing import TextProcessing
from common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu


class BetaBinomialInterpolator:
    """Interpolates alignment prior matrices to save computation.

    Calculating beta-binomial priors is costly. Instead cache popular sizes
    and use img interpolation to get priors faster.
    """
    def __init__(self, round_mel_len_to=100, round_text_len_to=20):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        self.bank = functools.lru_cache(beta_binomial_prior_distribution)

    def round(self, val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = self.round(w, to=self.round_mel_len_to)
        bh = self.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling=1.0):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P) #return an ndarray of evenly spaced values within a given interval
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b) #P,a,b as shape parameters
        mel_i_prob = rv.pmf(x) #pmf is probability mass function
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


def estimate_pitch(wav, mel_len, method='pyin', normalize_mean=None,
                   normalize_std=None, n_formants=1):

    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std) #use torch.tensor() to construct a tensor

    if method == 'pyin':

        snd, sr = librosa.load(wav) #sr is sampling rate   *** snd is np.ndarray of audio time series.
        #load audio files as a floating point time series.
        pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
            snd, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), frame_length=1024)
        #F0 estimation using probabilistic Yin.
        #pitch_mel is time series of fundamental frequencies in Hertz.
        #f0: np.ndarray[shape=(..., n_frames)]
        assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0  #mel_len is the number of mel frames

        pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
        pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
        pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1))) #torch.nn.functional as F

        if n_formants > 1:
            raise NotImplementedError

    else:
        raise ValueError

    pitch_mel = pitch_mel.float()

    if normalize_mean is not None:
        assert normalize_std is not None
        pitch_mel = normalize_pitch(pitch_mel, normalize_mean, normalize_std)

    return pitch_mel


def normalize_pitch(pitch, mean, std):
    zeros = (pitch == 0.0)
    pitch -= mean[:, None]
    pitch /= std[:, None]
    pitch[zeros] = 0.0
    return pitch


def upsampling_label(cwt_tensor, text_info):
    text_symbols = [x[1] for x in text_info]
    text_words = [x[0] for x in text_info]
    total_symbols = sum(text_symbols)
    tensor_list = cwt_tensor.tolist()

    upsampled = []
    words = re.compile('\w+')  # match for words
    tensor_index = 0

    for c in range(len(text_words)):
        if words.search(text_words[c]):
            t = [tensor_list[tensor_index]] * text_symbols[c]  # upsample label
            upsampled += t
            tensor_index += 1
        else:
            t = [0.0] * text_symbols[c]  # add label for non-words
            upsampled += t
    upsampled = torch.LongTensor(upsampled)
    return upsampled, total_symbols


class TTSDataset(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    #----------modified------------
    def __init__(self,
                 dataset_path,
                 audiopaths_and_text,
                 text_cleaners,
                 n_mel_channels,
                 symbol_set='english_basic',
                 p_arpabet=1.0,
                 n_speakers=1,
                 load_mel_from_disk=True,
                 load_pitch_from_disk=True,
                 load_cwt_prom_from_disk=True,
                 load_cwt_b_from_disk=True,
                 cwt_label=True,
                 b_label=True,
                 pitch_mean=214.72203,  # LJSpeech defaults
                 pitch_std=65.72038,
                 max_wav_value=None,
                 sampling_rate=None,
                 filter_length=None,
                 hop_length=None,
                 win_length=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 prepend_space_to_text=False,
                 append_space_to_text=False,
                 pitch_online_dir=None,
                 betabinomial_online_dir=None,
                 use_betabinomial_interpolator=True,
                 pitch_online_method='pyin',
                 **ignored):

        # Expect a list of filenames
        if type(audiopaths_and_text) is str:
            audiopaths_and_text = [audiopaths_and_text]

        self.dataset_path = dataset_path
        self.audiopaths_and_text = load_filepaths_and_text(
            audiopaths_and_text, dataset_path,
            has_speakers=(n_speakers > 1))  # load_filepaths_and_text is a function from common.utils ***see line 42
        self.load_mel_from_disk = load_mel_from_disk
        if not load_mel_from_disk:
            self.max_wav_value = max_wav_value
            self.sampling_rate = sampling_rate
            self.stft = layers.TacotronSTFT(
                filter_length, hop_length, win_length,
                n_mel_channels, sampling_rate, mel_fmin, mel_fmax)
        self.load_pitch_from_disk = load_pitch_from_disk

        # ----------load prominence from disk-------------
        self.load_cwt_prom_from_disk = load_cwt_prom_from_disk
        self.cwt_label = cwt_label

        self.load_cwt_b_from_disk = load_cwt_b_from_disk  # for boundary labels
        self.b_label = b_label

        self.prepend_space_to_text = prepend_space_to_text
        self.append_space_to_text = append_space_to_text

        assert p_arpabet == 0.0 or p_arpabet == 1.0, (
            'Only 0.0 and 1.0 p_arpabet is currently supported. '
            'Variable probability breaks caching of betabinomial matrices.')

        if self.cwt_label is True:
            self.tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet, get_count=True)
        else:
            self.tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet, get_count=False)

        self.n_speakers = n_speakers
        self.pitch_tmp_dir = pitch_online_dir
        self.f0_method = pitch_online_method
        self.betabinomial_tmp_dir = betabinomial_online_dir
        self.use_betabinomial_interpolator = use_betabinomial_interpolator

        if use_betabinomial_interpolator:
            self.betabinomial_interpolator = BetaBinomialInterpolator()

        # ---------should modify??-----------
        # expected_columns = (2 + int(load_pitch_from_disk) + (n_speakers > 1))

        assert not (load_pitch_from_disk and self.pitch_tmp_dir is not None)

        # if len(self.audiopaths_and_text[0]) < expected_columns:
        #     raise ValueError(f'Expected {expected_columns} columns in audiopaths file. '
        #                      'The format is <mel_or_wav>|[<pitch>|]<text>[|<speaker_id>]')
        # #using <mel_or_wav>|[<pitch>|][<prom>|]<text>[|<speaker_id>]
        #
        # if len(self.audiopaths_and_text[0]) > expected_columns:
        #     print('WARNING: Audiopaths file has more columns than expected')

        to_tensor = lambda x: torch.Tensor([x]) if type(x) is float else x
        self.pitch_mean = to_tensor(pitch_mean)
        self.pitch_std = to_tensor(pitch_std)

    def __getitem__(self, index):
        # Separate filename and text
        # -----------modified------------
        if self.n_speakers > 1:
            audiopath = self.audiopaths_and_text[index]['wav']  # modify to key of dictionary
            audiopath = self.dataset_path + "/" + audiopath
            text = self.audiopaths_and_text[index]['text']
            speaker = self.audiopaths_and_text[index]['speaker']
            speaker = int(speaker)
        else:
            audiopath = self.audiopaths_and_text[index]['wav']  # modify to key of dictionary
            audiopath = self.dataset_path + "/" + audiopath
            # print(f'audiopath: {audiopath}')
            text = self.audiopaths_and_text[index]['text']
            speaker = None

        mel = self.get_mel(audiopath)  # method see line 238
        # print(f'mel shape: {mel.shape}')
        text, text_info = self.get_text(text)  # text_info is tuple (word, number)
        # print(f'text shape: {text.shape}')
        pitch = self.get_pitch(index, mel.size(-1))
        # print(f'pitch shape: {pitch.shape}')
        energy = torch.norm(mel.float(), dim=0, p=2)
        attn_prior = self.get_prior(index, mel.shape[1], text.shape[0])

        if self.cwt_label is True:
            cwt_prom_tensor = self.get_prom_label(index, text_info)
            # print(f'cwt shape: {cwt_tensor.shape}')
        else:
            cwt_prom_tensor = None
        # text_len = text.shape[0]     mel_len = mel.shape[1]

        if self.b_label is True:
            cwt_b_tensor = self.get_b_label(index, text_info)
        else:
            cwt_b_tensor = None

        assert pitch.size(-1) == mel.size(-1)
        # sanity check for number of pitch values and number of frames

        # No higher formants?
        if len(pitch.size()) == 1:
            pitch = pitch[None, :]

        return (text, mel, len(text), pitch, energy, speaker, attn_prior,
                audiopath, cwt_prom_tensor, cwt_b_tensor)  # ----------modify-----------

    def __len__(self):
        return len(self.audiopaths_and_text)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate)) #see line 174 for stft
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm,
                                                 requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.load(filename)
            # assert melspec.size(0) == self.stft.n_mel_channels, (
            #     'Mel dimension mismatch: given {}, expected {}'.format(
            #         melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        # word input, phoneme tensor output
        if self.cwt_label:
            text, text_info = self.tp.encode_text(text)
            # see line 190, get a list of a sequence of index representing the text
        else:
            text = self.tp.encode_text(text)
            text_info = None

        if self.prepend_space_to_text:
            space = [self.tp.encode_text("A A")[1]]
            text = space + text

        if self.append_space_to_text:
            space = [self.tp.encode_text("A A")[1]]
            text = text + space

        return torch.LongTensor(text), text_info

    def get_prior(self, index, mel_len, text_len):

        if self.use_betabinomial_interpolator:
            return torch.from_numpy(self.betabinomial_interpolator(mel_len,
                                                                   text_len))

        if self.betabinomial_tmp_dir is not None:
            audiopath = self.audiopaths_and_text[index]['wav']
            fname = Path(audiopath).relative_to(self.dataset_path) if self.dataset_path else Path(audiopath)
            fname = fname.with_suffix('.pt')
            cached_fpath = Path(self.betabinomial_tmp_dir, fname) #what is cached_fpath?

            if cached_fpath.is_file():
                return torch.load(cached_fpath)

        attn_prior = beta_binomial_prior_distribution(text_len, mel_len)

        if self.betabinomial_tmp_dir is not None:
            cached_fpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(attn_prior, cached_fpath)

        return attn_prior

    def get_pitch(self, index, mel_len=None):
        # -----------modified----------
        audiopath = self.audiopaths_and_text[index]['wav']
        audiopath = self.dataset_path + '/' + audiopath

        if self.n_speakers > 1:
            spk = self.audiopaths_and_text[index]['speaker']
        else:
            spk = 0

        if self.load_pitch_from_disk:
            pitchpath = self.audiopaths_and_text[index]['pitch']
            pitchpath = self.dataset_path + '/' +pitchpath
            pitch = torch.load(pitchpath)
            if self.pitch_mean is not None:
                # print("doing pitch normalization")
                assert self.pitch_std is not None
                pitch = normalize_pitch(pitch, self.pitch_mean, self.pitch_std)
            return pitch

        if self.pitch_tmp_dir is not None:
            fname = Path(audiopath).relative_to(self.dataset_path) if self.dataset_path else Path(audiopath)
            fname_method = fname.with_suffix('.pt')
            cached_fpath = Path(self.pitch_tmp_dir, fname_method)
            if cached_fpath.is_file():
                return torch.load(cached_fpath)

        # No luck so far - calculate
        wav = audiopath
        if not wav.endswith('.wav'):
            wav = re.sub('/mels/', '/wavs/', wav)
            wav = re.sub('.pt$', '.wav', wav)

        pitch_mel = estimate_pitch(wav, mel_len, self.f0_method,
                                   self.pitch_mean, self.pitch_std)

        if self.pitch_tmp_dir is not None and not cached_fpath.is_file():
            cached_fpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pitch_mel, cached_fpath)

        return pitch_mel

    # ------------get_prominence------------
    def get_prom_label(self, index, text_info):
        if self.load_cwt_prom_from_disk:
            prompath = self.audiopaths_and_text[index]['prom']
            prompath = self.dataset_path + '/' + prompath
            cwt_prom = torch.load(prompath)
            # cwt_prom_list = prom.tolist()
            # print(cwt_list)
            # print(f'text info: {text_info}')

            cwt_prom_tensor, total_symbols = upsampling_label(cwt_prom, text_info)

            # cwt_prom_tensor = torch.Tensor(upsampled)  # for continuous label
            # print(cwt_tensor.type())
            assert list(cwt_prom_tensor.size())[0] == total_symbols

            return cwt_prom_tensor

    def get_b_label(self, index, text_info):
        if self.load_cwt_b_from_disk:
            bpath = self.audiopaths_and_text[index]['boundary']
            bpath = self.dataset_path + '/' + bpath
            cwt_b = torch.load(bpath)
            # cwt_b_list = b.tolist()

            cwt_b_tensor, total_symbols = upsampling_label(cwt_b, text_info)
            assert list(cwt_b_tensor.size())[0] == total_symbols

            return cwt_b_tensor


class TTSCollate:
    """Zero-pads model inputs and targets based on number of frames per step"""

    # return (text, mel, len(text), pitch, energy, speaker, attn_prior, audiopath, cwt_tensor)

    def __call__(self, batch):
        """Collate training batch from normalized text and mel-spec"""
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # padding for prominence label tensor -------------modified---------------
        if batch[0][8] is not None:
            # cwt_prom_padded = torch.FloatTensor(len(batch), max_input_len)  # for continuous label
            cwt_prom_padded = torch.LongTensor(len(batch), max_input_len)  # for categorical label
            cwt_prom_padded.zero_()
            for i in range(len(ids_sorted_decreasing)):
                cwt_prom = batch[ids_sorted_decreasing[i]][8]
                cwt_prom_padded[i, :cwt_prom.shape[0]] = cwt_prom
            # print(f'cwt_padded: {cwt_padded.shape}')
        else:
            cwt_prom_padded = None

        if batch[0][9] is not None:  # padding for boundary labels
            cwt_b_padded = torch.LongTensor(len(batch), max_input_len)  # for categorical label
            cwt_b_padded.zero_()
            for i in range(len(ids_sorted_decreasing)):
                cwt_b = batch[ids_sorted_decreasing[i]][9]
                cwt_b_padded[i, :cwt_b.shape[0]] = cwt_b
        else:
            cwt_b_padded = None

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # Include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        n_formants = batch[0][3].shape[0]
        pitch_padded = torch.zeros(mel_padded.size(0), n_formants,
                                   mel_padded.size(2), dtype=batch[0][3].dtype)
        energy_padded = torch.zeros_like(pitch_padded[:, 0, :])

        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]][3]
            energy = batch[ids_sorted_decreasing[i]][4]
            pitch_padded[i, :, :pitch.shape[1]] = pitch
            energy_padded[i, :energy.shape[0]] = energy

        if batch[0][5] is not None:
            speaker = torch.zeros_like(input_lengths)
            for i in range(len(ids_sorted_decreasing)):
                speaker[i] = batch[ids_sorted_decreasing[i]][5]
        else:
            speaker = None

        attn_prior_padded = torch.zeros(len(batch), max_target_len,
                                        max_input_len)
        attn_prior_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            prior = batch[ids_sorted_decreasing[i]][6]
            attn_prior_padded[i, :prior.size(0), :prior.size(1)] = prior

        # Count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)

        audiopaths = [batch[i][7] for i in ids_sorted_decreasing]

        return (text_padded, input_lengths, mel_padded, output_lengths, len_x,
                pitch_padded, energy_padded, speaker, attn_prior_padded,
                audiopaths, cwt_prom_padded, cwt_b_padded)  # ---------modified------------


def batch_to_gpu(batch):
    (text_padded, input_lengths, mel_padded, output_lengths, len_x,
     pitch_padded, energy_padded, speaker, attn_prior, audiopaths, cwt_prom_padded, cwt_b_padded) = batch
    # --------modified---------
    # _, input_lens, mels, mel_lens, _, pitch, _, _, attn_prior, fpaths, cwt = batch

    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    pitch_padded = to_gpu(pitch_padded).float()
    energy_padded = to_gpu(energy_padded).float()
    attn_prior = to_gpu(attn_prior).float()
    # cwt_prom_padded = to_gpu(cwt_prom_padded).float()  # for continuous
    cwt_prom_padded = to_gpu(cwt_prom_padded).long()  # for categorical
    cwt_b_padded = to_gpu(cwt_b_padded).long()  # for categorical
    if speaker is not None:
        speaker = to_gpu(speaker).long()

    # Alignments act as both inputs and targets - pass shallow copies
    x = [text_padded, input_lengths, mel_padded, output_lengths,
         pitch_padded, energy_padded, speaker, attn_prior, audiopaths, cwt_prom_padded, cwt_b_padded]
    y = [mel_padded, input_lengths, output_lengths, cwt_prom_padded, cwt_b_padded]  # something we predict in the model
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)
