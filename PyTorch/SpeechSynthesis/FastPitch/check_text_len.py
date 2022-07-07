import re
import torch
from common.text.text_processing import TextProcessing
from common.utils import load_filepaths_and_text

dataset_path = '/exports/eddie/scratch/s2226669/LJSpeech-1.1/'
data_path = ['/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226669_Wenjing_Shi/selfversion/PyTorch/SpeechSynthesis/FastPitch/filelists/ljs_audio_pitch_prom_text_train_v3.txt']
audiopaths_and_text = load_filepaths_and_text(data_path, has_speakers=False)
tp = TextProcessing('english_basic', "english_cleaners_v2", p_arpabet=1.0, get_count=True)
log_file = open("incorrect_label.txt", "a")



for file in audiopaths_and_text:
    audiopath = file['wav']
    text = file['text']
    cwt = file['prom']

    prom_tensor = torch.load(dataset_path + cwt)

    sentence, text_info = tp.encode(text)
    characters = re.compile('\w+')
    w_count = 0
    for i in text_info:
        if characters.search(i[0]):
            w_count += 1
        else:
            continue

    if not list(prom_tensor.size())[0] == w_count:
        log_file.write(f'{audiopath}\t{text}\n')

log_file.close()

