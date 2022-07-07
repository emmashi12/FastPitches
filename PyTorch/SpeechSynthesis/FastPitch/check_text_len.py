import re
import torch
from common.text.text_processing import TextProcessing
from common.utils import load_filepaths_and_text

dataset_path = '/exports/eddie/scratch/s2226669/LJSpeech-1.1'
audiopaths_and_text = load_filepaths_and_text("filelists/ljs_audio_pitch_prom_text_train_v3.txt",
                                              dataset_path, has_speakers=False)
tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet, get_count=True)
log_file = open("incorrect_label.txt", "a")



for file in audiopaths_and_text:
    audiopath = audiopaths_and_text[file]['wav']
    text = audiopaths_and_text[file]['text']
    cwt = audiopaths_and_text[file]['prom']

    prom_tensor = torch.load(cwt)

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

