import re
import os


def extract_text(infile="testone.txt", outfile=None):
    out_filepath = '/Users/emmashi/Desktop/FastPitches_notes/PyTorch/SpeechSynthesis/FastPitch/filelists/text'
    os.makedirs(out_filepath, exist_ok=True)
    with open(infile) as file:
        for l in file:
            #print(type(l))
            #print(l)
            matchline = re.match('(.*)\/(.*)\.wav\|(.*)', l)
            outname= matchline.group(2)
            #print(outname)
            #print(outname + ".lab")
            text = matchline.group(3)
            if outfile:
                os.chdir(out_filepath)
                with open(outname + ".lab", 'a') as f:
                    f.write('{}'.format(text))


extract_text(outfile=True)