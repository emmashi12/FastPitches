import re
import os


def extract_text(infile="ljs_audio_text.txt", outfile=None):
    out_filepath = '/Users/emmashi/Desktop/wavs'
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


def add_column(infile="ljs_audio_pitch_text_train_v3.txt", outfile=None):
    out_filepath = '/Users/emmashi/Desktop'
    with open(infile) as file:
        for l in file:
            matchline = re.match('(.*)\/(.*)\.wav\|(.*)\|(.*)', l)
            outname = matchline.group(2)
            pitchpath = matchline.group(3)
            text = matchline.group(4)
            wavpath = matchline.group(1) + '/' + outname + '.wav'
            prompath = 'prom/' + outname + '.pt'
            rewrite = wavpath + '|' + pitchpath + '|' + prompath + '|' + text
            #print(rewrite)
            if outfile:
                os.chdir(out_filepath)
                with open("ljs_audio_pitch_prom_text_train_v3.txt", 'a') as f:
                    f.write('{}\n'.format(rewrite))


#extract_text(outfile=True)
add_column(outfile=True)