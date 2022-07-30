import re
import os
import csv
import pandas as pd


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


def add_column(infile="/Users/emmashi/Desktop/control_boundary.txt", outfile=None):
    out_filepath = '/Users/emmashi/Desktop'
    with open(infile) as file:
        for l in file:
            # matchline = re.match('(.*)\/(.*)\.wav\|(.*)\|(.*)\|(.*)\|(.*)', l)
            # outname = matchline.group(2)
            # text = matchline.group(6)
            # wavpath = matchline.group(1) + '/' + outname + '.wav'
            # utt_path = 'pitch-prom-cat-' + outname + '.wav'
            matchline = re.match('(.*)\|(.*)\|(.*)', l)
            prompath = matchline.group(1)
            output = matchline.group(2)
            text = matchline.group(3)
            rewrite = prompath + '\t' + output + '\t' + text
            #print(rewrite)
            if outfile:
                os.chdir(out_filepath)
                with open("control_boundary.tsv", 'a') as f:
                    f.write('{}\n'.format(rewrite))


def remove_wav(infile="/Users/emmashi/Desktop/ljs_audio_pitch_prom_text_test.txt", infile2='/Users/emmashi/Desktop/incorrect_label_test.txt', outfile=None):
    out_filepath = '/Users/emmashi/Desktop'
    column_names = ['id', 'text']
    data = pd.read_csv(infile2, names=column_names, header=None, quoting=csv.QUOTE_NONE, delimiter='\t')
    print(data['id'])
    ids = data.id.tolist()
    print(ids)
    print(len(ids))
    with open(infile) as file:
        os.chdir(out_filepath)
        for l in file:
            matchline = re.match('(.*)\|(.*)\|(.*)\|(.*)', l)
            wavpath = matchline.group(1)
            print(wavpath)
            #print(type(wavpath))
            if wavpath not in ids and outfile:
                with open("ljs_audio_pitch_prom_text_test_con.txt", 'a') as f:
                    f.write('{}'.format(l))


def extract_infer_text(infile="/Users/emmashi/Desktop/test_infer.txt", outfile=None):
    out_filepath = '/Users/emmashi/Desktop'
    # os.makedirs(out_filepath, exist_ok=True)
    with open(infile) as file:
        for l in file:
            #print(type(l))
            #print(l)
            matchline = re.match('(.*)\.mp3\s(.*)', l)
            #print(outname)
            #print(outname + ".lab")
            text = matchline.group(2)
            print(text)
            if outfile:
                os.chdir(out_filepath)
                with open("infer_test.txt", 'a') as f:
                    f.write('{}\n'.format(text))


# extract_text(outfile=True)
add_column(outfile=True)
# remove_wav(outfile=True)
# extract_infer_text(outfile=True)