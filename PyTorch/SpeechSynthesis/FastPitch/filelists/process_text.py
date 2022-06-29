import re


def extract_text(infile="ljs_audio_text_val.txt", outfile=None):
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
                with open(outname + ".lab", 'a') as f:
                    f.write('{}'.format(text))


extract_text(outfile=True)