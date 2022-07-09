import glob, os
import pandas as pd
import torch
from pathlib import Path

def get_cwt_tensor(file, outpath):
    fname = Path(file).with_suffix('.pt').name
    fpath = Path(outpath, fname)
    tst = pd.read_csv(file, delimiter='\t')
    #print(tst)
    prom = tst['p_label'].to_list()
    #print(prom)
    p_tensor = torch.Tensor(prom)
    print(p_tensor)
    torch.save(p_tensor, fpath)


in_filepath = '/Users/emmashi/Desktop/labelled_file'
os.chdir(in_filepath)
head, tail = os.path.split(in_filepath)

out_filepath = head + '/prom_tensor/'
os.makedirs(out_filepath, exist_ok=True)

for file in glob.glob("*.prom"):
    get_cwt_tensor(file, out_filepath)

