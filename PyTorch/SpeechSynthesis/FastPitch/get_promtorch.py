import glob, os
import pandas as pd
import torch
from pathlib import Path


def get_cwt_tensor(file, outpath):
    fname = Path(file).with_suffix('.pt').name
    fpath = Path(outpath, fname)
    tst = pd.read_csv(file, delimiter='\t')
    # print(tst.dtypes)
    # prom = tst['p_label'].to_list()
    # print("prom label list", prom)
    b_tensor = torch.LongTensor(tst['b_label'].values)
    print("boundary label tensor", b_tensor)
    torch.save(b_tensor, fpath)


in_filepath = '/Users/emmashi/Desktop/labelled_file_3C'
os.chdir(in_filepath)
head, tail = os.path.split(in_filepath)

out_filepath = head + '/boundary/'
os.makedirs(out_filepath, exist_ok=True)

for file in glob.glob("*.prom"):
    get_cwt_tensor(file, out_filepath)

