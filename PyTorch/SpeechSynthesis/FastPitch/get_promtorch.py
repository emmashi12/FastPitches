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


# in_filepath = '/Users/emmashi/Desktop/labelled_file_3C'
in_filepath = '/Users/emmashi/Desktop/test_prom_control.tsv'
# os.chdir(in_filepath)
head, tail = os.path.split(in_filepath)

out_filepath = head + '/test_prom_cat/'
os.makedirs(out_filepath, exist_ok=True)
#
# for file in glob.glob("*.prom"):
#     get_cwt_tensor(file, out_filepath)
l1 = [1,1,1,1,3]
l2 = [1,1,1,3,1]
l3 = [3,1,1,1,1]
l4 = [1,3,1,1,1]
l5 = [1,1,1,3,1]
l6 = [1,1,1,1,3]


prom_tensor = torch.LongTensor(l4)
print(prom_tensor)
out_name = out_filepath + '002-1.pt'
torch.save(prom_tensor, out_name)