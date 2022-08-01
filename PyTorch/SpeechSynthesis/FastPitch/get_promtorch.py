import re
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


def get_controlled_tensor(list, file, outpath):
    tst = pd.read_csv(file, delimiter='\t')
    prom = tst['prom'].to_list()
    name_list = []
    for l in prom:
        print(type(l))
        matchline = re.match('(.*)\/(.*)', l)
        fname = matchline.group(2)
        name_list.append(fname)
    for j, p in enumerate(list):
        prom_tensor = torch.LongTensor(p)
        print('boundary tensor:', prom_tensor)
        fpath = Path(outpath, name_list[j])
        torch.save(prom_tensor, fpath)


# in_filepath = '/Users/emmashi/Desktop/labelled_file_3C'
in_filepath = '/Users/emmashi/Desktop/control_P+B.tsv'
# os.chdir(in_filepath)
head, tail = os.path.split(in_filepath)

out_filepath = head + '/test-PB1/'
os.makedirs(out_filepath, exist_ok=True)
#
# for file in glob.glob("*.prom"):
#     get_cwt_tensor(file, out_filepath)

# listening test for accent
accent = [[3, 1, 1, 1], [1, 3, 1, 1], [1, 1, 1, 3],
          [3, 1, 1, 1], [1, 3, 1, 1], [1, 1, 1, 3],
          [1, 3, 1, 1, 1], [1, 1, 3, 1, 1], [1, 1, 1, 1, 3],
          [3, 1, 1, 1], [1, 3, 1, 1], [1, 1, 1, 3],
          [1, 3, 1, 1, 1], [1, 1, 3, 1, 1], [1, 1, 1, 1, 3],
          [3, 1, 1, 1], [1, 3, 1, 1], [1, 1, 1, 3],
          [3, 1, 1, 1, 1], [1, 3, 1, 1, 1], [1, 1, 1, 1, 3],
          [3, 1, 1, 1], [1, 3, 1, 1], [1, 1, 1, 3],
          [1, 3, 1, 1, 1], [1, 1, 3, 1, 1], [1, 1, 1, 1, 3],
          [3, 1, 1, 1], [1, 3, 1, 1], [1, 1, 1, 3]]

# listening test for syntactic ambiguous sentences
boundary1 = [[1,1,3,3], [3,1,1,3],
             [1,1,3,1,1,1,3,1,1,2,1,1,1,3], [1,1,3,1,1,1,1,3,1,2,1,1,1,3],
             [1,1,1,1,1,1,3,1,2,1,1,1,3], [1,1,1,1,2,3,1,1,2,1,1,1,3],
             [1,1,1,1,1,3,1,1,2,1,3], [1,1,1,3,1,1,1,1,2,1,3],
             [3,1,1,2,1,1,1,1,2,3,1,1,3], [3,1,1,2,1,1,1,1,1,1,3,1,3]]

emphasis1 = [[1,1,3,3], [3,1,1,3],
             [1,1,3,1,1,1,3,1,1,2,1,1,1,3], [1,1,3,1,1,1,1,3,1,2,1,1,1,3],
             [1,1,1,1,1,1,3,1,2,1,1,1,3], [1,1,1,1,2,3,1,1,2,1,1,1,3],
             [1,1,1,1,1,3,1,1,2,1,3], [1,1,1,3,1,1,1,1,2,1,3],
             [3,1,1,2,1,1,1,1,2,3,1,1,3], [3,1,1,2,1,1,1,1,1,1,3,1,3]]

test_B = [[3,1,1,1], [1,3,1,1], [1,1,3,1], [1,1,1,3]]


prominence = [[1,1,1,2,3,2,2,1,2,1,1,2],[1,1,1,2,2,3,2,1,2,1,1,2],[2,1,2,2,2,2,2,2,2,3],[2,1,2,2,3,2,2,2,2,3]]
boundary = [[1,1,1,1,3,1,1,1,1,1,1,3],[1,1,1,3,1,1,1,1,1,1,1,3],[1,1,2,1,1,1,3,1,1,3],[1,1,1,3,1,1,1,1,1,3]]

get_controlled_tensor(prominence, in_filepath, out_filepath)