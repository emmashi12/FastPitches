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
    p_tensor = torch.LongTensor(tst['p_label'].values)
    print("prom label tensor", p_tensor)
    torch.save(p_tensor, fpath)

def store_list(file):
    tst = pd.read_csv(file, delimiter='\t')
    # print(tst.dtypes)
    prom = tst['p_label'].to_list()
    print("prom label list", prom)
    return prom

def get_accuracy(predictor, target):


target_filepath = '/Users/emmashi/Desktop/labelled_file_3C'
os.chdir(target_filepath)
# head, tail = os.path.split(in_filepath)
#
# out_filepath = head + '/cwt_cat_3C/'
# os.makedirs(out_filepath, exist_ok=True)

total_target=[]
total_predict=[]

for file in glob.glob("*.prom"):
    # get_cwt_tensor(file, out_filepath)
    target = store_list(file)
    total_target.extend(target)

baseline_filepath = '/Users/emmashi/Desktop/...'
os.chdir(baseline_filepath)

for file in glob.glob("*.prom"):
    predict = store_list(file)
    total_predict.extend(predict)


