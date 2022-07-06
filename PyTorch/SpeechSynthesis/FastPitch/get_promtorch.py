import glob, os
import pandas as pd
import torch

# def get_cwt_torch(in_filepath=None):
#     os.chdir(in_filepath)
#     for file in glob.glob("*.prom")


tst = pd.read_csv('./filelists/labelled_file/LJ001-0110.prom', delimiter='\t')
print(tst)
prom = tst['p_label'].to_list()
print(prom)

p_tensor = torch.Tensor(prom)
print(p_tensor)
