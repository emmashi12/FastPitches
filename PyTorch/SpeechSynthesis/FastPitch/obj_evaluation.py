import re
import glob, os
import pandas as pd
import torch
from pathlib import Path


def measure_performance(TN, FN, FP, TP):
    accuracy = (TN + TP) * 1.0 / (TN + FN + FP + TP)
    precision = TP * 1.0 / (TP + FP)
    recall = TP * 1.0 / (TP + FN)
    f_value = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, f_value


in_filepath1 = '/Users/emmashi/Desktop/PB_tgt_prom'
in_filepath2 = '/Users/emmashi/Desktop/PB_pred_prom'

total_tgt = []
total_pred = []

os.chdir(in_filepath1)
for file in glob.glob("*.pt"):
    tgt = torch.load(file, map_location=torch.device('cpu'))
    tgt_list = tgt.tolist()
    # print(f'tgt_list: {tgt_list}')
    total_tgt.extend(tgt_list)

print(len(total_tgt))
# print(total_tgt[:3])
# print([len(i) for i in total_tgt])

os.chdir(in_filepath2)
for file in glob.glob("*.pt"):
    pred = torch.load(file, map_location=torch.device('cpu'))
    pred_list = pred.tolist()
    # print(f'pred_list: {pred_list}')
    total_pred.extend(pred_list)

# print(total_pred[:3])
# print([len(i) for i in total_pred])

TN_1 = 0
FN_1 = 0
FP_1 = 0
TP_1 = 0
for i, j in enumerate(total_tgt):
    if j == 1 and total_pred[i] == 1:
        TP_1 += 1
    elif j == 1 and total_pred[i] != 1:
        FN_1 += 1
    elif j != 1 and total_pred[i] == 1:
        FP_1 += 1
    elif j != 1 and total_pred[i] != 1:
        TN_1 += 1

print(TP_1)
print(FN_1)
print(FP_1)
print(TN_1)

print(TP_1+FN_1+FP_1+TN_1)

acc1, prec1, rec1, f1 = measure_performance(TN_1, FN_1, FP_1, TP_1)
print(f'FP_P+B <p1>:\naccuracy:{acc1}, precision:{prec1}, recall:{rec1}, f-value:{f1}')

TN_2 = 0
FN_2 = 0
FP_2 = 0
TP_2 = 0
for i, j in enumerate(total_tgt):
    if j == 2 and total_pred[i] == 2:
        TP_2 += 1
    elif j == 2 and total_pred[i] != 2:
        FN_2 += 1
    elif j != 2 and total_pred[i] == 2:
        FP_2 += 1
    elif j != 2 and total_pred[i] != 2:
        TN_2 += 1
acc2, prec2, rec2, f2 = measure_performance(TN_2, FN_2, FP_2, TP_2)
print(f'FP_P+B <p2>:\naccuracy:{acc2}, precision:{prec2}, recall:{rec2}, f-value:{f2}')

TN_3 = 0
FN_3 = 0
FP_3 = 0
TP_3 = 0
for i, j in enumerate(total_tgt):
    if j == 3 and total_pred[i] == 3:
        TP_3 += 1
    elif j == 3 and total_pred[i] != 3:
        FN_3 += 1
    elif j != 3 and total_pred[i] == 3:
        FP_3 += 1
    elif j != 3 and total_pred[i] != 3:
        TN_3 += 1
acc3, prec3, rec3, f3 = measure_performance(TN_3, FN_3, FP_3, TP_3)
print(f'FP_P+B <p3>:\naccuracy:{acc3}, precision:{prec3}, recall:{rec3}, f-value:{f3}')