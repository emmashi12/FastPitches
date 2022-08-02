import pandas as pd
import glob, os


def load_file(in_file=None):
    column_names = ['id', 'start_time', 'end_time', 'unit', 'p_strength', 'b_strength']
    data = pd.read_csv(in_file, names=column_names, header=None, delimiter='\t')
    prom = data['p_strength'].max()
    #print(data['p_strength'].idxmax())
    return data, prom


def get_boundary_label(value):
    if value <= 0.5:
        return "b0"
    elif value > 1:
        return "b2"
    else:
        return "b1"


def get_prominence_label(value, data):
    if value == data:
        return 2
    else:
        return 1


def get_prominence_label_3C(value):
    if value < 0.6:
        return 1
    elif value >= 1.2:
        return 3
    else:
        return 2


def write_csv_file(out_filepath, data):
    data.to_csv(out_filepath, sep='\t')


in_filepath = '/Users/emmashi/Desktop/my_corpus'
os.chdir(in_filepath)
cwd = os.getcwd()
print("Current working directory is:", cwd)

head, tail = os.path.split(in_filepath)
out_filepath = head + '/labelled_file_3C/'
os.makedirs(out_filepath, exist_ok=True)

for file in glob.glob('*.prom'):
    data, prom = load_file(file)
    # data['b_label'] = data.apply(lambda x: get_boundary_label(x.b_strength), axis=1)
    data['p_label'] = data.apply(lambda x: get_prominence_label_3C(x.p_strength), axis=1)
    #print(data)
    #print(out_filepath + file)
    write_csv_file(out_filepath + file, data)

