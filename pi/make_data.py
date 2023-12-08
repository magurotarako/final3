import pickle
import sys
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


#いくつoutputファイルを読み込むかとalpha値をコマンドライン引数で指定（python3 make_data.py number lastNumber alpha match border fBorderの形で指定）
def get_params():
    number = int(sys.argv[1])
    lastNumber = int(sys.argv[2])
    alpha = float(sys.argv[3])
    match = int(sys.argv[4])
    border = int(sys.argv[5])
    fBorder = int(sys.argv[6])
    return number, lastNumber, alpha, match, border, fBorder

def get_outputs(number, lastNumber, alpha, match, border):
    data = {}
    N = lastNumber - number + 1
    for i in range(N):
        with open("output_seed{:04d}_alpha{}_match{}_border{}.pkl".format(i + number, alpha, match, border), 'rb') as tf:
            output = pickle.load(tf)
            #print(output)
            for key, value in output.items():
                if key not in data:
                    data[key] = value
                else:
                    v_0, v_1 = value[0], value[1]
                    a = data[key]
                    a[0] += v_0
                    a[1] += v_1
                    data[key] = [a[0], a[1]]
    return data

def make_data(data, fBorder):
    final_data = {}
    for key, value in data.items():
        if value[0] >= fBorder:
            final_data[key] = value[1] / value[0]
    return final_data


number, lastNumber, alpha, match, border, fBorder = get_params()
data = get_outputs(number, lastNumber, alpha, match, border)
final_data = make_data(data, fBorder)
#print(final_data)
with open("data_seed{:04d}to{:04d}_alpha{}_match{}_border{}_fBorder{}.pkl".format(number, lastNumber, alpha, match, border, fBorder), 'wb') as tf:
    pickle.dump(final_data, tf)

