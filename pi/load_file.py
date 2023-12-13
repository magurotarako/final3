import random
import numpy as np
import pickle
import sys
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

#コマンドライン引数としてファイル名を指定　(python3 geisterML.py data_seed{:04d}to{:04d}_alpha{}_match{}_border{}_fBorder{}.pkl)
def get_file():
    data_name = sys.argv[1]
    return data_name

def get_data(data_name):
    with open('{}'.format(data_name), 'rb') as tf:
        data = pickle.load(tf)
    return data

def main():
    data_name = get_file()
    data = get_data(data_name)
    print(max(list(data.values())))
    print(min(list(data.values())))
    #print(data)
    return

main()
