from statistics import mean
import copy
import random
import numpy as np
import sys
import re
import pickle
import yaml
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def make_board():
    x_1 = random.randint(1, 4)
    x_2 = random.randint(1, 4)
    x_3 = random.randint(1, 4)
    x_4 = random.randint(1, 4)
    x_5 = 36 - (x_1 + x_2 + x_3 + x_4)
    board = [-2] * x_1 + [-1] * x_2 + [1] * x_3 + [2] * x_4 + [0] * x_5
    random.shuffle(board)
    board_n = np.array(board)
    board_x = board_n.reshape([6, 6])
    return board_x

def load_model():
    model = create_model()
    model.load_weights('modelA').expect_partial()
    return model

def create_model():
    model = keras.Sequential ([
        keras.layers.Dense(144),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation ='tanh')
    ])
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    return model

def make_one_hot(board):
    #print(board)
    one_hot = np.zeros(144)
    for i in range(6):
        for j in range(6):
            if board[i][j] == -2:
                one_hot[4 * (i * 6 + j)] = 1
            elif board[i][j] == -1:
                one_hot[4 * (i * 6 + j)] = 1
            elif board[i][j] == 0:
                one_hot[4 * (i * 6 + j) + 1] = 1
            elif board[i][j] == 1:
                one_hot[4 * (i * 6 + j) + 2] = 1
            else:
                one_hot[4 * (i * 6 + j) + 3] = 1
    #print(one_hot)
    return one_hot

def main():
    times = 100
    test_model = load_model()
    boards = []
    predictions = []
    for i in range(times):
        test_board = make_board()
        test_one_hot = make_one_hot(test_board)
        test_test = []
        test_test.append(test_one_hot)
        test_test_test = np.array(test_test)
        prediction = test_model.predict(test_test_test)[0][0]
        boards.append(test_board)
        predictions.append(prediction)
    print("/////////")
    #print("予測値は")
    print(predictions)
    print("/////////")
    ma = max(predictions)
    maxindex = predictions.index(ma)
    #print("最大値とその盤面は")
    print(ma)
    print(boards[maxindex])
    print("/////////")
    mi = min(predictions)
    minindex = predictions.index(mi)
    #print("最小値とその盤面は")
    print(mi)
    print(boards[minindex])
    return

main()