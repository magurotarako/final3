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
    board = []
    board_1 = [0, 1, 0, 0, -2, 0]
    board_2 = [0, 0, 1, 0, -1, 0]
    board_3 = [0, 2, 0, -2, 0, 0]
    board_4 = [0, 0, -1, 0, 2, 0]
    board_5 = [0, 0, 2, -2, 0, 0]
    board_6 = [0, 1, 0, 0, 0, 0]
    board.append(board_1)
    board.append(board_2)
    board.append(board_3)
    board.append(board_4)
    board.append(board_5)
    board.append(board_6)
    print(board)
    return np.array(board)

def load_model():
    model = create_model()
    model.load_weights('modelA').expect_partial()
    return model

def create_model():
    model = keras.Sequential ([
        keras.layers.Dense(144),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(2, activation ='tanh')
    ])
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
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
    print(one_hot)
    return one_hot

def main(a):
    test_model = load_model()
    test_board = make_board()
    test_one_hot = make_one_hot(test_board)
    test_test = []
    test_test.append(test_one_hot)
    test_test_test = np.array(test_test)
    if a == 1:
        prediction = test_model.predict(test_test_test)[0]
        print("/////////")
        print(prediction)
    elif a == 2:
        prediction = test_model.predict(test_test_test)
        print("/////////")
        print(prediction)    
    return

a = int(sys.argv[1])
main(a)