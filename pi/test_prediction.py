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
    board_1 = [1, 0, -2, 0, -2, 0]
    board_2 = [0, 0, 0, 0, -1, 0]
    board_3 = [1, 0, -1, -2, 1, 0]
    board_4 = [0, 0, 2, 0, -1, 0]
    board_5 = [0, 0, 0, 2, 0, 0]
    board_6 = [0, 0, 0, 0, 0, 0]
    board.append(board_1)
    board.append(board_2)
    board.append(board_3)
    board.append(board_4)
    board.append(board_5)
    board.append(board_6)
    print(board_1)
    print(board_2)
    print(board_3)
    print(board_4)
    print(board_5)
    print(board_6)
    return board

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
    print(one_hot)
    return one_hot

def main():
    test_model = load_model()
    test_board = make_board()
    test_one_hot = make_one_hot(test_board)
    test_test = []
    test_test.append(test_one_hot)
    test_test_test = np.array(test_test)
    prediction = test_model.predict(test_test_test)[0]
    print("/////////")
    print(prediction)   
    return

main()