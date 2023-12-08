import tensorflow as tf
from tensorflow import keras

#コマンドライン引数としてファイル名を指定　(python3 geisterML.py data_seed{:04d}to{:04d}_alpha{}_match{}_border{}_fBorder{}.pkl data_seed{:04d}to{:04d}_alpha{}_match{}_border{}_fBorder{}.pkl)
#前者が学習データ、後者がテストデータ
data_name = sys.argv[1]
test_data_name = sys.argv[2]

def get_data():
    with open('{}'.format(data_name), 'rb') as tf:
        data = pickle.load(tf)
    with open('{}'.format(test_data_name), 'rb') as tf:
        test_data = pickle.load(tf)
    return data, test_data


def create_model():
    model = keras.Sequential ([
        keras.layers.Dense(144),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(2, activation ='tanh')
    ])
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model

def load_model():
    model = create_model()
    model.load_weights('modelA').expect_partial()
    return model

#['0', '0', '1', '0', '0', '1']を[0 0 1 0 0 1]（numpy配列）に変換
def change_list(strList):
    intList = [int(n) for n in strList] 
    npList = np.array(intList)
    return npList


#辞書データをnumpy配列に変換（変換例とともに記す）
def get_onehot(data_dict):
    #（例）data_dict = {"001001": 0.5, "010010": 0.45, "100010": 0.72, "010100": 0.4}
    key, value = list(data_dict.keys()), list(data_dict.values())
    #（例）key = ['001001', '010010', '100010', '010100'], value = [0.5, 0.45, 0.72, 0.4]
    a = [list(i) for i in key]
    #（例）a = [['0', '0', '1', '0', '0', '1'], ['0', '1', '0', '0', '1', '0'], ['1', '0', '0', '0', '1', '0'], ['0', '1', '0', '1', '0', '0']]
    b = [change_list(j) for j in a]
    #b = [[0 0 1 0 0 1], [0 1 0 0 1 0], [1 0 0 0 1 0], [0 1 0 1 0 0]]（numpy配列がそれぞれの要素となるリスト）
    return np.array(b), np.array(value)



def machineLearning(data, test_data):
    study_data, study_label = get_onehot(data)
    test_data, test_label = get_onehot(test_data)


data, test_data = get_data()
machineLearning(data, test_data)

