import pickle
import gzip
import numpy as np

binary_file = open("./data/binary_file.txt", "w")

def load_data():
    f = open('./data/data.txt', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (1, 1682)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (1, 1682)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (1, 1682)) for x in te_d[0]]
    test_results = [vectorized_result(y) for y in te_d[1]]
    test_data = list(zip(test_inputs, test_results))

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((1, 39))
    e[0][j] = 1.0

    return e