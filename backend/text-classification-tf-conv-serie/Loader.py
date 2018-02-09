import numpy as np
import re
import itertools
from collections import Counter
import pickle

vocabulary = {}

#Tokenization/string cleaning
def clean_str(sentence):

    sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
    sentence = re.sub(r"\'s", " \'s", sentence)
    sentence = re.sub(r"\'ve", " \'ve", sentence)
    sentence = re.sub(r"n\'t", " n\'t", sentence)
    sentence = re.sub(r"\'re", " \'re", sentence)
    sentence = re.sub(r"\'d", " \'d", sentence)
    sentence = re.sub(r"\'ll", " \'ll", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\(", " \( ", sentence)
    sentence = re.sub(r"\)", " \) ", sentence)
    sentence = re.sub(r"\?", " \? ", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return sentence.strip().lower()

#put label no. in binary vector form
def vectorized_result(j):
    e = []
    e = e + [0]*39
    e[j] = 1
    return e

#load from file
def load_my_data():
    f = open('./data/data.txt', 'rb')
    training_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, test_data)


def load_data_and_labels():

    #load training and test data
    tr_d, te_d = load_my_data()

    #put in vectors text, labels
    training_inputs = tr_d[0]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    test_inputs = te_d[0]
    test_results = [vectorized_result(y) for y in te_d[1]]

    #split by words
    training_inputs = [clean_str(tr_s) for tr_s in training_inputs]
    training_inputs = [s.split(" ") for s in training_inputs]
    test_inputs = [clean_str(te_s) for te_s in test_inputs]
    test_inputs = [s.split(" ") for s in test_inputs]

    return [training_inputs, training_results, test_inputs, test_results]


#pad sentences to longest sentence
def pad_sentences(sentences, maxWords):

    paddingWord = "<PAD>"
    padded_sentences = []

    for i in range(len(sentences)):
        sentence = sentences[i]
        pad = maxWords - len(sentence)
        newSentence = sentence + [paddingWord] * pad
        padded_sentences.append(newSentence)

    return padded_sentences


#build vocab from word -> index based on sentences
def build_vocab(sentences):

    #build vocabulary: most common words are first
    word_counts = Counter(itertools.chain(*sentences))

    #mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]

    #mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return [vocabulary, vocabulary_inv]


#map sentences + labels -> to vectors based on vocabulary
def build_input_data(sentences, labels, vocabulary):

    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)

    return [x, y]


#load + process data for the set; return inputs, lables, vocab, inverse vocab
def load_data():

    train_sentences, train_labels, test_sentences, test_labels = load_data_and_labels()
    maxWords = max(len(x) for x in train_sentences+test_sentences)

    train_sentences_padded = pad_sentences(train_sentences, maxWords)
    test_sentences_padded = pad_sentences(test_sentences, maxWords)

    vocabulary, vocabulary_inv = build_vocab(train_sentences_padded + test_sentences_padded)
    x_train, y_train = build_input_data(train_sentences_padded, train_labels, vocabulary)
    x_test, y_test = build_input_data(test_sentences_padded, test_labels, vocabulary)

    return [x_train, y_train, x_test, y_test, vocabulary, vocabulary_inv]


#extract data in batches
def batch_generator(data, batch_size, epochs):

    data = np.array(data)
    data_size = len(data)
    batches_per_epoch = int(data_size/batch_size) + 1

    for epoch in range(epochs):
        #shuffle at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]

        #use of generator -> iterate only once; don't store values in memory => speed
        for b in range(batches_per_epoch):
            start = b * batch_size
            end = min((b + 1) * batch_size, data_size)
            yield shuffled_data[start:end]
