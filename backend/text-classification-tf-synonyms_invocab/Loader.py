
import numpy as np
import re
import itertools
from collections import Counter
import pickle

import DictionarySyn

vocabulary = {}

#string cleaning, lowercasing, tokenize
def clean_string(sentence):

    sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)

    sentence = re.sub(r"[0-9]", "", sentence)

    sentence = re.sub(r"\'s", " \'s", sentence)
    sentence = re.sub(r"\'ve", " \'ve", sentence)
    sentence = re.sub(r"n\'t", " n\'t", sentence)
    sentence = re.sub(r"\'re", " \'re", sentence)
    sentence = re.sub(r"\'d", " \'d", sentence)
    sentence = re.sub(r"\'ll", " \'ll", sentence)

    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " \? ", sentence)
    sentence = re.sub(r"\(", " \( ", sentence)
    sentence = re.sub(r"\)", " \) ", sentence)

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
    training_data, test_data, eval_data = pickle.load(f, encoding='latin1')
    f.close()

    return (training_data, test_data, eval_data)


#load training, test, validation data
def load_sentences_and_labels(validation):

    tr_d, te_d, va_d = load_my_data()

    if (not validation):
        #put in vectors text, labels
        training_inputs = tr_d[0]
        training_results = [vectorized_result(y) for y in tr_d[1]]
        test_inputs = te_d[0]
        test_results = [vectorized_result(y) for y in te_d[1]]

        #split by words
        training_inputs = [clean_string(tr_s) for tr_s in training_inputs]
        training_inputs = [s.split(" ") for s in training_inputs]
        test_inputs = [clean_string(te_s) for te_s in test_inputs]
        test_inputs = [s.split(" ") for s in test_inputs]

        return [training_inputs, training_results, test_inputs, test_results]

    else:
        # put in vectors text, labels
        training_inputs = tr_d[0]
        training_results = [vectorized_result(y) for y in tr_d[1]]
        test_inputs = te_d[0]
        test_results = [vectorized_result(y) for y in te_d[1]]
        val_inputs = va_d[0]
        val_results = va_d[1]

        # split by words
        training_inputs = [clean_string(tr_s) for tr_s in training_inputs]
        training_inputs = [s.split(" ") for s in training_inputs]
        test_inputs = [clean_string(te_s) for te_s in test_inputs]
        test_inputs = [s.split(" ") for s in test_inputs]
        val_inputs = [clean_string(va_s) for va_s in val_inputs]
        val_inputs = [s.split(" ") for s in val_inputs]

        return [training_inputs, training_results, test_inputs, test_results, val_inputs, val_results]


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
    top_most_common_words = word_counts.most_common()

    vocabulary_inv = []
    dic_syn = DictionarySyn.DictionarySyn()
    for x in top_most_common_words:
        syn_found = False
        syns = dic_syn.get_syns(x[0])
        for syn in syns:
            if syn in vocabulary_inv:
                syn_found = True
                break
        if not syn_found:
            vocabulary_inv.append(x[0])

    #mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    vocabulary.update({'<UNK>': len(vocabulary_inv)})

    return vocabulary


#map sentences + labels -> to vectors based on vocabulary
def build_input_data(sentences, labels, vocabulary):

    temp_a = []
    temp_b = []
    for sentence in sentences:
        for word in sentence:
            try:
                code = vocabulary[word]
                temp_b.append(code)
            except KeyError:
                print('For word: {}'.format(word))
                dic_syn = DictionarySyn.DictionarySyn()
                syns = dic_syn.get_syns(word)
                for syn in syns:
                    try:
                        code = vocabulary[syn]
                        print('Synonym matched in vocabulary: {}\n'.format(syn))
                        break
                    except KeyError:
                        code = vocabulary['<UNK>']
                temp_b.append(code)
        temp_a.append(temp_b)
        temp_b = []

        #x = np.array([[ for word in sentence] for sentence in sentences])
    x = np.array(temp_a)
    y = np.array(labels)

    return [x, y]


#load + process data for the set; return inputs, lables, vocab, inverse vocab
def load_data():

    train_sentences, train_labels, test_sentences, test_labels = load_sentences_and_labels(validation=False)
    maxWords = max(len(x) for x in train_sentences+test_sentences)

    train_sentences_padded = pad_sentences(train_sentences, maxWords)
    test_sentences_padded = pad_sentences(test_sentences, maxWords)

    vocabulary = build_vocab(train_sentences_padded + test_sentences_padded)
    x_train, y_train = build_input_data(train_sentences_padded, train_labels, vocabulary)
    x_test, y_test = build_input_data(test_sentences_padded, test_labels, vocabulary)

    return [x_train, y_train, x_test, y_test, vocabulary]



def load_data_eval():

    train_sentences, train_labels, test_sentences, test_labels, val_sentences, val_labels = load_sentences_and_labels(validation=True)
    maxWords = max(len(x) for x in train_sentences+test_sentences+val_sentences)

    train_sentences_padded = pad_sentences(train_sentences, maxWords)
    test_sentences_padded = pad_sentences(test_sentences, maxWords)
    val_sentences_padded = pad_sentences(val_sentences, maxWords)

    vocabulary = build_vocab(train_sentences_padded + test_sentences_padded)
    x_val, y_val = build_input_data(val_sentences_padded, val_labels, vocabulary)

    return [x_val, y_val, vocabulary]


def load_sentence_eval(my_sentence):

    train_sentences, train_labels, test_sentences, test_labels = load_sentences_and_labels(validation=False)
    maxWords = max(len(x) for x in train_sentences + test_sentences)
    train_sentences_padded = pad_sentences(train_sentences, maxWords)
    test_sentences_padded = pad_sentences(test_sentences, maxWords)

    val_s = clean_string(my_sentence)
    val_sentence = val_s.split(" ")
    val_senteces = []
    val_senteces.append(val_sentence)
    val_labels = 2
    val_results = vectorized_result(val_labels)

    val_sentences_padded = pad_sentences(val_senteces, maxWords)

    vocabulary = build_vocab(train_sentences_padded + test_sentences_padded)
    x_val, y_val = build_input_data(val_sentences_padded, val_results, vocabulary)

    return [x_val, y_val, vocabulary]




