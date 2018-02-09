import tensorflow as tf
import numpy as np
import math

class Network(object):

    def __init__(self, sentence_size, categories, vocabulary_size, embedding_size):

        #placeholders of input, output
        self.x = tf.placeholder(tf.int32, [None, sentence_size], name="x")
        self.y = tf.placeholder(tf.float32, [None, categories], name="y")

        #embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W_emebeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="W_emebeddings")
            #W_emebeddings = tf.Print(W_emebeddings, [W_emebeddings], message="W W_emebeddings : ", first_n=8, summarize=10)
            self.embedded_words = tf.nn.embedding_lookup(W_emebeddings, self.x)
            #self.embedded_words = tf.Print(self.embedded_words, [self.embedded_words], message="embedded_words chars", first_n=8, summarize=10)
            self.embedded_words_aggregated = tf.reduce_sum(self.embedded_words, [1])
        #hidden layer 1
        with tf.name_scope("hidden-1"):
            W = tf.Variable(tf.truncated_normal([embedding_size, vocabulary_size], stddev=0.1 / math.sqrt(embedding_size)), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[vocabulary_size]), name="b")
            self.h_1 = tf.nn.relu(tf.add(tf.matmul(self.embedded_words_aggregated, W), b))

        #dropout 1
        self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep")
        with tf.name_scope("dropout-1"):
            self.h_drop_1 = tf.nn.dropout(self.h_1, self.dropout_keep)

        # hidden layer 2
        with tf.name_scope("hidden-2"):
            W = tf.Variable(tf.truncated_normal([vocabulary_size, vocabulary_size*2], stddev=0.1 / math.sqrt(embedding_size)), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[vocabulary_size*2]), name="b")
            self.h_2 = tf.nn.relu(tf.add(tf.matmul(self.h_drop_1, W), b))

        # dropout 2
        with tf.name_scope("dropout-2"):
            self.h_drop_2 = tf.nn.dropout(self.h_2, self.dropout_keep)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[vocabulary_size*2, categories], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[categories]), name="b")
            self.y_ = tf.nn.xw_plus_b(self.h_drop_2, W, b, name="scores")
            self.prediction = tf.argmax(self.y_, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("cost"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.y_, self.y)
            self.cost = tf.reduce_mean(cross_entropy)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
