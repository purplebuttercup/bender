
import tensorflow as tf
import numpy as np


class Network(object):

    def __init__(self, sentence_size, categories, vocabulary_size, embedding_size, filters, filter_size):

        #placeholders of input, output
        self.x = tf.placeholder(tf.int32, [None, sentence_size], name="x")
        self.y = tf.placeholder(tf.float32, [None, categories], name="y")

        #embedding layer
        with tf.name_scope("embedding"):
            W_emebeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="W_emebeddings")
            #W_emebeddings = tf.Print(W_emebeddings, [W_emebeddings], message="W embedded : ")
            self.embedded_words = tf.nn.embedding_lookup(W_emebeddings, self.x)
            #self.embedded_words = tf.Print(self.embedded_words, [self.embedded_words], message="embedded_words chars")
            self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)

        #convolution + maxpool layer for each filter size
        total_maxpools = []
        for i, filter in enumerate(filters):
            with tf.name_scope("convolution-maxpool-%s" % filter):
                filter_shape = [filter, embedding_size, 1, filter_size]
                #convolution
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[filter_size]), name="b")
                conv = tf.nn.conv2d(self.embedded_words_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="convolution")
                print(conv.get_shape())
                #activation function
                h_conv = tf.nn.relu(conv + b, name="relu")
                #maxpool
                max_pool = tf.nn.max_pool(h_conv, ksize=[1, sentence_size - filter + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="maxpool")
                total_maxpools.append(max_pool)

        #combine all the maxpool features
        total_filters_size = filter_size * len(filters)
        self.h_pool = tf.concat(3, total_maxpools, name="concat")
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_filters_size])

        #dropout
        self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep")
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep)

        #output layer with (unnormalized) costs and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[total_filters_size, categories], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[categories]), name="b")
            self.y_conv = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.prediction = tf.argmax(self.y_conv, 1, name="predictions")

        #cross-entropy cost
        with tf.name_scope("cost"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.y_conv, self.y)
            self.cost = tf.reduce_mean(cross_entropy)

        #accuracy
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.y, 1), name="correct-predictions")
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
