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
            W_emebeddings = tf.Print(W_emebeddings, [W_emebeddings], message="W embedded : ")
            self.embedded_words = tf.nn.embedding_lookup(W_emebeddings, self.x)
            #self.embedded_words = tf.Print(self.embedded_words, [self.embedded_words], message="embedded_words chars")
            self.embedded_words_extended = tf.expand_dims(self.embedded_words, -1)

        #convolution + maxpool layer 1
        with tf.name_scope("convolution-maxpool-1"):
            filter_shape = [filters[0], filters[0], 1, filter_size]
            #convolution
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[filter_size]), name="b")
            conv = tf.nn.conv2d(self.embedded_words_extended, W, strides=[1, 1, 1, 1], padding="VALID", name="convolution")
            #activation function
            h_conv_1 = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            #maxpool
            self.max_pool_1 = tf.nn.max_pool(h_conv_1, ksize=[1, 1, 2, 1], strides=[1, 1, 1, 1], padding='VALID', name="maxpool")

        # convolution + maxpool layer 2
        with tf.name_scope("convolution-maxpool-2"):
            filter_shape = [filters[1], filters[1], filter_size, filter_size]
            # convolution
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[filter_size]), name="b")
            conv = tf.nn.conv2d(self.max_pool_1, W, strides=[1, 1, 1, 1], padding="VALID", name="convolution")
            # activation function
            h_conv_2 = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # maxpool
            self.max_pool_2 = tf.nn.max_pool(h_conv_2, ksize=[1, 1, 2, 1], strides=[1, 1, 1, 1], padding='VALID', name="maxpool")

        # convolution + maxpool layer 3
        with tf.name_scope("convolution-maxpool-3"):
            filter_shape = [filters[2], filters[2], filter_size, filter_size]
            # convolution
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[filter_size]), name="b")
            conv = tf.nn.conv2d(self.max_pool_2, W, strides=[1, 1, 1, 1], padding="VALID", name="convolution")
            # activation function
            h_conv_3 = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # maxpool
            self.max_pool_3 = tf.nn.max_pool(h_conv_3, ksize=[1, 1, 2, 1], strides=[1, 1, 1, 1], padding='VALID', name="maxpool")


        #dropout
        self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep")
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.max_pool_3, self.dropout_keep)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            h_drop_shape = self.h_drop.get_shape().as_list()
            W = tf.get_variable("W", shape=[h_drop_shape[1]*h_drop_shape[2]*h_drop_shape[3], categories], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[categories]), name="b")
            h_drop_flat = tf.reshape(self.h_drop, [-1, h_drop_shape[1]*h_drop_shape[2]*h_drop_shape[3]])
            self.y_conv = tf.nn.xw_plus_b(h_drop_flat, W, b, name="scores")
            self.prediction = tf.argmax(self.y_conv, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("cost"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.y_conv, self.y)
            self.cost = tf.reduce_mean(cross_entropy)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
