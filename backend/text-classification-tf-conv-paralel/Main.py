#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import Loader
from Network import Network

# ------------ Parameters ------------

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_size", 84, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filters", "3,4,5", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("filer_size", 64, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")

# Training parameters
tf.flags.DEFINE_integer("categories", 39, "Number of categories")
tf.flags.DEFINE_integer("batch", 100, "Batch Size")
tf.flags.DEFINE_integer("epochs", 200, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on test set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr, value))
print("")


# ------------ Data Prep ------------

# Load data
print("Loading data")
x_train, y_train, x_test, y_test, vocabulary, vocabulary_inv = Loader.load_data()

print("Vocabulary size: {:d}".format(len(vocabulary)))
print("Train data : {:d} - Test data : {:d}".format(len(y_train), len(y_test)))


#  ------------ Train ------------
g = tf.Graph()
with g.as_default():
    sess = tf.InteractiveSession()
    with sess.as_default():
        network = Network(sentence_size=x_train.shape[1], categories=FLAGS.categories, vocabulary_size=len(vocabulary),
                          embedding_size=FLAGS.embedding_size, filters=list(map(int, FLAGS.filters.split(","))), filter_size=FLAGS.filer_size)

        #training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(network.cost)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        #models and summaries dir
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("First checkpoint saved: {}\n".format(out_dir))

        #checkpoint dir
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        #summaries for cost and accuracy
        cost_summary = tf.scalar_summary("cost", network.cost)
        accuracy_summary = tf.scalar_summary("accuracy", network.accuracy)

        #train summaries
        train_summary_op = tf.merge_summary([cost_summary, accuracy_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        #test summaries
        test_summary_op = tf.merge_summary([cost_summary, accuracy_summary])
        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.train.SummaryWriter(test_summary_dir, sess.graph_def)


        sess.run(tf.initialize_all_variables())

        #training
        def train_step(x_batch, y_batch):
            feed_dict = {network.x: x_batch, network.y: y_batch, network.dropout_keep: FLAGS.dropout_keep}
            _, step, summaries = sess.run([train_op, global_step, train_summary_op], feed_dict)

            train_summary_writer.add_summary(summaries, step)

        #testing
        def test_step(x_batch, y_batch):
            feed_dict = {network.x: x_batch, network.y: y_batch, network.dropout_keep: 1.0}
            step, summaries, cost, accuracy = sess.run([global_step, test_summary_op, network.cost, network.accuracy], feed_dict)

            print("Step: {} \nAccuracy: {: 0.3f}".format(step, accuracy))
            test_summary_writer.add_summary(summaries, step)

        #generate batches
        batches = Loader.batch_generator(list(zip(x_train, y_train)), FLAGS.batch, FLAGS.epochs)


        #main loop (train then test at every n step)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                test_step(x_test, y_test)
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Checkpoint saved: {}\n".format(path))
