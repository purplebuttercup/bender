#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import Loader


# ------------ Parameters ------------

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1612, "Batch size")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1465304439/checkpoints", "Directory with trained model")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr, value))
print("")


# ------------ Data Prep ------------

print("Loading data...")
x_eval, y_eval, vocabulary = Loader.load_data_eval()
all_y_eval = []
all_y_eval = np.concatenate([all_y_eval, y_eval])

print("Vocabulary size: {:d}".format(len(vocabulary)))
print("Evaluation data : {:d}".format(len(y_eval)))


#  ------------ Evaluation ------------

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        #load saved meta graph, restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        #restore placeholders from graph
        x = graph.get_operation_by_name("x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep = graph.get_operation_by_name("dropout_keep").outputs[0]

        #tensor
        prediction_op = graph.get_operation_by_name("output/predictions").outputs[0]

        #get model predictions
        predictions = sess.run(prediction_op, {x: x_eval, dropout_keep: 1.0})

# Print accuracy
correct_predictions = sum(predictions == all_y_eval)
print("Accuracy: {:g}".format(correct_predictions / len(y_eval)))
#print(ClassMap.classes[batch_predictions[0]])