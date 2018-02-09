
import tensorflow as tf
import sys
import Loader
import ClassMap

# ------------ Arguments ------------
arguments = sys.argv[1:]
#print("\nArguments:", arguments)


# ------------ Parameters ------------

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1466172419/checkpoints", "Checkpoint directory from training run")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
#print("\nParameters:")
#for attr, value in sorted(FLAGS.__flags.items()):
    #print("{} = {}".format(attr, value))
#print("")


# ------------ Data Prep ------------

#print("Loading data...")
sentence = arguments[0]
x_eval, y_eval, vocabulary = Loader.load_sentence_eval(sentence)
#y_eval = np.argmax(y_eval, axis=1)

#print("Vocabulary size: {:d}".format(len(vocabulary)))
#print("Evaluation data : {:d}".format(len(y_eval)))


#  ------------ Evaluation ------------

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        x = graph.get_operation_by_name("x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep = graph.get_operation_by_name("dropout_keep").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        #batches = Loader.batch_generator(x_eval, FLAGS.batch_size, epochs=1, validation=False)

        # Collect the predictions here
        all_predictions = []

        #for x_eval_batch in batches:
        batch_predictions = sess.run(predictions, {x: x_eval, dropout_keep: 1.0})
            #all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy
#correct_predictions = sum(all_predictions == y_eval)
#print("Total number of evaluation examples: {}\n".format(len(y_eval)))

#print("Accuracy: {:g}".format(correct_predictions / len(y_eval)))
#print("Service operation : ", sentence)
print(ClassMap.classes[batch_predictions[0]])