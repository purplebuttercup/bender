
#get MNIST data - 55.000 = training; 10.000 = test; 5.000 = validation
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import Loader

import tensorflow as tf
from random import shuffle

train_data, validatin_data, test_data = Loader.load_data_wrapper()

sess = tf.InteractiveSession() #start interactive session

#for many weights and biases initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name="W")

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name="b")

#for many convolutions and poolings
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name="convolution")

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')


#----- first convolutional layer -------
W_conv1 = weight_variable([1, 3, 1, 84])
b_conv1 = bias_variable([84])


#symbolic var for describing interacting ops
x = tf.placeholder(tf.float32, [None, 1682], name="input_x") #input imgs

x_image = tf.reshape(x, [-1, 58, 29, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name="relu_1")
h_pool1 = max_pool_2x2(h_conv1)


#----- second convolutional layer -------
W_conv2 = weight_variable([1, 3, 84, 84])
b_conv2 = bias_variable([84])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name="relu_2")
h_pool2 = max_pool_2x2(h_conv2)


#----- third convolutional layer -------
W_conv3 = weight_variable([1, 2, 84, 84])
b_conv3 = bias_variable([84])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

#----- 4th convolutional layer -------
#W_conv4 = weight_variable([2, 2, 6, 6])
#b_conv4 = bias_variable([6])

#h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)


#----- 5th convolutional layer -------
#W_conv5 = weight_variable([2, 2, 6, 6])
#b_conv5 = bias_variable([6])

#h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)


#----- 6th convolutional layer -------
#W_conv6 = weight_variable([2, 2, 6, 6])
#b_conv6 = bias_variable([6])

#h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)
#h_pool6 = max_pool_2x2(h_conv6)


#----- densely connected layer -------
W_fc1 = weight_variable([504, 1024])
b_fc1 = bias_variable([1024])

h_pool6_flat = tf.reshape(h_pool3, [-1, 504])
h_fc1 = tf.nn.relu(tf.matmul(h_pool6_flat, W_fc1) + b_fc1)

#----- densely connected layer 2-------
#W_fc2 = weight_variable([324, 648])
#b_fc2 = bias_variable([648])

#h_pool3_flat = tf.reshape(h_pool3, [-1, 248])
#h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="dropout_keep_prob")


#----- readout(softmax) layer -------
W_fc3 = weight_variable([1024, 39])
b_fc3 = bias_variable([39])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)



#cross-entropy between target and model's prediction
y_ = tf.placeholder(tf.float32, [None, 39], name="input_y") #targeted output label
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-7).minimize(cross_entropy)

#initialize all declared vars and run
init = tf.initialize_all_variables()
sess.run(init)

#train
mini_batch_size = 100000
n = len(list(train_data))
acc_sum = 0
for i in range(1000):
    print('Epoch {0} complete '.format(i))
    shuffle(list(train_data))
    mini_batches = [train_data[k: k+mini_batch_size]
                          for k in range(0, n, mini_batch_size)]

    for mini_batch in mini_batches:
        for batch_xs, batch_ys in mini_batch:
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    #see how well it predicts
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #ask for accuracy on test data
    for x_test, y_test in test_data:
        acc_sum = acc_sum + sess.run(accuracy, feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
    print('Accuracy {0} from {1}'.format(acc_sum, len(list(test_data))))
    acc_sum = 0