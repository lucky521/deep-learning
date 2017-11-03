# Convolutional Neural Network

'''
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py

Target is to handle MNIST

'''

from __future__ import division

import tensorflow as tf

## Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print mnist
print mnist.train.images.shape
print mnist.test.images.shape
print mnist.validation.images.shape


# hyper Parameters 
learning_rate = 0.001
num_step = 500
batch_size = 128
display_step = 10


# network parameters
num_input = 784 # input img shape 28*28
num_classes = 10 # 0-9

# tf i/o flaceholder 
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
DROPOUT = tf.placeholder(tf.float32) 



# Wrapper function
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1,k,k,1], padding='SAME')


# init w and b
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
} 

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create Model
def conv_nn(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1,28,28,1]) # [Batch Size, Height, Width, Channel]

    # conv layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    # conv layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    # fully connected layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # output 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out



# model
logits_nn = conv_nn(X, weights, biases, DROPOUT)
predict = tf.nn.softmax(logits_nn)

# loss 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_nn, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

# evaluate 
correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print "Start to train..."
    for step in range(1, num_step+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, DROPOUT: 0.8})

        if step % display_step == 0 or step == 1:
            los, acc = sess.run([loss, accuracy], feed_dict={X: batch_x, Y: batch_y, DROPOUT: 1.0})
            print "step " + str(step) + " loss=" + "{:.4f}".format(los) + " accuracy=" + "{:.3f}".format(acc)

    print "Testing accuracy: "
    print sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, DROPOUT: 1.0})

