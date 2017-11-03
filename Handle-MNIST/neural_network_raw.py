# 2-hidden layer fully connected NN

'''
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network_raw.py

Target is to handle MNIST
'''

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print mnist

import tensorflow as tf

# hyper Parameters 
learning_rate = .1
num_step = 500
batch_size = 128
display_step = 100

# network parameters
num_input = 784 # input img shape 28*28
n_hidden_1 = 256
n_hidden_2 = 256
num_classes = 10 # 0-9

# tf flaceholder
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
} 

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# model
def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer

logits_nn = neural_net(X)
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

    for step in range(1, num_step+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            los, acc = sess.run([loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print "step " + str(step) + " loss=" + "{:.4f}".format(los) + " accuracy=" + "{:.3f}".format(acc)

    print "Testing accuracy: "
    print sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})

