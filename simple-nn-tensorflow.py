import numpy

# Target is to train a NN for y=x*x-0.5

# define data
x_data = numpy.linspace(-2, 2, 1000)[:, numpy.newaxis]
noise = numpy.random.normal(0, 0.05, x_data.shape)
y_data = numpy.square(x_data) - 0.5 + noise
print x_data.shape, y_data.shape


# define NN model
import tensorflow as tf
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
print xs.shape, ys.shape

def add_layer(inputs, in_size, out_size, activation=None):
    weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    wx_plus_b = tf.matmul(inputs, weights) + biases

    if activation is None:
        outputs = wx_plus_b
    else:
        outputs = activation(wx_plus_b)
    return outputs

h1 = add_layer(xs, 1, 20, activation=tf.nn.relu)

predict = add_layer(h1, 20, 1)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predict),
                                    reduction_indices=[1])) 

# define train and loss
learning_rate = 0.1
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# train
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data,
                                    ys: y_data})
    if i % 50 == 0:
        print sess.run(loss, feed_dict={xs: x_data,
                                        ys: y_data})

# predict new data
x_real = numpy.zeros([1000,1])
x_real[999][0] = 0.5
print sess.run(predict, feed_dict={xs: x_real})

