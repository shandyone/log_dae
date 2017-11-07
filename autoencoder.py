# -*- coding:utf-8 -*-
# from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

<<<<<<< HEAD
import numpy as np
import utils
=======

import numpy as np

>>>>>>> fbe8d34eae8a264e37821c4f720393b81258c6a7
import tensorflow as tf
import matplotlib.pyplot as plt
import read_csv

# Parameters
learning_rate = 0.006
training_epochs = 20000
batch_size = 10
display_step = 1
examples_to_show = 10
time_step = 30
<<<<<<< HEAD
corruption_level=0.5

def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs+ n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)
=======
corruption_level=0.6
>>>>>>> fbe8d34eae8a264e37821c4f720393b81258c6a7

def get_data():#if time_step is too long, the batch_index will be none
    #data = read_csv.data
    data = read_csv.normalize_data
    train_x = []
    batch_index = []
    train_max = (len(data)+time_step-1)//time_step
    for i in range(train_max):
        if i % batch_size == 0:
            batch_index.append(i)
        x = data[i:i+time_step]
        train_x.append(x.tolist())
    return train_x,batch_index

<<<<<<< HEAD
# Network Parameters
# n_hidden_1 = 256 # 1st layer num features
# n_hidden_2 = 128 # 2nd layer num features
# n_input = 784 # MNIST data input (img shape: 28*28)

=======
def xavier_init(fan_in,fan_out,constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

# Network Parameters
>>>>>>> fbe8d34eae8a264e37821c4f720393b81258c6a7
n_hidden_1 = 15 # 1st layer num features
n_hidden_2 = 8 # 2nd layer num features
n_input = 30 # data input (img shape: 28*28)

# native data
X = tf.placeholder("float", [None, n_input])
mask = tf.placeholder("float", [None, n_input], name='mask')

# 两层encoder，两层decoder
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
<<<<<<< HEAD
    # 'encoder_h1': tf.Variable(xavier_init(n_input, n_hidden_1)),
    # 'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    # 'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    # 'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

=======
}
>>>>>>> fbe8d34eae8a264e37821c4f720393b81258c6a7
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

<<<<<<< HEAD
=======
# weights = {
#     'encoder_h1': tf.Variable(xavier_init(n_input, n_hidden_1)),
#     'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
#     'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
#     # 'encoder_h2': tf.Variable(tf.zeros([n_hidden_1, n_hidden_2])),
#     # 'decoder_h1': tf.Variable(tf.zeros([n_hidden_2, n_hidden_1])),
#     # 'decoder_h2': tf.Variable(tf.zeros([n_hidden_1, n_input])),
# }
# biases = {
#     'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'decoder_b2': tf.Variable(tf.random_normal([n_input])),
# }

>>>>>>> fbe8d34eae8a264e37821c4f720393b81258c6a7
def encoder(x):
    mask_x = x * mask
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(mask_x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

data, batch_index = get_data()
data_max = read_csv.data_max
data_min = read_csv.data_min
saver=tf.train.Saver(tf.global_variables())

def train():
    cost = tf.reduce_mean(tf.pow(X - decoder_op, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            for i in range(len(batch_index)-1):
                batch_xs = np.array(data[batch_index[i]:batch_index[i+1]]).reshape([-1,time_step])
                mask_np = np.random.binomial(1, 1 - corruption_level, batch_xs.shape)
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs,mask:mask_np})

            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c)
<<<<<<< HEAD
        saver.save(sess, '/home/zshang/Pycharm/log_dae/model/dae.model')
=======
        #saver.save(sess, '/home/zshang/Pycharm/log_dae/model/dae.model')
        saver.save(sess, 'model/dae.model')
>>>>>>> fbe8d34eae8a264e37821c4f720393b81258c6a7
        print "Optimization Finished!"

def test():
    with tf.Session() as sess:
<<<<<<< HEAD
        module_file = tf.train.latest_checkpoint('/home/zshang/Pycharm/log_dae/model')
=======
        #module_file = tf.train.latest_checkpoint('/home/zshang/Pycharm/log_dae/model')
        module_file = tf.train.latest_checkpoint('model')
>>>>>>> fbe8d34eae8a264e37821c4f720393b81258c6a7
        saver.restore(sess, module_file)
        data_test=np.array(data[2]).reshape([-1,time_step])
        #mask_np = np.random.binomial(1, 1 - corruption_level, data_test.shape)
        mask_np = np.ones(data_test.shape)
        encode_decode = sess.run(decoder_op, feed_dict={X: data_test,mask:mask_np})

    data_test_re = (data_test.reshape([-1]))*(data_max-data_min)+data_min
    ende = (encode_decode.reshape([-1]))*(data_max-data_min)+data_min
<<<<<<< HEAD
    print data_test_re
    print ende

    plt.figure()
    plt.plot(data_test_re,c='b')
    plt.plot(ende,c='y')
    plt.savefig('diff.png')
    plt.show()

if __name__ == '__main__':
    #train()
    try:
        test()
    except:
        train()
        test()
=======
    #print data_test_re
    #print ende
    print np.array(data).shape

    plt.figure()
    plt.plot(data_test_re,c='b')
    plt.plot(ende,c='r')
    plt.savefig('pic/diff.png')
    plt.show()

if __name__ == '__main__':
    train()
    test()
>>>>>>> fbe8d34eae8a264e37821c4f720393b81258c6a7

# f, a = plt.subplots(2, 10, figsize=(10, 2))
# for i in range(examples_to_show):
#     a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
#     a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
# f.show()
# plt.draw()
# plt.waitforbuttonpress()
