# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import read_csv

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

learning_rate = 0.06
training_epochs = 1000
#batch_size = 256
batch_size = 10
display_step = 1 #which step to show the cost
#n_input = 784
n_input = 30
corruption_level=0.25
time_step=30

X = tf.placeholder("float", [None, time_step],name='X')
mask = tf.placeholder("float", [None, time_step], name='mask')

n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], 0.0,1.0)),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],0.0,1.0 )),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],0.0,1.0 )),
    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4], 0.0,1.0)),
    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3], 0.0,1.0)),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],0.0,1.0 )),
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], 0.0,1.0)),
    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], 0.0,1.0)),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([n_input])),
}

def get_data():
    #data = read_csv.data
    #normalize_data = read_csv.normalize_data
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

def encoder(x,mask):
    mask_x = x * mask
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(mask_x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    # for the output of code layer, we don`t use activation function
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                     biases['encoder_b4'])
    return layer_4


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                   biases['decoder_b4']))
    return layer_4


encoder_op = encoder(X,mask)
#devide encoder process and decoder process to make some special issue
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#cost = -tf.reduce_mean(tf.reduce_sum(y_pred*tf.log(y_true)+()))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #c = 0
    train_x,batch_index = get_data()
    for epoch in range(training_epochs):
        for i in range(len(batch_index)-1):

            batch_xs = np.array(train_x[batch_index[i]:batch_index[i+1]]).reshape([-1,time_step])
            #print batch_xs
            mask_np = np.random.binomial(1, 1 - corruption_level, batch_xs.shape)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs,mask:mask_np})
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c)
    print("Optimization Finished!")

    batch_tx = np.array(train_x[1]).reshape([-1,time_step])
    mask_np = np.random.binomial(1, 1 - corruption_level, batch_tx.shape)
    decoder_result = sess.run(decoder_op, feed_dict={X: batch_tx,mask:mask_np})
    # #at last, make this process to get these data
    # plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    # plt.colorbar()
    # plt.savefig('test.png')
    # plt.show()
    print batch_tx
    print decoder_result
    plt.figure()
    plt.plot(batch_tx)
    plt.show()