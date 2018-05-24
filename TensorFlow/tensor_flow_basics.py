#!/usr/bin/python3

#import input_data

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scipy.io as sio

import tensorflow as tf

from datetime import datetime

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

#from tensorflow.models.official.mnist import dataset
from tensorflow.examples.tutorials.mnist import input_data
if __name__ == "__main__":

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    MNIST = input_data.read_data_sets("MNIST_data", one_hot = True)

    learning_rate = 0.01
    batch_size = 128
    n_epochs = 25

    X = tf.placeholder(tf.float32, [batch_size, 784], name = "image")
    Y = tf.placeholder(tf.float32, [batch_size, 10], name = "label")

    w = tf.Variable(tf.random_normal(shape = [784, 10], stddev = 0.01), name = "weights")
    b = tf.Variable(tf.zeros([1, 10]), name = "bias")

    logits = tf.matmul(X, w) + b

    entropy = tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = logits)
    loss = tf.reduce_mean(entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()

        writer = tf.summary.FileWriter(logdir, sess.graph)
        
        n_batches = int(MNIST.train.num_examples/batch_size)

        # Train the model
        for i in range(n_epochs):

            for _ in range(n_batches):

                X_batch, Y_batch = MNIST.train.next_batch(batch_size)
                opt, loss_val = sess.run([optimizer, loss], feed_dict = {X: X_batch, Y: Y_batch})
        

            print("Epoch", i, loss_val)
            #print(sess.eval(loss))


        # Test the model
        n_batches = int(MNIST.test.num_examples/batch_size)
        total_correct_preds = 0

        for i in range(n_batches):
            X_batch, Y_batch = MNIST.test.next_batch(batch_size)
            _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict = {X: X_batch, Y: Y_batch})
            preds = tf.nn.softmax(logits_batch)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct_preds += sess.run(accuracy)

        print("Accuracy", total_correct_preds/MNIST.test.num_examples)
        

            
    writer.close()
    tf.logging.set_verbosity(old_v)
