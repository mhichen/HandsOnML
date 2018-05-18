#!/usr/bin/python3

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
        
if __name__ == "__main__":


    n_epochs = 1000
    learning_rate = 0.01


    housing = fetch_california_housing()

    m, n = housing.data.shape

    # (20640, 8)
    print(m, n)

    housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

    # Need to normalize input feature vectors
    scaler = StandardScaler()
    scaler.fit(housing_data_plus_bias)
    scaled_housing_data_plus_bias = scaler.transform(housing_data_plus_bias)

    
    X = tf.constant(scaled_housing_data_plus_bias, dtype = tf.float32, name = "X")
    Y = tf.constant(housing.target.reshape(-1, 1), dtype = tf.float32, name = "Y")

    beta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = "beta")

    Y_pred = tf.matmul(X, beta, name = "prediction")

    error = Y_pred - Y

    mse = tf.reduce_mean(tf.square(error), name = "mse")

    # gradients = 2/m * tf.matmul(tf.transpose(X), error)

    # Use autodiff instead
    # gradients = tf.gradients(mse, [beta])[0]
    # training_op = tf.assign(beta, beta - learning_rate * gradients)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    training_op = optimizer.minimize(mse)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        print(X)


        for epoch in range(n_epochs + 1):

            if epoch % 100 == 0:

                print("Epoch", epoch, "MSE = ", mse.eval())

            sess.run(training_op)

        best_beta = beta.eval()
        
        print("best beta is ", best_beta)

