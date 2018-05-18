#!/usr/bin/python3

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# housing = fetch_california_housing()
# m, n = housing.data.shape
# housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# n_epochs = 1000
# learning_rate = 0.01


# scaler = StandardScaler()
# scaled_housing_data = scaler.fit_transform(housing.data)
# scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
# y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

# theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
# y_pred = tf.matmul(X, theta, name="predictions")
# error = y_pred - y
# mse = tf.reduce_mean(tf.square(error), name="mse")
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# training_op = optimizer.minimize(mse)

# init = tf.global_variables_initializer()

# n_epochs = 10

# batch_size = 100
# n_batches = int(np.ceil(m / batch_size))

# def fetch_batch(epoch, batch_index, batch_size):
#     np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
#     indices = np.random.randint(m, size=batch_size)  # not shown
#     X_batch = scaled_housing_data_plus_bias[indices] # not shown
#     y_batch = housing.target.reshape(-1, 1)[indices] # not shown
#     return X_batch, y_batch

# with tf.Session() as sess:
#     sess.run(init)

#     for epoch in range(n_epochs):
#         for batch_index in range(n_batches):
#             X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

#     best_theta = theta.eval()

#     print(best_theta)
    
def fetch_batch(X_, Y_, m, epoch, batch_index, n_batches, batch_size):

    np.random.seed(epoch * n_batches + batch_index)

    indices = np.random.randint(m, size = batch_size)

    print("indices.shape ", indices.shape)
    
    X_batch = X_[indices]
    Y_batch = Y_.reshape(-1, 1)[indices]

    return X_batch, Y_batch
    
    

if __name__ == "__main__":

    
    # First get data
    housing = fetch_california_housing()
    m, n = housing.data.shape

    learning_rate = 0.01

    # Need to normalize input feature vectors
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)

    # Add bias vector
    scaled_housing_data_plus_bias = np.c_[np.ones((m,1)), scaled_housing_data]

    # Start with tensorflow components
    X = tf.placeholder(tf.float32, shape = (None, n + 1), name = "X")
    Y = tf.placeholder(tf.float32, shape = (None, 1), name = "Y")

    beta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = "beta")

    Y_pred = tf.matmul(X, beta, name = "prediction")

    error = Y_pred - Y

    mse = tf.reduce_mean(tf.square(error), name = "mse")

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    # Target
    #target = housing.target.reshape(-1,1)
    target = housing.target
    
    # For training
    n_epochs = 10

    

    # For mini-batch
    batch_size = 100
    n_batches = int(np.ceil(m/batch_size))

    # def fetch_batch(epoch, batch_index, batch_size):
    #     np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    #     indices = np.random.randint(m, size=batch_size)  # not shown
    #     X_batch = scaled_housing_data_plus_bias[indices] # not shown
    #     y_batch = housing.target.reshape(-1, 1)[indices] # not shown
        
    #     return X_batch, y_batch



    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(n_epochs):
        
            for batch_index in range(n_batches):

                print("batch index is ", batch_index)

                X_batch, Y_batch = fetch_batch(scaled_housing_data_plus_bias, target, m, epoch, batch_index, n_batches, batch_size)
                #X_batch, Y_batch = fetch_batch(epoch, batch_index, batch_size)
                
                sess.run(training_op, feed_dict = {X: X_batch, Y: Y_batch})

                
            # if epoch % 100 == 0:

            #     print("Epoch", epoch, "MSE = ", mse.eval())


        best_beta = beta.eval()
        
        print("best beta is ", best_beta)

