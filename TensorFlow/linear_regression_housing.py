#!/usr/bin/python3

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
        
if __name__ == "__main__":

    

    housing = fetch_california_housing()

    m, n = housing.data.shape

    # (20640, 8)
    print(m, n)

    housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

    m, n = housing.data.shape

    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

    X = tf.constant(housing_data_plus_bias, dtype = tf.float32, name = "X")
    Y = tf.constant(housing.target.reshape(-1, 1), dtype = tf.float32, name = "Y")

    XT = tf.transpose(X)

    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), Y)


    with tf.Session() as sess:

        theta_value = theta.eval()

        
