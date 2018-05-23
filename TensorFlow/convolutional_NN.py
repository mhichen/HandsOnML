#!/usr/bin/python3

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import load_sample_images, load_sample_image

if __name__ == "__main__":

    china = load_sample_image("china.jpg")
    flower = load_sample_image("flower.jpg")
    dataset = np.array([china, flower], dtype = np.float32)
    batch_size, height, width, channels = dataset.shape


    filters = np.zeros(shape = (7, 7, channels, 2), dtype = np.float32)
    filters[:, 3, :, 0] = 1
    filters[3, :, :, 1] = 1

    X = tf.placeholder(shape = (None, height, width, channels), dtype = tf.float32)
    # s_h and s_w are both 2
    #convolution = tf.nn.conv2d(X, filters, strides = [1, 2, 2, 1], padding = "SAME")
    conv = tf.layers.conv2d(X, filters = 2, kernel_size = 7, strides = [2, 2], padding = "SAME")

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        init.run()
        
        output = sess.run(conv, feed_dict = {X: dataset})

    plt.imshow(output[1, :, :, 1], cmap = "gray")
    plt.show()
    
