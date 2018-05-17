#!/usr/bin/python3

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
        
if __name__ == "__main__":

    x = tf.Variable(3, name = "x")
    y = tf.Variable(4, name = "y")
    f = x * x * y + y + 2

    # One way to run a session
    sess = tf.Session()
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)
    print(result)

    sess.close()


    # Another way to run a session
    with tf.Session() as sess:
        x.initializer.run()
        y.initializer.run()
        result = f.eval()
        print(result)
        

    # Another way to do things
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        result = f.eval()
