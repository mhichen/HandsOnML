#!/usr/bin/python3

from sklearn.datasets import fetch_mldata
import scipy
import scipy.io as sio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


#mnist = fetch_mldata('MINST original')
mnist = sio.loadmat('/home/ivy/scikit_learn_data/mldata/mnist-original', squeeze_me = True)

X, y = mnist["data"].T, mnist["label"].T

# (70000, 784) - 70000 images with 784 features each
# np.sqrt(784) = 28 -- images are 28 x 28 pixels
print(X.shape)

# (70000, )
print(y.shape)

# Check out an example in the dataset
i_ind = 36000

some_digit = X[i_ind]
some_digit_image = some_digit.reshape(28, 28)

print("Showing image of digit", y[i_ind])
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")
plt.axis("off")
plt.show()

# Separate out train vs test
b_ind = 60000
X_train, X_test, y_train, y_test = X[:b_ind], X[b_ind:], y[:b_ind], y[b_ind:]

# Shuffle dataset
shuffle_ind = np.random.permutation(b_ind)
X_train, y_train = X_train[shuffle_ind], y_train[shuffle_ind]
