#!/usr/bin/python3

import os
import tarfile
from six.moves import urllib

import pandas as pd

import scipy
import scipy.io as sio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import real_estate_functions as ref

from sklearn.model_selection import train_test_split

## Build model of housing prices in California using census data

def split_train_test(data, test_ratio, seed = None):

    if seed is not None:
        np.random.seed(seed)

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == "__main__":

    #ref.fetch_housing_data()
    
    housing = ref.load_housing_data()

    print("Peek at first few lines")
    print(housing.head())
    print()


    print("Summary of data")
    print(housing.info())
    print()


    print("count categorical instances of ocean_proximity")
    print(housing["ocean_proximity"].value_counts())
    print()

    print("Look at summary of numerical fields")
    print(housing.describe())
    print()

    # Plot a histogram of th
    housing.hist(bins = 50, figsize = (20, 15))
    #plt.show()

    

    # Split training and test data
    #train_set, test_set = split_train_test(housing, 0.2, 20180405)

    
    train, test = train_test_split(housing, test_size = 0.2, random_state = 42)
    print("Size of train data", len(train))
    print("Size of test data", len(test))
    
