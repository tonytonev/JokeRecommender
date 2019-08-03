#!/usr/bin/env python3

"""
Created on Wed Jul 31, 2019

@author: Tony Tonev
"""
import os
from model import *
import numpy as np
import pandas as pd

try:
    # If we already have a cache of test_R load that
    test_R = np.load('cache/test_R.npy')
    print('Loaded test_R from cache')
except:
    # Otherwise, read the test csv and reshape it to an R that is easier to work with
    test_df = pd.read_csv('data/test.csv', index_col = 'id')
    print('Reshaping test data')
    test_R = reshape_data_R(test_df)

    if not os.path.isdir('cache'):
        os.mkdir('cache')
    np.save('cache/test_R.npy', test_R)

try:
    # If we already have a model with trained parameters, use that
    X = np.load('cache/X.npy')
    Theta = np.load('cache/Theta.npy')

    print('Loaded X and Theta from cache')
except:
    # Otherwise, train a model and cache the parameters
    X, Theta = model(n_iterations = 200, learning_rate = 0.001, n_features = 200, lamb = 5, print_cost = True)

    np.save('cache/X.npy', X)
    np.save('cache/Theta.npy', Theta)


if not os.path.isdir('predictions'):
    os.mkdir('predictions')

# Make predictions and write them in the expected contest format
prediction_matrix = predict(X, Theta)
write_predictions(prediction_matrix, test_R, 'predictions/pred.csv')