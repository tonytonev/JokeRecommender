#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:23:34 2018

@author: Tony
"""

import pandas as pd
import numpy as np
import os


from helpers import *


# Randomly initialize X and Theta based on the dimensions given
#
# n_users is the number of users
# n_jokes is the number of users
# n_features is the number of features to use
#
# Returns a tuple of (X, Theta)

def initialize_parameters(n_users, n_jokes, n_features):
    X = np.random.randn(n_jokes, n_features)*0.01
    Theta = np.random.randn(n_users, n_features)*0.01
    
    return X, Theta

# Compute the cost for parameters X and Theta for the ground truth defined by Y and R with regularization parameter lamb
#
# X is (num_jokes x num_features) size parameter matrix
# Theta is a (num_users x num_features) size parameter matrix
# Y is a (num_jokes x num_users) size matrix which contains the value of the users' preferences in a value from 0 to 1
# R is a (num_jokes x num_users) size indicator matrix which is 1 if that user has rated that joke and 0 otherwise
# lamb (short for lambda) is a float parameter controlling regularization
#
# Returns a single float cost value

def cost(X, Theta, Y, R, lamb=0):
    return 1/2*np.sum(np.power((np.dot(X, Theta.T) - Y) * R, 2)) + \
           lamb/2*np.sum(np.power(X, 2)) + lamb/2*np.sum(np.power(Theta, 2))

# Compute numerical gradients to sanity check the analytic gradients
#
# X is (num_jokes x num_features) size parameter matrix
# Theta is a (num_users x num_features) size parameter matrix
# Y is a (num_jokes x num_users) size matrix which contains the value of the users' preferences in a value from 0 to 1
# R is a (num_jokes x num_users) size indicator matrix which is 1 if that user has rated that joke and 0 otherwise
#
# Returns a tuple of (X_grad, Theta_grad)

def num_gradients(X, Theta, Y, R):
    n_users = np.shape(Theta)[0]
    n_jokes = np.shape(X)[0]
    n = np.shape(X)[1]

    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    curr_cost = cost(X, Theta, Y, R)
    delta = 0.000001
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_temp = X[:,:]
            X_temp[i,j] += delta
            
            X_grad[i,j] = (curr_cost - cost(X_temp, Theta, Y, R))/delta
    for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            Theta_temp = Theta[:,:]
            Theta_temp[i,j] += delta
            
            Theta_grad[i,j] = (curr_cost - cost(X, Theta_temp, Y, R))/delta

    return X_grad, Theta_grad

# Compute gradients for parameters X and Theta
#
# X is (num_jokes x num_features) size parameter matrix
# Theta is a (num_users x num_features) size parameter matrix
# Y is a (num_jokes x num_users) size matrix which contains the value of the users' preferences in a value from 0 to 1
# R is a (num_jokes x num_users) size indicator matrix which is 1 if that user has rated that joke and 0 otherwise
#
# Returns a tuple of (X_grad, Theta_grad)

def gradients(X, Theta, Y, R, lamb=0):
    n_users = np.shape(Theta)[0]
    n_jokes = np.shape(X)[0]
    n = np.shape(X)[1]

    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    for i in range(n_jokes):
        idx = np.where(R[i,:] == 1)
        Theta_temp = Theta[idx]
        Y_temp = Y[i, idx]

        #print("X: %s, Y_temp: %s, Theta_temp: %s" % (X.shape, Y_temp.shape, Theta_temp.shape))
        X_grad[i] = np.dot(np.dot(X[i, :], Theta_temp.T) - Y_temp, Theta_temp) + lamb*X[i]

    for j in range(n_users):
        idx = np.where(R[:,j] == 1)
        X_temp = X[idx]
        Y_temp = Y[idx, j]

        #print("Theta: %s, Y_temp: %s, X_temp: %s, j: %s" % (Theta.shape, Y_temp.shape, X_temp.shape, j))
        Theta_grad[j] = np.dot(np.dot(X_temp, Theta[j, :].T) - Y_temp, X_temp) + lamb*Theta[j]

    return X_grad, Theta_grad

# Test calculating the gradient with random small values and compare numerical to analytic gradients

def test_grad():
    n_users = 3
    n_jokes = 5
    n = 2

    print("Test initalize parameters: ")
    print("n_users: %s, n_jokes: %s, n: %s" % (n_users, n_jokes, n))
    X, Theta = initialize_parameters(n_users, n_jokes, n)
    print("X: %s\nTheta: %s" % (X, Theta))

    Y = np.random.random((n_jokes, n_users))
    R = (np.random.random((n_jokes, n_users)) > 0.5) * 1.0

    print("Y: %s\n R: %s" % (Y, R))

    print("Cost: " + str(cost(X, Theta, Y, R)))

    X_grad_num, Theta_grad_num = num_gradients(X, Theta, Y, R)
    X_grad, Theta_grad = gradients(X, Theta, Y, R)

    print("Grad Diff X: %s\nGrad Diff Theta: %s" % (X_grad_num - X_grad, Theta_grad_num - Theta_grad))

# Train a model to predict joke preferences of users using collaborative filtering. 
#
# n_iterations is the number of iterations to train for
# learning_rate is the learning rate for gradient descent
# n_features is the number of features to learn
# lamb (short for lambda) is the regularization parameter
# print_cost is whether to print the cost while training or not
# params is an optional parameter containing a tuple of X and Theta, if we have a previous model
#        we want to continue training
#
# Returns trained parameter X and Theta

def model(n_iterations = 200, learning_rate = 0.001, n_features = 100, lamb = 10, print_cost = True, params = None):
    # Load CSV's
    jokes = pd.read_csv('data/jokes.csv')
    ratings = pd.read_csv('data/train.csv')

    n_users = len(ratings.user_id.unique())
    n_jokes = len(ratings.joke_id.unique())
    
    try:
        # Try to load cached Y and R from file, so we don't have to reshape the data each time
        Y = np.load('cache/Y.npy')
        R = np.load('cache/R.npy')
    except:
        if print_cost:
            print('Reshaping training data to get Y and R')
        Y, R = reshape_data(ratings)

        np.save('cache/Y.npy', Y)
        np.save('cache/R.npy', R)

    X = None
    Theta = None
    if (params):
        # If params are given continue training existing model
        X, Theta = params
    else:
        # If no params given, initialize X and Theta with 100 features
        X, Theta = initialize_parameters(n_users, n_jokes, n_features)

    J = None
    start_time = time.time()
    last_print = start_time

    for itr in range(n_iterations):
        # Calculate cost
        J = cost(X, Theta, Y, R, lamb)

        if print_cost:
            last_print = print_progress(itr, n_iterations, start_time, last_print,
                        "Training model. Iteration: %s of %s, Cost: %.0f" % (itr, n_iterations, J))
        
        # Calculate gradients   
        X_grad, Theta_grad = gradients(X, Theta, Y, R, lamb)      

        # Update paramters
        X -= learning_rate * X_grad
        Theta -= learning_rate * Theta_grad

    if print_cost:
        print("Finished training model in %.1f minutes. Cost: %.0f" % ((time.time() - start_time) / 60, J))

    return X, Theta

# Returns predictions for parameters X and Theta with predicted ratings with values between -10 and 10
# to match the original training set. Row i corresponds to joke_id i+1. Column j corresponds to user_id j+1.
#
# X is (num_jokes x num_features) size parameter matrix
# Theta is a (num_users x num_features) size parameter matrix
#
# Returns prediction matrix Y of size (num_jokes, num_users)

def predict(X, Theta):
    return np.minimum(np.maximum((np.dot(X, Theta.T) * 20) - 10, -10), 10)

