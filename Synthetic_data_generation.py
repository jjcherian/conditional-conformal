#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:15:20 2023

@author: isaacgibbs
"""

import pandas as pd 
import numpy as np
import math

from sklearn.model_selection import train_test_split


def get_groups(x):
    """
    Returns binary membership vectors for groups
    
    expects x to be a binary matrix of dimension (N, p)
    returns a membership matrix of dimension (N, 2 * p)
    """
    group_ids = np.zeros((x.shape[0], x.shape[1] * 2))
    for row in range(x.shape[0]):
        for i, x_val in enumerate(x[row,:]):
            group_ids[row, int(2 * i + x_val)] = 1
    return group_ids

def generate_group_synthetic_data(n, x_std, y_std, d, std_dev_list, theta, n_test, n_cal, n_groups):
    # d-dimension features - first 10 features are binary
    xs_binvars = np.random.randint(low = 0, high = 2, size = (n, n_groups))
    xs_remvars = np.random.normal(loc=np.zeros(d - n_groups), scale=x_std, size=(n, d - n_groups))
    xs = np.concatenate((xs_binvars, xs_remvars), axis = 1)

    std_dev = np.dot(xs_binvars, std_dev_list) + y_std
    ys = np.dot(xs, theta) + np.random.normal(loc=0, scale= std_dev, size=n)
    # Separate training data into training (for point-predictor) and calibration

    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=n_test, random_state=42)

    calibration_set_size = n_cal
    train_set_size = len(y_train) - calibration_set_size
    x_train_final = x_train[ : train_set_size]
    x_calib = x_train[train_set_size : ]
    y_train_final = y_train[ : train_set_size]
    y_calib = y_train[train_set_size : ]
    
    return x_train_final, y_train_final, x_calib, y_calib, x_test, y_test



def generate_cqr_data(n_train = 2000,n_test = 5000):
    def f(x):
        ''' Construct data (1D example)
        '''
        ax = 0*x
        for i in range(len(x)):
            ax[i] = np.random.poisson(np.sin(x[i])**2+0.1) + 0.03*x[i]*np.random.randn(1)
            ax[i] += 25*(np.random.uniform(0,1,1)<0.01)*np.random.randn(1)
        return ax.astype(np.float32)

    # training features
    x_train = np.random.uniform(0, 5.0, size=n_train).astype(np.float32)

    # test features
    x_test = np.random.uniform(0, 5.0, size=n_test).astype(np.float32)

    # generate labels
    y_train = f(x_train)
    y_test = f(x_test)

    # reshape the features
    x_train = np.reshape(x_train,(n_train,1))
    x_test = np.reshape(x_test,(n_test,1))
    
    calibration_set_size = math.floor(0.5*n_train)
    train_set_size = len(y_train) - calibration_set_size
    x_train_final = x_train[ : train_set_size]
    x_calib = x_train[train_set_size : ]
    y_train_final = y_train[ : train_set_size]
    y_calib = y_train[train_set_size : ]
    
    return x_train_final, y_train_final, x_calib, y_calib, x_test, y_test


def generate_ddim_cqr_data(n_train = 2000,n_test = 5000,d = 10):
    def f(x):
        ''' Construct data (1D example)
        '''
        ax = [0.0]*x.shape[0]
        for i in range(x.shape[0]):
            ax[i] = np.random.poisson(np.sin(x[i])**2+0.1) + 0.03*x[i]*np.random.randn(1)
            ax[i] += 25*(np.random.uniform(0,1,1)<0.01)*np.random.randn(1)
        return ax.astype(np.float32)

    # training features
    x_train = np.random.uniform(0, 5.0, size=n_train).astype(np.float32)

    # test features
    x_test = np.random.uniform(0, 5.0, size=n_test).astype(np.float32)

    # generate labels
    y_train = f(x_train)
    y_test = f(x_test)

    # reshape the features
    x_train = np.reshape(x_train,(n_train,1))
    x_test = np.reshape(x_test,(n_test,1))
    
    calibration_set_size = math.floor(0.5*n_train)
    train_set_size = len(y_train) - calibration_set_size
    x_train_final = x_train[ : train_set_size]
    x_calib = x_train[train_set_size : ]
    y_train_final = y_train[ : train_set_size]
    y_calib = y_train[train_set_size : ]
    
    return x_train_final, y_train_final, x_calib, y_calib, x_test, y_test

