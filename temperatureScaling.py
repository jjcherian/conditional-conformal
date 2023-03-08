#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:20:02 2023

@author: isaacgibbs
"""

import cvxpy as cvx
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torch.optim as optim

def temperature_scaling(X,Y):
    T = cvx.Variable()
        
    targetFeatures = np.zeros(len(Y))
    for i in range(len(Y)):
        targetFeatures[i] = X[i,int(Y[i])]
        
    prob = cvx.Problem(cvx.Minimize( -T*sum(targetFeatures) + cvx.sum([cvx.log_sum_exp(v) for v in T*X])   ))
    #print((T*X).shape)
    #prob = cvx.Problem(cvx.Minimize( -T*sum(targetFeatures) + cvx.sum(np.apply_along_axis(cvx.log_sum_exp,1,T*X)   )))
    
    prob.solve()
    
    return 1/T.value

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def fitted_probs(X,T):
    return np.apply_along_axis(softmax,1,X*T)
    

class myDataset(tdata.Dataset):
    def __init__(self, X, labels):
        self.labels = torch.tensor(labels)
        self.X = torch.tensor(X)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label = self.labels[idx]
        Xs = self.X[idx,:]
        return Xs, label

def torch_ts(X,Y, max_iters=1000, lr=0.1, epsilon=0.0001):
    dat = myDataset(X,Y)

    calib_loader = tdata.DataLoader(dat, batch_size=128, shuffle=True, pin_memory=True)
    nll_criterion = nn.CrossEntropyLoss()

    T = nn.Parameter(torch.Tensor([1.3]))

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T 

### Sanity check the fit
x = np.random.normal(size=(100,5))
theta = 5
y = np.zeros(100)
for i in range(100):
    probs = softmax(x[i,:]*5)
    y[i] = np.argmax(np.random.multinomial(1,probs))
    
T = temperature_scaling(x,y)
Tprime = torch_ts(x,y)
print(T)
print(Tprime)