#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:13:04 2023

@author: isaacgibbs
"""

import numpy as np
import cvxpy as cp
import mosek
from tqdm import tqdm

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import KFold

from rkhs import setup_full_cvx_dual


def setup_density_estimation_problem(K_cc,K_ct,nc,nt):
    rkhs_weight = cp.Variable(name="weights", shape=nc)
    intercept = cp.Variable(name="intercept")
    lamb = cp.Parameter(name="lambda", shape=(1,1), nonneg=True)
    
    K_ccVar = cp.Constant(K_cc)
    K_ctVar = cp.Constant(K_ct)
    L = cp.Constant(np.linalg.cholesky(K_cc))
    
    
    prob = cp.Problem(cp.Minimize( (1/2)*(1/nc)*cp.sum_squares(intercept + K_ccVar@rkhs_weight) +  lamb*cp.sum_squares(L.T@rkhs_weight) 
                                  - (1/nt)*cp.sum(intercept + K_ctVar@rkhs_weight)))
    
    return prob


def kernel_density_est(x_calib,x_target,kernel,gamma,k,min_radius,max_radius,num_radii):
    nc = x_calib.shape[0]
    nt = x_target.shape[0]

    
    if max_radius is None:
        max_radius = nc
    radii = np.linspace(min_radius,max_radius,num_radii)
        
    folds = KFold(n_splits = k)
    cvxProblemList = []
    Kcclist = []
    Kctlist = []
    KccTrainToTestList = []
    KctTrainToTestList = []
    for i, (train_index, train_test_index) in enumerate(folds.split(x_calib)):
        for j, (target_index, target_test_index) in enumerate(folds.split(x_target)):
            if i==j:
                Kcclist.append(pairwise_kernels(
                    X=x_calib[train_index,:],
                    metric=kernel,
                    gamma=gamma
                ) + 1e-5 * np.eye(len(train_index)))
                Kctlist.append(pairwise_kernels(
                            X=x_target[target_index,:],
                            Y=x_calib[train_index,:],
                            metric=kernel,
                            gamma=gamma
                ))
                KccTrainToTestList.append(pairwise_kernels(
                    X=x_calib[train_test_index,:],
                    Y=x_calib[train_index,:],
                    metric=kernel,
                    gamma=gamma
                ))
                KctTrainToTestList.append(pairwise_kernels(
                    X=x_target[target_test_index,:],
                    Y=x_calib[train_index,:],
                    metric=kernel,
                    gamma=gamma
                ))
                prob = setup_density_estimation_problem(Kcclist[i],Kctlist[i],len(train_index),len(target_index))

            cvxProblemList.append(prob)
        
        
    allLosses = np.zeros(len(radii))
    countR = 0
    for radius in tqdm(radii):
        for i, (train_index, train_test_index) in enumerate(folds.split(x_calib)):   
            for j, (target_index, target_test_index) in enumerate(folds.split(x_target)):
                if i==j:
                    cvxProblemList[i].param_dict['lambda'].value = np.asarray([[1/radius]])
                    cvxProblemList[i].solve(
                        solver='MOSEK', 
                        verbose=False, 
                        mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}
                    )
                    intercept = cvxProblemList[i].var_dict['intercept'].value
                    weights = cvxProblemList[i].var_dict['weights'].value
                    loss = ((1/2)*(1/len(train_test_index))*sum(np.square(intercept + KccTrainToTestList[i]@weights)) - 
                                   (1/len(target_test_index))*sum(intercept + KctTrainToTestList[i]@weights) )
    
                    allLosses[countR] = allLosses[countR] + loss/k
            
        countR = countR + 1
        
        
    selectedRadius = radii[np.argmin(allLosses)]
    print(allLosses)
    print('Selected Radius For Density Fit:',selectedRadius)
    
    
    K_cc = pairwise_kernels(
            X=x_calib,
            metric=kernel,
            gamma=gamma
    )
    K_cc_stable = K_cc + 1e-5 * np.eye(nc)
    
    K_ct = pairwise_kernels(
            X=x_target,
            Y=x_calib,
            metric=kernel,
            gamma=gamma
    )

    rkhs_weight = cp.Variable(name="weights", shape=nc)
    intercept = cp.Variable(name="intercept")
    
    prob = cp.Problem(
        cp.Minimize( (1/2)*(1/nc)*cp.sum_squares(intercept + K_cc_stable@rkhs_weight) - (1/nt)*cp.sum(intercept + K_ct@rkhs_weight) +
                     (1/selectedRadius)*cp.sum_squares(np.linalg.cholesky(K_cc_stable).T@rkhs_weight)  )
    )
    prob.solve(
        solver='MOSEK', 
        verbose=False, 
        mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}
    )
        
    return K_cc, prob.var_dict['weights'].value, radii, allLosses

def compute_coverage_under_density_shift(x_calib,scores_calib,x_test,kernel,gamma,alpha,radius,
                                                   k, min_radius, max_radius, num_radii):
    K_cc, densityWeights, _ , _ = kernel_density_est(x_calib,x_test,kernel,gamma,
                                                   k, min_radius, max_radius, num_radii)
    
    prob = setup_full_cvx_dual(
                x_calib,
                scores_calib,
                kernel=kernel,
                gamma=gamma,
                alpha=alpha,
                z_calib = None
            )
    prob.param_dict['radius'].value = np.asarray([[radius]])
    prob.solve(
            solver='MOSEK', 
            verbose=False, 
            mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}
          )
    
    
    inner_prod = prob.var_dict['weights'].value @ K_cc @ densityWeights
    g_sum = np.mean( K_cc@densityWeights )
    return alpha - (1/radius)*(1/g_sum)*inner_prod

    
    
    