import numpy as np
import cvxpy as cp
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import KFold
from conditionalconformal.condconf import setup_cvx_problem_calib

### Run cross validation to determine the hyperparameter in front of the kernel penalty
def runCV(XCalib,scoresCalib,kernel,gamma,alpha,k,minRad,maxRad,numRad,phiCalib):    
    radii = np.linspace(minRad,maxRad,numRad)
        
    folds = KFold(n_splits = k, shuffle = True)
    Klist = []
    for i, (trainIndex, testIndex) in enumerate(folds.split(XCalib)):
        Klist.append(pairwise_kernels(
            X=XCalib[testIndex,:],
            Y=XCalib[trainIndex,:],
            metric=kernel,
            gamma=gamma
        ))
               
    allLosses = np.zeros(len(radii))
    countR = 0
    for radius in radii:
        for i, (trainIndex, testIndex) in enumerate(folds.split(XCalib)):        
            prob = setup_cvx_problem_calib(1-alpha,XCalib[trainIndex,:],scoresCalib[trainIndex], phiCalib[trainIndex,:],
                                           {'kernel': 'rbf', 'gamma': gamma, 'lambda' : 1/radius})
            if "MOSEK" in cp.installed_solvers():
                prob.solve(solver="MOSEK")
            else:
                prob.solve()
            resid = (scoresCalib[testIndex] -
                    Klist[i] @ prob.var_dict['weights'].value -
                    phiCalib[testIndex,:]@prob.constraints[2].dual_value)
            loss = sum(0.5 * np.abs(resid) + (1 - alpha - 0.5)*resid) * (1/(len(testIndex)))
    
            allLosses[countR] = allLosses[countR] + loss/k
            
        countR = countR + 1
        
        
    return allLosses, radii

