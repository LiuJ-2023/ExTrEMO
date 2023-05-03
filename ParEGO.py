import os
import torch
import numpy as np
from numpy import ndarray
from Utils import decomposition
import random
from Utils import cga
from Utils import gp
from scipy.stats import norm, qmc
from scipy.optimize import minimize, Bounds
from functools import partial

# Decide the query solution
def optimize_acquisition_function(fsurr, problem, opt_method = 'l-bfgs'):
    if opt_method == 'l-bfgs':
        # Optimize LCB
        acq_f = partial(fsurr,mode='min')
        bounds = Bounds(lb=np.zeros(problem.dim),ub=np.ones(problem.dim))
        output = minimize(acq_f, problem.standard_bounds[1],method='L-BFGS-B',bounds=bounds )
        # Observe new solution 
        new_x = output.x
        new_x = new_x * (problem.standard_bounds[1] - problem.standard_bounds[0]) + problem.standard_bounds[0]
        new_x = np.array([new_x])
        new_obj_true = problem(new_x)
        new_obj = new_obj_true
        print(str(new_obj))
    elif opt_method == 'ga':
        acq_f = partial(fsurr,mode='max')
        new_x = cga.RKGA(acq_f, problem.dim, 10, 300)
        new_x = new_x * (problem.standard_bounds[1] - problem.standard_bounds[0]) + problem.standard_bounds[0]
        new_x = np.array([new_x])
        new_obj_true = problem(new_x)
        new_obj = new_obj_true
        print(str(new_obj))
    return new_x, new_obj

# Normalize
def source_to_torch(xS, yS, bounds):
    yS = (yS - yS.mean(0)) / yS.std(0)
    xS = (xS - bounds[0]) / (bounds[1] - bounds[0])
    train_xS = torch.from_numpy(xS)
    train_yS = torch.from_numpy(yS)
    return train_xS, train_yS

# ParEGO
def ParEGO(problem, 
    data_xT: ndarray = None, 
    data_yT: ndarray = None, 
    init_sample_size = 20, 
    max_iters = 50, 
    likelihood = 'gaussian',
    kernel = 'RBF',
    opt_train = 'l-bfgs',
    opt_acqf = 'l-bfgs'
    ):
    # Initialization
    if (data_xT is None) & (data_yT is None):
        sampler = qmc.LatinHypercube(d=problem.dim)
        xT = sampler.random(n=init_sample_size)
        yT = problem(xT)
    else:
        xT = data_xT
        yT = data_yT     
    
    # Iterations
    lamda_ = decomposition.generate_wv(problem.obj_num)   
    for iter in range(max_iters):
        # Generate weight for the sub-objective function
        lmda = random.choice(lamda_)

        # Generate target data for the sub-objective
        norm_y = decomposition.normalize(problem.obj_num, yT)[0]
        F = decomposition.scalarize_tchebycheff(norm_y, lmda, augmented=True)
        train_xT,train_yT = source_to_torch(xT,F,problem.standard_bounds)
        
        # Train GP
        model_t = gp.VanillaGP(train_xT, train_yT, opt_train = opt_train, likelihood = likelihood, kernel = kernel)
        model_t.train()

        # Decide the query solution
        new_x, new_y = optimize_acquisition_function(model_t.LCB, problem, opt_method=opt_acqf)
        xT = np.vstack((xT,new_x))
        yT = np.vstack((yT,new_y))

    return xT, yT