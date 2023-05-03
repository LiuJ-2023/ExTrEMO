# This Python code is for ExTrEMO
# Implemented by Jiao Liu (jiao.liu@ntu.edu.sg) and Abhishek Gputa (abhishek_gupta@simtech.a-star.edu.sg)

import os
import torch
import numpy as np
from numpy import ndarray
from Utils import decomposition
import random
from Utils import cga
from Utils import ftgp
from scipy.stats import norm, qmc
from scipy.optimize import minimize, Bounds
from functools import partial
import time
from datetime import datetime

# Decide the query solution
def optimize_acquisition_function(fsurr, problem, opt_method = 'l-bfgs'):
    if opt_method == 'l-bfgs':
        # Optimize LCB
        acq_f = partial(fsurr,mode='min')
        bounds = Bounds(lb=np.zeros(problem.dim),ub=np.ones(problem.dim))
        output = minimize(acq_f, np.ones(problem.dim), method='L-BFGS-B',bounds=bounds )
        # Observe new solution 
        new_x = output.x
        new_x = new_x * (problem.standard_bounds[1] - problem.standard_bounds[0]) + problem.standard_bounds[0]
        new_x = np.array([new_x])
        new_obj_true = problem(new_x)
        new_obj = new_obj_true
        print(str(new_obj))
    elif opt_method == 'ga':
        acq_f = partial(fsurr,mode='max')
        new_x = cga.RKGA(acq_f, problem.dim, 10, 200)
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

# SoftMax
def softmax(x,l=1):
    t = np.exp(x/l)
    prob = t/np.sum(t)
    return prob

# ExTrEMO
def ExTrEMO(problem, 
    data_xS: list[ndarray], 
    data_yS: list[ndarray], 
    data_xT: ndarray = None, 
    data_yT: ndarray = None, 
    init_sample_size = 20, 
    max_iters = 30,
    likelihood = 'gaussian',
    kernel = 'RBF',
    transfer = 'transfer_kernel',
    opt_train = 'l-bfgs',
    opt_acqf = 'ga',
    subset_sel = None,
    q_sub = 5,
    ):
    ################# initialization #################
    if (data_xT is None) & (data_yT is None):
        sampler = qmc.LatinHypercube(d=problem.dim)
        xT = sampler.random(n=init_sample_size)
        yT = problem(xT)
    else:
        xT = data_xT
        yT = data_yT     

    if (type(data_xS) is not list) | (type(data_yS) is not list):
        raise Exception("Source data should be in list form.")
    else:
        if (len(data_xS) == len(data_yS)):
            # # Control the number of samples in each source
            # for kk in range(len(data_yS)):
            #     data_xS[kk] = data_xS[kk][:20]
            #     data_yS[kk] = data_yS[kk][:20]
            xS = data_xS
            yS = data_yS                                
        else:
            raise Exception("The length of x source and y souce should have the same length.")
    
    ################# Iterations #################
    lamda_ = decomposition.generate_wv(problem.obj_num)   
    Index = np.array(range(len(data_xS)))
    Pi = 0.5*np.ones(len(data_xS))
    Prob = softmax(Pi,0.05)
    record_sel_task = []
    for iter in range(max_iters):
        # Generate weight for the sub-problem
        lmda = random.choice(lamda_)

        # Generate sources data for the sub-problem
        train_xS, train_yS = [], []
        for j in range(len(yS)):
            norm_yS = decomposition.normalize(problem.obj_num, yS[j])[0]
            FS = decomposition.scalarize_tchebycheff(norm_yS, lmda, augmented=True)   
            t_xS, t_yS = source_to_torch(xS[j], FS, problem.standard_bounds)
            train_xS.append(t_xS)
            train_yS.append(t_yS)

        # Generate target data for the sub-problem
        norm_y = decomposition.normalize(problem.obj_num, yT)[0]
        F = decomposition.scalarize_tchebycheff(norm_y, lmda, augmented=True)
        train_xT,train_yT = source_to_torch(xT,F,problem.standard_bounds)
        
        # Surrogate Building
        if subset_sel == 'es_select': 
            # Evolutionary Source Selection
            selected_tasks = np.random.choice(Index, p=Prob, replace=False ,size=q_sub)
            train_xS_select = [train_xS[t] for t in selected_tasks]
            train_yS_select = [train_yS[t] for t in selected_tasks]
            model_ftgp = ftgp.FTGP(train_xT, train_yT, train_xS_select, train_yS_select, opt_train = opt_train, likelihood = likelihood, kernel = kernel)
            model_ftgp.train()

            # Update the average of the history correlation coefficients 
            if iter == 0:
                F0 = np.mean(np.abs(model_ftgp.similarities))
            else:
                F0 = F0 + (np.mean(np.abs(model_ftgp.similarities)) - F0)/(iter+1)

            # Update Pi
            dPi = np.zeros(len(data_xS))
            dPi[selected_tasks] = (np.abs(model_ftgp.similarities) - F0)*(1-Prob[selected_tasks])
            Pi = Pi + dPi
            record_sel_task.append(selected_tasks)

            # Calculate the probabilities for selection     
            Prob = softmax(Pi,0.05)

        
        elif subset_sel == 'random':
            # Select sources uniformly
            selected_tasks = np.random.choice(Index, replace=False ,size=q_sub)
            train_xS_select = [train_xS[t] for t in selected_tasks]
            train_yS_select = [train_yS[t] for t in selected_tasks]
            model_ftgp = ftgp.FTGP(train_xT, train_yT, train_xS_select, train_yS_select, opt_train = opt_train, likelihood = likelihood, kernel = kernel)
            model_ftgp.train()     
            record_sel_task.append(selected_tasks)  
        
        
        elif subset_sel is None:
            # Build factorized TGP by using all of the sources
            model_ftgp = ftgp.FTGP(train_xT, train_yT, train_xS, train_yS, opt_train = opt_train, likelihood = likelihood, kernel = kernel, transfer = transfer)
            model_ftgp.train()

        # Evolutionary solution generation and evaluation
        new_x, new_y = optimize_acquisition_function(model_ftgp.LCB, problem, opt_method = opt_acqf)
        xT = np.vstack((xT,new_x))
        yT = np.vstack((yT,new_y))

    return xT, yT, np.array(record_sel_task)