import math
import torch
import gpytorch
import mtgp
import stgp
import scipy
import numpy as np

def build_stgp(train_x, train_y, kernel = 'RBF'):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = stgp.ExactGPModel(train_x, train_y, likelihood)

    return model.likelihood, model


def train_stgp(train_x, train_y, likelihood, model):
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.SGD([{'params': model.parameters()},], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iter = 200
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    return likelihood, model


def build_mtgp(train_xS, train_xT, train_yS, train_yT, kernel = 'RBF'):

    # S ==> represents source
    # T ==> represents target
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    train_i_taskS = torch.zeros(train_xS.size(0),dtype=torch.long)
    train_i_taskT = torch.ones(train_xT.size(0),dtype=torch.long)
    full_train_x = torch.cat([train_xS, train_xT],0)
    full_train_i = torch.cat([train_i_taskS, train_i_taskT],0)
    full_train_y = torch.cat([train_yS, train_yT],0)

    model = mtgp.MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood, kernel = kernel)

    return full_train_x, full_train_i, full_train_y, likelihood, model


def train_mtgp(full_train_x, full_train_i, full_train_y, likelihood, model, st_model, verbose = False):

    # set param that does not require training
    model.mean_module.constant.data = st_model.mean_module.constant.data
    model.covar_module.lengthscale = st_model.covar_module.lengthscale
    model.likelihood.noise_covar.raw_noise = st_model.likelihood.noise_covar.raw_noise
    model.mean_module.constant.requires_grad = False
    model.covar_module.raw_lengthscale.requires_grad = False
    model.likelihood.noise_covar.raw_noise.requires_grad = False

    # Training based on Chin Sheng's code
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(200):
        optimizer.zero_grad()
        output = model(full_train_x, full_train_i)
        loss = -mll(output, full_train_y)
        loss.backward()
        # if i%20 == 0:
        #     print('Iter %d/%d - Loss: %.3f' % (i + 1, i, loss.item()))
        optimizer.step()
        # if i == training_iter-1: loss = loss.item()
        # print('mt - Loss: %.3f' % (loss.item()))

    return likelihood, model

