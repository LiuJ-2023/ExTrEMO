import gpytorch
import torch
from Utils import gp
from Utils import tgp
import numpy as np
from torchmin import Minimizer
from gpytorch.constraints import GreaterThan
import time

class FTGP():
    def __init__(self, train_xT, train_yT, train_xS, train_yS, beta = 0.5, opt_train = 'l-bfgs', likelihood = 'gaussian', kernel = 'RBF', transfer = 'transfer_kernel'):
        # Parameter settings
        self.beta = beta
        self.opt = opt_train
        self.source_num = len(train_yS)
        self.kernel = kernel
        self.likelihood = likelihood
        self.transfer = transfer
        
        # Built single task GP for the target
        self.train_xT = train_xT
        self.train_yT = train_yT
        self.build_gp()
        
        # Built multi task GP for each of the source-target pair
        self.train_xS = train_xS
        self.train_yS = train_yS
        self.build_tgp()

    def build_gp(self):
        self.model_t = gp.ExactGPModel(self.train_xT, self.train_yT, self.likelihood, kernel = self.kernel)
        self.model_t.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

    def build_tgp(self):
        self.full_train_x = []
        self.full_train_y = []
        self.full_train_i = []
        self.likelihood_s = []
        self.model_s = []
        for i in range(self.source_num):
            train_i_taskS = torch.zeros(self.train_xS[i].size(0),dtype=torch.long)
            train_i_taskT = torch.ones(self.train_xT.size(0),dtype=torch.long)
            full_train_x = torch.cat([self.train_xS[i], self.train_xT],0)
            full_train_i = torch.cat([train_i_taskS, train_i_taskT],0)
            full_train_y = torch.cat([self.train_yS[i], self.train_yT],0)
            model = tgp.TransferGPModel((full_train_x, full_train_i), full_train_y, likelihood = self.likelihood, kernel = self.kernel, transfer = self.transfer)
            likelihood = model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
            

            self.likelihood_s.append(likelihood)
            self.full_train_x.append(full_train_x)
            self.full_train_i.append(full_train_i)
            self.full_train_y.append(full_train_y)
            self.model_s.append(model)

    # Train a TBCM
    def train(self):
        # Train single task GP for target
        self.model_t.train()
        self.model_t.likelihood.train()
        if self.opt == 'adam':
            optimizer = torch.optim.Adam(params=self.model_t.parameters(),lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model_t.likelihood, self.model_t)
            training_iter = 200
            for i in range(training_iter):
                optimizer.zero_grad()
                output = self.model_t(self.train_xT)
                loss = -mll(output,self.train_yT)
                loss.backward()
                # if i%50== 0:
                optimizer.step()
            # print('Iter %d/%d - Loss: %.3f' % (i + 1, i, loss.item()))
        elif self.opt == 'l-bfgs':            
            optimizer = Minimizer(self.model_t.parameters(),
                          method='l-bfgs',
                          tol=1e-6,
                          max_iter=200,
                          disp=0)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model_t.likelihood, self.model_t)       
            def closure():
                optimizer.zero_grad()
                output = self.model_t(self.train_xT)
                loss = -mll(output,self.train_yT)
                # loss.backward()
                return loss
            optimizer.step(closure = closure)

            # optimizer = torch.optim.LBFGS(params=self.model_t.parameters(),lr=0.1,history_size=10, max_iter=15000, max_eval=15000)
            # mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model_t.likelihood, self.model_t)
            # training_iter = 1
            # for i in range(training_iter):
            #     def closure():
            #         optimizer.zero_grad()
            #         output = self.model_t(self.train_xT)
            #         loss = -mll(output,self.train_yT)
            #         loss.backward()
            #         return loss
            #     optimizer.step(closure = closure)
            # print('Iter %d/%d - Loss: %.3f' % (1, 1, closure().item()))

        # Train multi-task GP for each source-target pair
        self.similarities = []
        for j in range(self.source_num):
            # Set param that does not require training
            self.model_s[j].mean_module.constant.data = self.model_t.mean_module.constant.data
            self.model_s[j].covar_module.lengthscale = self.model_t.covar_module.lengthscale
            self.model_s[j].likelihood.noise_covar.raw_noise = self.model_t.likelihood.noise_covar.raw_noise
            self.model_s[j].mean_module.constant.requires_grad = False
            self.model_s[j].covar_module.raw_lengthscale.requires_grad = False
            self.model_s[j].likelihood.noise_covar.raw_noise.requires_grad = False

            # Tran TGPs
            self.model_s[j].train()
            self.model_s[j].likelihood.train()
            start = time.time()
            if self.opt == 'adam':
                optimizer = torch.optim.Adam(params=self.model_s[j].parameters(),lr=0.1)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model_s[j].likelihood, self.model_s[j])
                training_iter = 200
                for i in range(training_iter):
                    optimizer.zero_grad()
                    output = self.model_s[j](self.full_train_x[j],self.full_train_i[j])
                    loss = -mll(output,self.full_train_y[j])
                    loss.backward()
                    optimizer.step()
                # print('Iter %d/%d - Loss: %.3f' % (i + 1, i, loss.item()))
            elif self.opt == 'l-bfgs':
                optimizer = Minimizer(self.model_s[j].parameters(),
                          method='l-bfgs',
                          tol=1e-6,
                          max_iter=200,
                          disp=0)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model_s[j].likelihood, self.model_s[j])
                def closure():
                    optimizer.zero_grad()
                    output = self.model_s[j](self.full_train_x[j],self.full_train_i[j])
                    loss = -mll(output,self.full_train_y[j])
                    # loss.backward()
                    return loss
                optimizer.step(closure = closure)  

                # optimizer = torch.optim.LBFGS(params=self.model_s[j].parameters(), lr=0.1,history_size=20, max_iter=2000, max_eval=2000)
                # mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model_s[j].likelihood, self.model_s[j])
                # training_iter = 1
                # for i in range(training_iter):    
                #     def closure():
                #         optimizer.zero_grad()
                #         output = self.model_s[j](self.full_train_x[j],self.full_train_i[j])
                #         loss = -mll(output,self.full_train_y[j])
                #         loss.backward()
                #         return loss
                #     optimizer.step(closure = closure)
                # end = time.time()
                # # # print(end-start)
                # # print('Iter %d/%d - Loss: %.3f' % (1, 1, closure().item()))
            if self.transfer == 'transfer_kernel':
                self.similarities.append(np.abs(self.model_s[j].task_covar_module.similarity.detach().numpy().squeeze()))
            # print(self.model_s[j].task_covar_module.similarity.detach().numpy().squeeze())
        self.similarities = np.array(self.similarities)

    def LCB(self,x, mode = 'max'):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")

        X = torch.tensor(np.array(x))
        self.model_t.eval()
        for j in range(self.source_num):
            self.model_s[j].eval()

        M = len(self.model_s)
        # with torch.no_grad():
        predictions_s = []
        test_i_task = torch.ones(X.size(0), dtype=torch.long)
        for j in range(M):
            predictions_s.append(self.model_s[j](X, test_i_task))          
        prediction_t = self.model_t(X)

        # TBCM
        predicted_variance_inv_temp = torch.stack([1 / predictions_s[j].variance for j in range(M)]).sum(axis=0)
        predicted_variance_inv = predicted_variance_inv_temp + (1 - M) * 1.0 / prediction_t.variance
        predicted_variance_TBCM = 1 / predicted_variance_inv
        predicted_mean_TBCM = 0
        for j in range(M):
            predicted_mean_TBCM += (predictions_s[j].mean/ predictions_s[j].variance)
        predicted_mean_TBCM += (1 - M) * (prediction_t.mean/ prediction_t.variance)
        predicted_mean_TBCM = predicted_mean_TBCM * predicted_variance_TBCM

        # GPOE
        predicted_variance_GPOE = M / predicted_variance_inv_temp
        predicted_mean_GPOE = 0
        for j in range(M):
            predicted_mean_GPOE += (predictions_s[j].mean / predictions_s[j].variance)
        predicted_mean_GPOE = predicted_mean_GPOE * predicted_variance_GPOE / M

        # If not positive, use GPOE
        predicted_mean = (predicted_variance_inv>0)*predicted_mean_TBCM + (predicted_variance_inv<=0)*predicted_mean_GPOE
        predicted_variance = (predicted_variance_inv>0)*predicted_variance_TBCM + (predicted_variance_inv<=0)*predicted_variance_GPOE
        
        predicted_std = predicted_variance.sqrt()
        # Return UCB/LCB
        if mode == 'max':
            return -(predicted_mean - self.beta*predicted_std).detach().numpy()
        elif mode == 'min':
            return (predicted_mean - self.beta*predicted_std).detach().numpy()