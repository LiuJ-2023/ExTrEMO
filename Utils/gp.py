import gpytorch
from gpytorch.constraints.constraints import GreaterThan, Interval, Positive
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from torchmin import Minimizer
import torch
import numpy as np

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood='gaussain', kernel='RBF'):  
        if likelihood == 'gaussian':
            likelihood_model = GaussianLikelihood(noise_constraint=Interval(lower_bound=1e-4,upper_bound=5))
        elif likelihood == 'gaussian_with_gamma_prior':
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood_model = GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=GreaterThan(
                    1e-4,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood_model)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == 'ARDMatern52':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims = train_x.size(1))
        elif kernel == 'ARDMatern52_with_gamma_prior':
            self.covar_module = MaternKernel(
                                    nu=2.5,
                                    ard_num_dims=train_x.size(1),
                                    lengthscale_prior=GammaPrior(3.0, 6.0)
                                )
        elif kernel == 'ARDRBF':
            self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims = train_x.size(1))
        elif kernel == 'ARDRBF_with_gamma_prior':
            self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims = train_x.size(1),lengthscale_prior=GammaPrior(3.0, 6.0))
        elif kernel == 'Matern52':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)
        elif kernel == 'RBF':
            self.covar_module = gpytorch.kernels.RBFKernel(lengthscale_constraint=GreaterThan(1e-4))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class VanillaGP():
    def __init__(self, train_xT, train_yT, beta = 0.5, opt_train = 'l-bfgs', likelihood = 'gaussian', kernel = 'RBF'):
        # Parameter settings
        self.beta = beta
        self.opt = opt_train
        self.kernel = kernel
        self.likelihood = likelihood
        
        # Built single task GP for the target
        self.train_xT = train_xT
        self.train_yT = train_yT
        self.build_gp()

    # Built a single-task GP
    def build_gp(self):
        self.model_t = ExactGPModel(self.train_xT, self.train_yT, self.likelihood, kernel = self.kernel)
        self.model_t.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

    # Train a single-task GP
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
                #     print('Iter %d/%d - Loss: %.3f' % (i + 1, i, loss.item()))
                optimizer.step()
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

            # optimizer = torch.optim.LBFGS(params=self.model_t.parameters(),
            #         lr = 0.1, 
            #         line_search_fn="strong_wolfe")
            # mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model_t.likelihood, self.model_t)
            # mll = mll.to(self.train_xT)
            # training_iter = 10
            # for i in range(training_iter):    
            #     def closure():
            #         optimizer.zero_grad()
            #         output = self.model_t(self.train_xT)
            #         loss = -mll(output,self.train_yT)
            #         loss.backward()
            #         return loss
            #     optimizer.step(closure = closure)
            # print('Iter %d/%d - Loss: %.3f' % (1, 1, closure().item()))

    def LCB(self,x, mode = 'max'):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")

        X = torch.tensor(np.array(x))
        self.model_t.eval()         
        prediction_t = self.model_t(X)
        predicted_std = prediction_t.variance.sqrt()
        predicted_mean = prediction_t.mean
        if mode == 'max':
            return -(predicted_mean - self.beta*predicted_std).detach().numpy()
        elif mode == 'min':
            return (predicted_mean - self.beta*predicted_std).detach().numpy()