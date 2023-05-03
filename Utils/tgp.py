import gpytorch
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.index_kernel import IndexKernel
from TransferKernel import TransferKernel
from typing import Any, Dict, List

class TransferGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='RBF',transfer = 'transfer_kernel'):
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

        super(TransferGPModel, self).__init__(train_x, train_y, likelihood_model)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == 'ARDMatern52':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims = train_x.size(1))
        elif kernel == 'ARDMatern52_with_gamma_prior':
            self.covar_module = MaternKernel(
                                    nu=2.5,
                                    ard_num_dims=train_x[0].size(1),
                                    lengthscale_prior=GammaPrior(3.0, 6.0)
                                )                                
        elif kernel == 'ARDRBF':
            self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims = train_x.size[0](1))
        elif kernel == 'ARDRBF_with_gamma_prior':
            self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims = train_x.size(1),lengthscale_prior=GammaPrior(3.0, 6.0))
        elif kernel == 'Matern52':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)
        elif kernel == 'RBF':
            self.covar_module = gpytorch.kernels.RBFKernel(lengthscale_constraint=GreaterThan(1e-4))

        if transfer == 'transfer_kernel':
            self.task_covar_module = TransferKernel()
        elif transfer == 'multitask_kernel':
            self.task_covar_module = IndexKernel(num_tasks=2)


    def forward(self,x,i):
        mean_x = self.mean_module(x)
        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)