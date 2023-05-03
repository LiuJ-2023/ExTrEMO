from typing import Optional

import torch
from linear_operator.operators import (
    DiagLinearOperator,
    InterpolatedLinearOperator,
    PsdSumLinearOperator,
    RootLinearOperator,
)

from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior
from gpytorch.kernels.kernel import Kernel


class TransferKernel(Kernel):
    r"""
    A transfer kernel for TGP

    .. math::

        \begin{equation}
            k(i, j) = [[1, lambda],
                       [lambda, 1]]
        \end{equation}

    Args:
        num_tasks (int):
            Total number of indices.
        batch_shape (torch.Size, optional):
            Set if the MultitaskKernel is operating on batches of data (and you want different
            parameters for each batch)
        rank (int):
            Rank of :math:`B` matrix. Controls the degree of
            correlation between the outputs. With a rank of 1 the
            outputs are identical except for a scaling factor.
        prior (:obj:`gpytorch.priors.Prior`):
            Prior for :math:`B` matrix.
        var_constraint (Constraint, optional):
            Constraint for added diagonal component. Default: `Positive`.

    Attributes:
        covar_factor:
            The :math:`B` matrix.
        raw_var:
            The element-wise log of the :math:`\mathbf v` vector.
    
    Note: This transfer kernel only supports two tasks.
    """    

    def __init__(
        self,
        lam = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if lam is not None:
            self.similarity = torch.tensor(lam)
        else:
            var_constraint = Interval(upper_bound=1,lower_bound=-1)
            self.register_parameter(
                name="raw_similarity", parameter=torch.nn.Parameter(1.0986*torch.ones(*self.batch_shape, 1))
                )
            self.register_constraint("raw_similarity", var_constraint)

    @property
    def similarity(self):
        return self.raw_similarity_constraint.transform(self.raw_similarity)

    @similarity.setter
    def similarity(self, value):
        self._set_similarity(value)

    def _set_similarity(self, value):
        self.initialize(raw_similarity=self.raw_similarity_constraint.inverse_transform(value))

    def _eval_covar_matrix(self):
        var1 = torch.tensor([[0,1],[1,0]])*self.similarity
        var2 = torch.tensor([[1,0],[0,1]])
        return var1+var2

    @property
    def covar_matrix(self):
        var1 = torch.tensor([[0,1],[1,0]])*self.similarity
        var2 = torch.tensor([[1,0],[0,1]])
        res = InterpolatedLinearOperator(var1+var2)
        return res

    def forward(self, i1, i2, **params):

        i1, i2 = i1.long(), i2.long()
        covar_matrix = self._eval_covar_matrix()
        batch_shape = torch.broadcast_shapes(i1.shape[:-2], i2.shape[:-2], self.batch_shape)

        res = InterpolatedLinearOperator(
            base_linear_op=covar_matrix,
            left_interp_indices=i1.expand(batch_shape + i1.shape[-2:]),
            right_interp_indices=i2.expand(batch_shape + i2.shape[-2:]),
        )
        return res
