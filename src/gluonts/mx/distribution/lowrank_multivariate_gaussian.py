# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import math
from typing import Optional, Tuple

import numpy as np
import mxnet as mx

from gluonts.core.component import validated
from gluonts.mx import Tensor

from . import bijection
from .distribution import Distribution, _sample_multiple, getF
from .distribution_output import (
    ArgProj,
    DistributionOutput,
    AffineTransformedDistribution,
)
from gluonts.mx.distribution.transformed_distribution import TransformedDistribution

sigma_minimum = 1e-3


# sort samples in quantile bins
def get_quantile(sorted_samples, q):
    # sorted_samples has shape = (num_samples,batch_size,seq_len,1), dimension is fixed at this point
    # sorted_samples = mx.nd.squeeze(sorted_samples, axis=-1) #remove dim axis which only has length 1

    # same sample_idx *for each* batch_size and seq_len point.
    num_samples = sorted_samples.shape[0]
    sample_idx = int(np.round((num_samples - 1) * q))  # round up because y >= q_pred

    return sorted_samples[sample_idx, :, :]  # return dim is (batch_size, seq_len)


# compute quantile loss for single quantile
def quantile_loss(sorted_samples, y, q):
    # sorted_samples has shape = (num_samples,batch_size,seq_len,1)
    # q is a scalar

    # I think get_quantile function is outside of the mxnet 'path'
    # quantile_pred has shape = (batch_size,seq_len,1)
    quantile_pred = get_quantile(sorted_samples, q)  # shape = (batch_size, seq_len, 1)

    assert (y.shape == quantile_pred.shape)

    # return shape is (batch_size,seq_len,1)
    return mx.nd.where(
        y >= quantile_pred,
        q * (y - quantile_pred),  # if >=
        (1 - q) * (quantile_pred - y)
    )

def capacitance_tril(F, rank: Tensor, W: Tensor, D: Tensor) -> Tensor:
    r"""

    Parameters
    ----------
    F
    rank
    W : (..., dim, rank)
    D : (..., dim)

    Returns
    -------
        the capacitance matrix :math:`I + W^T D^{-1} W`

    """
    # (..., dim, rank)
    Wt_D_inv_t = F.broadcast_div(W, D.expand_dims(axis=-1))

    # (..., rank, rank)
    K = F.linalg_gemm2(Wt_D_inv_t, W, transpose_a=True)

    # (..., rank, rank)
    Id = F.broadcast_mul(F.ones_like(K), F.eye(rank))

    # (..., rank, rank)
    return F.linalg.potrf(K + Id)


def log_det(F, batch_D: Tensor, batch_capacitance_tril: Tensor) -> Tensor:
    r"""
    Uses the matrix determinant lemma.

    .. math::
        \log|D + W W^T| = \log|C| + \log|D|,

    where :math:`C` is the capacitance matrix :math:`I + W^T D^{-1} W`, to compute the log determinant.

    Parameters
    ----------
    F
    batch_D
    batch_capacitance_tril

    Returns
    -------

    """
    log_D = batch_D.log().sum(axis=-1)
    log_C = 2 * F.linalg.sumlogdiag(batch_capacitance_tril)
    return log_C + log_D


def mahalanobis_distance(
    F, W: Tensor, D: Tensor, capacitance_tril: Tensor, x: Tensor
) -> Tensor:
    r"""
    Uses the Woodbury matrix identity

    .. math::
        (W W^T + D)^{-1} = D^{-1} - D^{-1} W C^{-1} W^T D^{-1},

    where :math:`C` is the capacitance matrix :math:`I + W^T D^{-1} W`, to compute the squared
    Mahalanobis distance :math:`x^T (W W^T + D)^{-1} x`.

    Parameters
    ----------
    F
    W
        (..., dim, rank)
    D
        (..., dim)
    capacitance_tril
        (..., rank, rank)
    x
        (..., dim)

    Returns
    -------

    """
    xx = x.expand_dims(axis=-1)

    # (..., rank, 1)
    Wt_Dinv_x = F.linalg_gemm2(
        F.broadcast_div(W, D.expand_dims(axis=-1)), xx, transpose_a=True
    )

    # compute x^T D^-1 x, (...,)
    maholanobis_D_inv = F.broadcast_div(x.square(), D).sum(axis=-1)

    # (..., rank)
    L_inv_Wt_Dinv_x = F.linalg_trsm(capacitance_tril, Wt_Dinv_x).squeeze(
        axis=-1
    )

    maholanobis_L = L_inv_Wt_Dinv_x.square().sum(axis=-1).squeeze()

    return F.broadcast_minus(maholanobis_D_inv, maholanobis_L)


def lowrank_log_likelihood(
    rank: int, mu: Tensor, D: Tensor, W: Tensor, x: Tensor
) -> Tensor:

    F = getF(mu)

    dim = F.ones_like(mu).sum(axis=-1).max()

    dim_factor = dim * math.log(2 * math.pi)

    if W is not None:
        batch_capacitance_tril = capacitance_tril(F=F, rank=rank, W=W, D=D)

        log_det_factor = log_det(
            F=F, batch_D=D, batch_capacitance_tril=batch_capacitance_tril
        )

        mahalanobis_factor = mahalanobis_distance(
            F=F, W=W, D=D, capacitance_tril=batch_capacitance_tril, x=x - mu
        )
    else:
        log_det_factor = D.log().sum(axis=-1)
        x_centered = x - mu
        mahalanobis_factor = F.broadcast_div(x_centered.square(), D).sum(axis=-1)

    ll: Tensor = -0.5 * (
        F.broadcast_add(dim_factor, log_det_factor) + mahalanobis_factor
    )

    return ll


class LowrankMultivariateGaussian(Distribution):
    r"""
    Multivariate Gaussian distribution, with covariance matrix parametrized
    as the sum of a diagonal matrix and a low-rank matrix

    .. math::
        \Sigma = D + W W^T

    When `W = None` the covariance matrix is just diagonal.

    The implementation is strongly inspired from Pytorch:
    https://github.com/pytorch/pytorch/blob/master/torch/distributions/lowrank_multivariate_normal.py.

    Complexity to compute log_prob is :math:`O(dim * rank + rank^3)` per element.

    Parameters
    ----------
    dim
        Dimension of the distribution's support
    rank
        Rank of W
    mu
        Mean tensor, of shape (..., dim)
    D
        Diagonal term in the covariance matrix, of shape (..., dim)
    W
        Low-rank factor in the covariance matrix, of shape (..., dim, rank)
    """

    is_reparameterizable = True

    @validated()
    def __init__(
        self, dim: int, rank: int, mu: Tensor, D: Tensor, W: Optional[Tensor] = None
    ) -> None:
        self.dim = dim
        self.rank = rank
        self.mu = mu
        self.D = D
        self.W = W
        self.Cov = None

    @property
    def F(self):
        return getF(self.mu)

    @property
    def batch_shape(self) -> Tuple:
        return self.mu.shape[:-1]

    @property
    def event_shape(self) -> Tuple:
        return self.mu.shape[-1:]

    @property
    def event_dim(self) -> int:
        return 1

    def log_prob(self, x: Tensor) -> Tensor:
        return lowrank_log_likelihood(
            rank=self.rank, mu=self.mu, D=self.D, W=self.W, x=x
        )

    def crps(self, samples: Tensor, y: Tensor, quantiles=np.arange(0.1, 1.0, 0.1)) -> Tensor:
        r"""
        Compute the *continuous rank probability score* (CRPS) of `y` according
        to the distribution.

        Parameters
        ----------
        samples
            Tensor of shape `(*batch_shape, *event_shape)`.
        y
            Tensor of ground truth

        Returns
        -------
        Tensor
            Tensor of shape `batch_shape` containing the CRPS score,
            according to the distribution, for each event in `x`.
        """
        # y is ground truth. Has shape (batch_size, seq_len, m)
        # samples has shape = (num_samples, batch_size, seq_len, m)

        # sum over m axis, sum over T axis, sum over bs axis

        # loss for single ground truth point across all dimensions
        # loop through dimensions
        losses = []
        for d in range(samples.shape[-1]):

            # dim of dim_slice = (num_samples,batch_size,seq_len,1)
            dim_slice = mx.nd.slice_axis(samples, axis=-1, begin=d, end=d + 1)

            # sort samples along sample axis. shape = (num_samples,batch_size,seq_len,1)
            sorted_slice = mx.nd.sort(dim_slice, axis=0)  # sort along sample axis (first axis)

            # slice of y for dimension d. shape = (batch_size, seq_len,1)
            y_slice = mx.nd.slice_axis(y, axis=-1, begin=d, end=d + 1)

            # compute quantile loss, shape = (batch_size, seq_len, 1)
            #qloss = mx.nd.zeros((y_slice.shape))
            qlosses = []
            for q in quantiles:
                qlosses.append(quantile_loss(sorted_slice, y_slice, q))
            #qloss = quantile_loss(sorted_slice, y_slice, .1)
            qloss = mx.nd.stack(*qlosses, axis=-1) #shape = (batch_size, seq_len, 1, Q)

            #take average
            qloss = (1/len(qlosses)) * mx.nd.sum(qloss, axis=-1)  #shape = (batch_size, seq_len,1)

            # append qloss tensor
            losses.append(mx.nd.squeeze(qloss))  # remove dummy last axis of dim_slice and append

        loss = mx.nd.stack(*losses, axis=-1)  # shape = (batch_size, seq_len,m)
        #return mx.nd.sum(loss, axis=-1).expand_dims(-1)  # shape = (batch_size, seq_len,1)
        return loss #shape = (batch_size, seq_len,m)

    @property
    def mean(self) -> Tensor:
        return self.mu

    @property
    def variance(self) -> Tensor:
        assert self.dim is not None
        F = self.F

        if self.Cov is not None:
            return self.Cov
        # reshape to a matrix form (..., d, d)
        D_matrix = self.D.expand_dims(-1) * F.eye(self.dim)

        if self.W is not None:
            W_matrix = F.linalg_gemm2(self.W, self.W, transpose_b=True)
            self.Cov = D_matrix + W_matrix
        else:
            self.Cov = D_matrix

        return self.Cov

    def sample_rep(self, num_samples: Optional[int] = None, dtype=np.float32) -> Tensor:
        r"""
        Draw samples from the multivariate Gaussian distribution:

        .. math::
            s = \mu + D u + W v,

        where :math:`u` and :math:`v` are standard normal samples.

        Parameters
        ----------
        num_samples
            number of samples to be drawn.
        dtype
            Data-type of the samples.

        Returns
        -------
            tensor with shape (num_samples, ..., dim)
        """

        def s(mu: Tensor, D: Tensor, W: Optional[Tensor]=None) -> Tensor:
            F = getF(mu)

            samples_D = F.sample_normal(
                mu=F.zeros_like(mu), sigma=F.ones_like(mu), dtype=dtype
            )
            cov_D = D.sqrt() * samples_D

            if W is not None:
                # dummy only use to get the shape (..., rank, 1)
                dummy_tensor = F.linalg_gemm2(
                    W, mu.expand_dims(axis=-1), transpose_a=True
                ).squeeze(axis=-1)

                samples_W = F.sample_normal(
                    mu=F.zeros_like(dummy_tensor),
                    sigma=F.ones_like(dummy_tensor),
                    dtype=dtype,
                )

                cov_W = F.linalg_gemm2(W, samples_W.expand_dims(axis=-1)).squeeze(
                    axis=-1
                )

                samples = mu + cov_D + cov_W
            else:
                samples = mu + cov_D

            return samples

        return _sample_multiple(
            s, mu=self.mu, D=self.D, W=self.W, num_samples=num_samples
        )


def inv_softplus(y):
    if y < 20.0:
        # y = log(1 + exp(x))  ==>  x = log(exp(y) - 1)
        return np.log(np.exp(y) - 1)
    else:
        return y


class LowrankMultivariateGaussianOutput(DistributionOutput):
    @validated()
    def __init__(
        self,
        dim: int,
        rank: int,
        sigma_init: float = 1.0,
        sigma_minimum: float = sigma_minimum,
    ) -> None:
        super().__init__(self)
        self.distr_cls = LowrankMultivariateGaussian
        self.dim = dim
        self.rank = rank
        if rank == 0:
            self.args_dim = {"mu": dim, "D": dim}
        else:
            self.args_dim = {"mu": dim, "D": dim, "W": dim * rank}
        self.mu_bias = 0.0
        self.sigma_init = sigma_init
        self.sigma_minimum = sigma_minimum

    def get_args_proj(self, prefix: Optional[str] = None) -> ArgProj:
        return ArgProj(
            args_dim=self.args_dim,
            domain_map=mx.gluon.nn.HybridLambda(self.domain_map),
            prefix=prefix,
        )

    def distribution(
        self, distr_args, loc=None, scale=None, **kwargs
    ) -> LowrankMultivariateGaussian:

        distr = LowrankMultivariateGaussian(self.dim, self.rank, *distr_args)
        if loc is None and scale is None:
            return distr
        else: #call the transformed distribution defined below
            return TransformedLowrankMultivariateGaussian(
                distr, [bijection.AffineTransformation(loc=loc, scale=scale)]
            )

    def domain_map(self, F, mu_vector, D_vector, W_vector=None):
        r"""

        Parameters
        ----------
        F
        mu_vector
            Tensor of shape (..., dim)
        D_vector
            Tensor of shape (..., dim)
        W_vector
            Tensor of shape (..., dim * rank )

        Returns
        -------
        Tuple
            A tuple containing tensors mu, D, and W, with shapes
            (..., dim), (..., dim), and (..., dim, rank), respectively.

        """

        d_bias = (
            inv_softplus(self.sigma_init ** 2)
            if self.sigma_init > 0.0
            else 0.0
        )

        # sigma_minimum helps avoiding cholesky problems, we could also jitter
        # However, this seems to cause the maximum likelihood estimation to
        # take longer to converge. This needs to be re-evaluated.
        D_diag = (
            F.Activation(D_vector + d_bias, act_type="softrelu")
            + self.sigma_minimum ** 2
        )

        if self.rank == 0:
            return mu_vector + self.mu_bias, D_diag
        else:
            assert W_vector is not None, \
                "W_vector cannot be None if rank is not zero!"
            # reshape from vector form (..., d * rank) to matrix form (..., d, rank)
            W_matrix = W_vector.reshape((-2, self.dim, self.rank, -4), reverse=1)
            return mu_vector + self.mu_bias, D_diag, W_matrix

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)


# Need to inherit from LowrankMultivariateGaussian to get the overwritten loss method.
class TransformedLowrankMultivariateGaussian(TransformedDistribution, LowrankMultivariateGaussian):
    @validated()
    def __init__(
        self, base_distribution: LowrankMultivariateGaussian, transforms
    ) -> None:
        super().__init__(base_distribution, transforms)

    def crps(self, samples: Tensor, y: Tensor) -> Tensor:
        # TODO: use event_shape
        F = getF(y)
        scale = 1.0
        for t in self.transforms[::-1]:
            assert isinstance(
                t, bijection.AffineTransformation
            ), "Not an AffineTransformation"
            y = t.f_inv(y)
            samples = t.f_inv(samples)
            scale *= t.scale
        p = self.base_distribution.crps(samples, y) #shape (batch_size,seq_len,m)

        scaled_p = F.broadcast_mul(p, scale)
        return F.sum(scaled_p, axis=-1).expand_dims(-1)