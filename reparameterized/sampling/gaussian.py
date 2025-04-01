"""Basic posterior models (sampling functions + variational parameters)."""

import math

import torch
from torch.nn.functional import softplus
from torch.distributions import Normal, MultivariateNormal, LowRankMultivariateNormal

from typing import Tuple, Callable, Dict

import logging


def create_factorized_gaussian_sampler(
    parameter: torch.Tensor,
    device=None,
    epsilon_scale=1e-8,
    loc_initalization=lambda parameter: parameter.clone().detach(),
    uscale_initialization=lambda parameter: torch.randn_like(parameter),
    **ignored_params,
) -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    """Creates a function that samples from Normal(loc, softplus(unnormalized_scale)).

    Returns:
        sampling function: which takes n_samples and outputs tuple(sample, NLL)
        dictionary {name: tensor} with variational parameters (loc and unnormalized_scale)
        dictionary with auxiliary objects
    """
    device = device or parameter.device

    loc = loc_initalization(parameter)
    unnormalized_scale = uscale_initialization(parameter)

    loc = loc.to(device).requires_grad_(True)
    unnormalized_scale = unnormalized_scale.to(device).requires_grad_(True)

    def sample_factorized_gaussian(n_samples=1):
        q = Normal(loc, softplus(unnormalized_scale) + epsilon_scale)
        sample = q.rsample(torch.Size([n_samples]))

        # calc total NLL for all params (out shape==n_samples)
        data_dims = list(range(1, len(sample.shape)))
        nll = -q.log_prob(sample).sum(dim=data_dims)
        nll = nll.to(sample.device)

        return sample, nll

    variational_params = {"loc": loc, "unnormalized_scale": unnormalized_scale}
    return sample_factorized_gaussian, variational_params, {}


def create_gaussian_tril_sampler(
    parameter: torch.Tensor,
    device=None,
    loc_initialization: Callable = lambda parameter: parameter.flatten()
    .clone()
    .detach(),
    unnormalized_diag_initialization: Callable = lambda parameter: torch.randn_like(
        parameter.flatten()
    ),
    cov_tril_initialization: Callable = lambda parameter: 0.01
    * torch.randn(
        torch.Size([parameter.flatten().shape[0], parameter.flatten().shape[0]])
    ),
    **ignored_params,
) -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    """Creates a function that samples from MultivariateNormal(loc, cov).

    The covariance matrix is composed as cov := diag(softplus(unnormalized_diag)) + tril(cov_tril)
    Returns:
        sampling function: which takes n_samples and outputs tuple(sample, NLL)
        dictionary {name: tensor} with variational parameters (loc, cov parameters)
        dictionary with auxiliary objects (currently empty)
    """
    device = device or parameter.device

    loc = loc_initialization(parameter)
    unnormalized_diag = unnormalized_diag_initialization(parameter)
    cov_tril = cov_tril_initialization(parameter)

    loc = loc.to(device).requires_grad_(True)
    unnormalized_diag = unnormalized_diag.to(device).requires_grad_(True)
    cov_tril = cov_tril.to(device).requires_grad_(True)

    def sample_gaussian_tril(n_samples=1):
        cov = torch.tril(cov_tril, diagonal=-1) + torch.diag(
            softplus(unnormalized_diag) + 1e-8
        )
        q = MultivariateNormal(loc=loc, scale_tril=cov)
        sample = q.rsample(torch.Size([n_samples]))

        # calc total NLL for all params (shape==n_samples)
        nll = -q.log_prob(sample)
        nll = nll.to(sample.device)

        sample = sample.reshape(torch.Size([n_samples]) + parameter.shape)
        return sample, nll

    variational_params = {
        "loc": loc,
        "unnormalized_diag": unnormalized_diag,
        "cov_tril": cov_tril,
    }
    return sample_gaussian_tril, variational_params, {}


def cov_L_random_initalization(parameter, var_init=1 / 5e-4, cov_init_noise_scale=1e-3):
    dim = parameter.numel()
    cov_L = torch.tril(cov_init_noise_scale * torch.randn([dim, dim])) + torch.eye(
        dim
    ) * math.sqrt(var_init)
    return cov_L


def create_full_rank_gaussian_sampler(
    parameter: torch.Tensor,
    device: str = None,
    loc_initalization: Callable = lambda parameter: parameter.clone().detach(),
    cov_L_initalization: Callable = cov_L_random_initalization,
    optimize_loc: bool = True,
    optimize_cov: bool = True,
    **ignored_params,
) -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    """Creates a function that samples from full-rank MultivariateNormal(loc, cov).

    Returns:
        sampling function: which takes n_samples and outputs tuple(sample, NLL)
        dictionary {name: tensor} with variational parameters (loc, cov parameters)
        dictionary with auxiliary objects (currently empty)
    """
    device = device or parameter.device

    # Prepare variational parameters
    loc = loc_initalization(parameter.flatten())
    cov_L = cov_L_initalization(parameter)

    loc = loc.to(device).requires_grad_(optimize_cov)
    cov_L = cov_L.to(device).requires_grad_(optimize_cov)

    def sample_gaussian(n_samples=1):
        cov = cov_L @ cov_L.t()
        cov = 0.5 * (cov.T + cov)
        q = MultivariateNormal(loc=loc, covariance_matrix=cov)
        sample = q.rsample(torch.Size([n_samples]))

        # calc total NLL for all params (shape==n_samples)
        nll = -q.log_prob(sample)
        nll = nll.to(sample.device)

        sample = sample.reshape(torch.Size([n_samples]) + parameter.shape)
        return sample, nll

    variational_params = {
        "loc": loc,
        "cov_L": cov_L,
    }
    variational_params = {
        n: p for n, p in variational_params.items() if p.requires_grad
    }  # exclude no-grad params
    return sample_gaussian, variational_params, {}


def create_gaussian_lowrank_sampler(
    parameter: torch.Tensor,
    device=None,
    loc_initialization: Callable = lambda parameter: parameter.flatten()
    .clone()
    .detach(),
    cov_diag_initialization: Callable = lambda parameter: torch.ones_like(
        parameter.flatten()
    )
    * 0.01,
    cov_factor_initialization: Callable = lambda parameter: 0.01
    * torch.randn(
        torch.Size(
            [parameter.flatten().shape[0], min(10, parameter.flatten().shape[0])]
        )
    ),
    rank: int = 5,
    **ignored_params,
) -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    """Creates a function that samples from LowRankMultivariateNormal(loc, cov).

    The covariance matrix is represented in low-rank form as:
    cov = cov_factor @ cov_factor.T + cov_diag

    This is more efficient than full-rank for high-dimensional parameters.

    Args:
        parameter: The parameter tensor to be sampled
        device: Device to place tensors on
        loc_initialization: Function to initialize the mean vector
        cov_diag_initialization: Function to initialize the diagonal covariance
        cov_factor_initialization: Function to initialize the low-rank factor
        rank: The rank of the low-rank component (default: 5)

    Returns:
        sampling function: which takes n_samples and outputs tuple(sample, NLL)
        dictionary {name: tensor} with variational parameters (loc, cov parameters)
        dictionary with auxiliary objects (currently empty)
    """
    device = device or parameter.device
    param_size = parameter.flatten().shape[0]

    # Ensure rank is not larger than parameter size
    actual_rank = min(rank, param_size)
    if actual_rank != rank:
        logging.warning(
            f"[create_gaussian_lowrank_sampler] actual_rank={actual_rank} != requested_rank={rank}!"
        )

    loc = loc_initialization(parameter)
    cov_diag = cov_diag_initialization(parameter)
    cov_factor = cov_factor_initialization(parameter)

    # Ensure cov_factor has correct shape
    if cov_factor.shape[1] != actual_rank:
        cov_factor = 0.01 * torch.randn(
            torch.Size([param_size, actual_rank]), device=device
        )

    loc = loc.to(device).requires_grad_(True)
    cov_diag = cov_diag.to(device).requires_grad_(True)
    cov_factor = cov_factor.to(device).requires_grad_(True)

    def sample_gaussian_lowrank(n_samples=1):
        # Ensure diagonal covariance is positive
        positive_diag = softplus(cov_diag) + 1e-8

        # Create distribution
        q = LowRankMultivariateNormal(
            loc=loc, cov_factor=cov_factor, cov_diag=positive_diag
        )
        sample = q.rsample(torch.Size([n_samples]))

        # Calculate total NLL for all params (shape==n_samples)
        nll = -q.log_prob(sample)
        nll = nll.to(sample.device)

        # Reshape sample to match original parameter shape
        sample = sample.reshape(torch.Size([n_samples]) + parameter.shape)
        return sample, nll

    variational_params = {
        "loc": loc,
        "cov_diag": cov_diag,
        "cov_factor": cov_factor,
    }

    return sample_gaussian_lowrank, variational_params, {}
