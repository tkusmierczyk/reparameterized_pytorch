""" This module provides functions to create samplers for various posterior model architectures. 
    The samplers can be used for joint sampling of multiple parameters or independent sampling 
    for each parameter in a dictionary. The module supports different architectures such as 
    RealNVP-based flows, factorized Gaussian, Gaussian with Cholesky decomposition, and full-rank Gaussian.

    Functions:
        create_joint_sampler(parameters: Dict[str, torch.Tensor], architecture: str):
            Creates a joint sampler for multiple parameters based on the specified architecture.
            Supports architectures like RealNVP, factorized Gaussian, Gaussian with Cholesky decomposition, 
            and full-rank Gaussian.

        create_parameter_sampler(parameter: torch.Tensor, architecture: str):
            Creates a sampler for a single parameter based on the specified architecture.
            Supports architectures like RealNVP, factorized Gaussian, Gaussian with Cholesky decomposition, 
            and full-rank Gaussian.

        create_independent_samplers(parameters: Dict[str, torch.Tensor], architecture: str):
            Creates independent samplers for each parameter in the provided dictionary. Each parameter 
            is sampled independently based on the specified architecture.

        ValueError: If the specified architecture is not supported.

    Currently supported architecture names:
        "rnvp_rezero",
        "rnvp",
        "rnvp_rezero_small",
        "rnvp_small",
        "factorized_gaussian",
        "factorized_gaussian_rezero",
        "gaussian_tril",
        "gaussian_tril_rezero",
        "gaussian_full",
        "gaussian_full_rezero",
"""

import torch
from typing import Dict


from .__init__ import *
from .realnvp import build_realnvp
from ..parameters import parameter_samplers_to_joint_sampler

import logging


def create_joint_sampler(
    parameters: Dict[str, torch.Tensor], architecture: str, **kwargs
):
    """Create a joint sampler for multiple parameters."""

    if "svd" in architecture and "rnvp" in architecture:
        logging.info(
            "[create_joint_sampler] create_multiparameter_svd_sampler + create_flow_sampler"
        )
        sampler, variational_params, aux_objs = create_multiparameter_svd_sampler(
            create_flow_sampler,
            parameters,
            svd_residuals=kwargs.pop("svd_residuals", ("residuals" in architecture)),
            build_flow_func=build_realnvp,
            realnvp_rezero_trick=kwargs.pop(
                "realnvp_rezero_trick", ("rezero" in architecture)
            ),
            realnvp_num_layers=kwargs.pop(
                "realnvp_num_layers", (8 if "small" in architecture else 32)
            ),
            realnvp_m=kwargs.pop(
                "realnvp_m", (128 if "small" in architecture else 6 * 128)
            ),
            **kwargs,
        )

    elif "svd" in architecture and "factorized_gaussian" in architecture:
        logging.info(
            "[create_joint_sampler] create_multiparameter_svd_sampler + create_factorized_gaussian_sampler"
        )
        sampler, variational_params, aux_objs = create_multiparameter_svd_sampler(
            create_factorized_gaussian_sampler,
            parameters,
            svd_residuals=kwargs.pop("svd_residuals", ("residuals" in architecture)),
            **kwargs,
        )

    elif "svd" in architecture and "gaussian_lowrank" in architecture:
        logging.info(
            "[create_joint_sampler] create_multiparameter_svd_sampler + create_gaussian_lowrank_sampler"
        )
        sampler, variational_params, aux_objs = create_multiparameter_svd_sampler(
            create_gaussian_lowrank_sampler,
            parameters,
            svd_residuals=kwargs.pop("svd_residuals", ("residuals" in architecture)),
            **kwargs,
        )

    elif "rnvp" in architecture:
        logging.info(
            "[create_joint_sampler] create_multiparameter_sampler_dict + create_flow_sampler"
        )
        sampler, variational_params, aux_objs = create_multiparameter_sampler_dict(
            create_flow_sampler,
            parameters,
            build_flow_func=build_realnvp,
            realnvp_rezero_trick=kwargs.pop(
                "realnvp_rezero_trick", ("rezero" in architecture)
            ),
            realnvp_num_layers=kwargs.pop(
                "realnvp_num_layers", (8 if "small" in architecture else 32)
            ),
            realnvp_m=kwargs.pop(
                "realnvp_m", (128 if "small" in architecture else 6 * 128)
            ),
            **kwargs,
        )

    elif architecture == "factorized_gaussian":
        logging.info(
            "[create_joint_sampler] create_multiparameter_sampler_dict + create_factorized_gaussian_sampler"
        )
        sampler, variational_params, aux_objs = create_multiparameter_sampler_dict(
            create_factorized_gaussian_sampler, parameters, **kwargs
        )

    elif architecture == "factorized_gaussian_rezero":
        logging.info(
            "[create_joint_sampler] create_multiparameter_sampler_dict + create_factorized_gaussian_sampler"
        )
        sampler, variational_params, aux_objs = create_multiparameter_sampler_dict(
            create_factorized_gaussian_sampler,
            parameters,
            loc_initalization=lambda parameter_init_value: torch.zeros_like(
                parameter_init_value
            ),
            uscale_initialization=lambda parameter_init_value: torch.ones_like(
                parameter_init_value
            )
            * -3.0,
            **kwargs,
        )

    elif architecture == "gaussian_tril":
        logging.info(
            "[create_joint_sampler] create_multiparameter_sampler_dict + create_gaussian_tril_sampler"
        )
        sampler, variational_params, aux_objs = create_multiparameter_sampler_dict(
            create_gaussian_tril_sampler, parameters, **kwargs
        )

    elif architecture == "gaussian_tril_rezero":
        logging.info(
            "[create_joint_sampler] create_multiparameter_sampler_dict + create_gaussian_tril_sampler + rezero"
        )
        sampler, variational_params, aux_objs = create_multiparameter_sampler_dict(
            create_gaussian_tril_sampler,
            parameters,
            loc_initialization=lambda p: (0.0 * p.flatten().clone().detach()),
        )

    elif architecture == "gaussian_full":
        logging.info(
            "[create_joint_sampler] create_multiparameter_sampler_dict + create_full_rank_gaussian_sampler"
        )
        sampler, variational_params, aux_objs = create_multiparameter_sampler_dict(
            create_full_rank_gaussian_sampler, parameters, **kwargs
        )

    elif architecture == "gaussian_full_rezero":
        logging.info(
            "[create_joint_sampler] create_multiparameter_sampler_dict + create_full_rank_gaussian_sampler + rezero"
        )
        sampler, variational_params, aux_objs = create_multiparameter_sampler_dict(
            create_full_rank_gaussian_sampler,
            parameters,
            loc_initialization=lambda p: (0.0 * p.flatten().clone().detach()),
            **kwargs,
        )

    elif architecture == "gaussian_lowrank":
        logging.info(
            "[create_joint_sampler] create_multiparameter_sampler_dict + create_gaussian_lowrank_sampler"
        )
        sampler, variational_params, aux_objs = create_multiparameter_sampler_dict(
            create_gaussian_lowrank_sampler, parameters, **kwargs
        )

    elif architecture == "gaussian_lowrank_rezero":
        logging.info(
            "[create_joint_sampler] create_multiparameter_sampler_dict + create_gaussian_lowrank_sampler + rezero"
        )
        sampler, variational_params, aux_objs = create_multiparameter_sampler_dict(
            create_gaussian_lowrank_sampler,
            parameters,
            loc_initialization=lambda p: (0.0 * p.flatten().clone().detach()),
            **kwargs,
        )

    else:
        raise ValueError(
            f"Posterior model architecture = {architecture} not supported!"
        )

    return sampler, variational_params, aux_objs


def create_parameter_sampler(parameter, architecture, **create_func_kwargs):

    if "svd" in architecture and "rnvp" in architecture:
        sampler, variational_params, aux_objs = create_svd_sampler(
            parameter,
            create_flow_sampler,
            svd_residuals=("residuals" in architecture),
            build_flow_func=build_realnvp,
            realnvp_rezero_trick=("rezero" in architecture),
            realnvp_num_layers=(8 if "small" in architecture else 32),
            realnvp_m=(128 if "small" in architecture else 6 * 128),
            **create_func_kwargs,
        )

    elif "rnvp" in architecture:
        sampler, variational_params, aux_objs = create_flow_sampler(
            parameter,
            build_flow_func=build_realnvp,
            realnvp_rezero_trick=("rezero" in architecture),
            realnvp_num_layers=(8 if "small" in architecture else 32),
            realnvp_m=(128 if "small" in architecture else 6 * 128),
            **create_func_kwargs,
        )

    elif "svd" in architecture and "factorized_gaussian" in architecture:
        sampler, variational_params, aux_objs = create_svd_sampler(
            parameter,
            create_factorized_gaussian_sampler,
            svd_residuals=("residuals" in architecture),
            loc_initalization=lambda parameter_init_value: torch.zeros_like(
                parameter_init_value
            ),
            uscale_initialization=lambda parameter_init_value: torch.ones_like(
                parameter_init_value
            )
            * -3.0,
            **create_func_kwargs,
        )

    elif (
        "svd" in architecture
        and "gaussian" in architecture
        and "lowrank" in architecture
    ):
        sampler, variational_params, aux_objs = create_svd_sampler(
            parameter,
            create_gaussian_lowrank_sampler,
            svd_residuals=("residuals" in architecture),
            **create_func_kwargs,
        )

    elif architecture == "factorized_gaussian_rezero":
        sampler, variational_params, aux_objs = create_factorized_gaussian_sampler(
            parameter,
            loc_initalization=lambda parameter_init_value: torch.zeros_like(
                parameter_init_value
            ),
            uscale_initialization=lambda parameter_init_value: torch.ones_like(
                parameter_init_value
            )
            * -3.0,
            **create_func_kwargs,
        )

    elif architecture == "factorized_gaussian":
        sampler, variational_params, aux_objs = create_factorized_gaussian_sampler(
            parameter, **create_func_kwargs
        )

    elif architecture == "gaussian_tril":
        sampler, variational_params, aux_objs = create_gaussian_tril_sampler(
            parameter, **create_func_kwargs
        )

    elif architecture == "gaussian_tril_rezero":
        sampler, variational_params, aux_objs = create_gaussian_tril_sampler(
            parameter,
            loc_initialization=lambda p: (0.0 * p.flatten().clone().detach()),
            **create_func_kwargs,
        )

    elif architecture == "gaussian_full":
        sampler, variational_params, aux_objs = create_full_rank_gaussian_sampler(
            parameter,
            **create_func_kwargs,
        )

    elif architecture == "gaussian_full_rezero":
        sampler, variational_params, aux_objs = create_full_rank_gaussian_sampler(
            parameter,
            loc_initialization=lambda p: (0.0 * p.flatten().clone().detach()),
            **create_func_kwargs,
        )

    elif architecture == "gaussian_lowrank":
        sampler, variational_params, aux_objs = create_gaussian_lowrank_sampler(
            parameter, **create_func_kwargs
        )

    elif architecture == "gaussian_lowrank_rezero":
        sampler, variational_params, aux_objs = create_gaussian_lowrank_sampler(
            parameter,
            loc_initialization=lambda p: (0.0 * p.flatten().clone().detach()),
            **create_func_kwargs,
        )

    else:
        raise ValueError(
            f"Posterior model architecture = {architecture} not supported!"
        )

    return sampler, variational_params, aux_objs


def create_independent_samplers(
    parameters: Dict[str, torch.Tensor], architecture: str, **create_func_kwargs
):
    """Create (independent) samplers for each parameter in the dictionary {parameter name: parameter tensor}."""

    parameter2sampler, variational_params, aux_objs = {}, {}, {}
    for name, parameter in parameters.items():
        sampler1, variational_params1, aux_objs1 = create_parameter_sampler(
            parameter,
            architecture,
            **create_func_kwargs,
        )
        aux_objs1 = {name + "." + k: v for k, v in aux_objs1.items()}

        parameter2sampler[name] = sampler1
        for k, v in variational_params1.items():
            variational_params[name + "." + k] = v
        aux_objs.update(aux_objs1)

    return parameter2sampler, variational_params, aux_objs


def create_sampler(
    parameters: Dict[str, torch.Tensor], architecture: str, **create_func_kwargs
):
    """Creates a sampler for the provided parameters based on architecture type.

    This function either creates a joint sampler for all parameters or individual
    samplers for each parameter, depending on the specified architecture.

    Args:
        parameters (Dict[str, torch.Tensor]): A dictionary where keys are parameter
            names and values are their corresponding parameter tensors.
        architecture (str): A string specifying the architecture type. If the string
            contains "joint", a joint sampler is created; otherwise, independent
            samplers are created.
        **create_func_kwargs: Additional arguments passed to the sampler
            creation functions.

    Returns a tuple containing:
            - sampler: The created sampler, either joint or derived from individual samplers.
            - variational_params: Variational parameters associated with the sampler.
            - aux_objs: Auxiliary objects created during the sampler setup.
    """
    if "joint" in architecture:
        logging.info(
            "[create_sampler] All parameters are put together and use a joint sampler"
        )
        sampler, variational_params, aux_objs = create_joint_sampler(
            parameters, architecture
        )
    else:
        logging.info("[create_sampler] Each parameter gets its own sampler")
        parameter2sampler, variational_params, aux_objs = create_independent_samplers(
            parameters, architecture
        )
        sampler = parameter_samplers_to_joint_sampler(parameter2sampler)

    return sampler, variational_params, aux_objs
