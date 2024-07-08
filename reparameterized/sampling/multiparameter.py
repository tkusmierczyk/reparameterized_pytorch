"""Joint sampling for multiple parameters."""

import torch

from typing import Callable, Iterable, Tuple, Dict


def merge_parameters(parameter_samples: Iterable[torch.Tensor]):
    """Flattens and stacks parameter (separated) samples.

    Returns:
        tensor of shape (n samples, total dimension of all parameters)
    """
    return torch.hstack([sample.flatten(start_dim=1) for sample in parameter_samples])


def separate_parameters(
    joint_samples, named_parameters: Iterable[Tuple[str, torch.Tensor]]
):
    """Splits flattened output from joint sampler into individual parameters.

    Returns a list of samples ordered and reshaped according to named_parameters.
    """
    named_parameters = list(named_parameters)
    parameter2shape = [(n, p.shape) for n, p in named_parameters]

    samples = []
    start_ix = 0
    n_samples = joint_samples.shape[0]
    for parameter_name, shape in parameter2shape:
        npositions = shape.numel()
        parameter_samples = joint_samples[:, start_ix : (start_ix + npositions), ...]
        parameter_samples = parameter_samples.reshape(torch.Size([n_samples]) + shape)

        # samples[parameter_name] = parameter_samples   # return a dict
        samples.append(parameter_samples)  # return an ordered list

        start_ix += npositions
    return samples


def create_multiparameter_sampler(
    sampler_create_func: Callable,
    named_parameters: Iterable[Tuple[str, torch.Tensor]],
    **sampler_create_func_args
):
    """Builds one joint sampler for multiple parameters.

    Builds a sampler producing unnamed samples ordered according to named_parameters.
    NLLs are calculated jointly for all the parameters.
    """
    named_parameters = list(named_parameters)

    # flatten to a single vector:
    parameters_shapes = [p.shape for _, p in named_parameters]
    parameters_jointly = torch.concat([p.flatten() for _, p in named_parameters])
    sampler, variational_params, aux_objs = sampler_create_func(
        parameters_jointly,
        parameters_shapes=parameters_shapes,  # additional information which may be used by samplers
        **sampler_create_func_args
    )

    def _sampler_list_wrapper(n_samples=1):
        joint_samples, joint_nlls = sampler(n_samples)
        samples = separate_parameters(joint_samples, named_parameters)
        return samples, joint_nlls

    return _sampler_list_wrapper, variational_params, aux_objs


def create_multiparameter_sampler_dict(
    sampler_create_func: Callable,
    named_parameters: Dict[str, torch.Tensor],
    **sampler_create_func_args
):
    """Builds one joint sampler for multiple parameters.

    Builds a sampler producing dictionary with named samples: {name [str]: sample [tensor]}.
    NLLs are calculated jointly for all the parameters.
    """
    named_parameters = list(named_parameters.items())
    parameter_names = [n for n, _ in named_parameters]  # extract parameter names
    sampler, variational_params, aux_objs = create_multiparameter_sampler(
        sampler_create_func, named_parameters, **sampler_create_func_args
    )

    def _sampler_dict_wrapper(*args, **kwargs):
        """Matches unnamed samples from sampler with their names."""
        parameter_samples, joint_nlls = sampler(*args, **kwargs)
        return dict(zip(parameter_names, parameter_samples)), joint_nlls

    return _sampler_dict_wrapper, variational_params, aux_objs
