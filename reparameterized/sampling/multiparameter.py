"""Joint sampling for multiple parameters."""

import torch

from typing import Callable, Iterable, Tuple, Dict
import logging


def extract_parameters_shapes(named_parameters):
    return [p.shape for _, p in named_parameters]


def get_parameters_total_n_elements(named_parameters):
    return sum(p.numel() for _, p in named_parameters)


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
    **sampler_create_func_args,
):
    """Builds one joint sampler for multiple parameters.

    Builds a sampler producing unnamed samples ordered according to named_parameters.
    NLLs are calculated jointly for all the parameters.

    Sample use:
        sampler, variational_params, aux_objs = multiparameter.create_multiparameter_sampler_dict(
            create_flow_sampler,  # creates a flow for all parameters considered jointly
            model.named_parameters(),  # list of model parameters
            build_flow_func = neural_spline_flow.build_spline_flow,  # flow type to be built
            spline_flow_hidden_units=16,  # some additional parameters to be passed to the building function
        )
    """
    named_parameters = list(named_parameters)

    # flatten to a single vector:
    parameters_shapes = extract_parameters_shapes(named_parameters)
    parameters_jointly = torch.concat([p.flatten() for _, p in named_parameters])
    logging.debug(
        f"[create_multiparameter_sampler] parameters_shapes={parameters_shapes} "
        f"parameters_jointly={parameters_jointly.shape}"
    )
    sampler, variational_params, aux_objs = sampler_create_func(
        parameters_jointly,
        parameters_shapes=parameters_shapes,  # additional information which may be used by samplers
        **sampler_create_func_args,
    )

    def _sampler_list_wrapper(n_samples=1):
        joint_samples, joint_nlls = sampler(n_samples)
        assert (
            joint_samples.shape[0] == n_samples
        ), f"sampler={sampler} returned wrong no of samples"
        samples = separate_parameters(joint_samples, named_parameters)
        return samples, joint_nlls

    return _sampler_list_wrapper, variational_params, aux_objs


def create_multiparameter_sampler_dict(
    sampler_create_func: Callable,
    named_parameters: Dict[str, torch.Tensor],
    **sampler_create_func_args,
):
    """Builds one joint sampler for multiple parameters (wrapper for create_multiparameter_sampler).

    Builds a sampler producing dictionary with named samples: {name [str]: sample [tensor]}.
    NLLs are calculated jointly for all the parameters.

    Sample use:
        sampler, variational_params, aux_objs = multiparameter.create_multiparameter_sampler_dict(
            create_flow_sampler,  # creates a flow for all parameters considered jointly
            dict(model.named_parameters()),  # model parameters
            build_flow_func = neural_spline_flow.build_spline_flow,  # flow type to be built
            spline_flow_hidden_units=16,  # some additional parameters to be passed to the building function
        )

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
