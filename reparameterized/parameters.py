"""Code for sampling parameters and loading them for native pytorch models."""
import torch

import warnings
from typing import Iterable, Union, Tuple, Callable, Dict
import logging

StateDict = Dict[str, torch.Tensor]

ParamsKey = Union[str, Iterable[str]]
NLLs = Dict[ParamsKey, torch.Tensor]
Samplers = Union[Dict[ParamsKey, Callable], Iterable[Tuple[ParamsKey, Callable]]]
DensityEsimators = Samplers  # an alias to avoid repeating the same structure


def _are_shapes_compatible(shape1, shape2):
    """
    Check if tensor1 and tensor2 agree on all trailing dimensions, but can have additional leading dimensions.

    Returns:
        bool: True if the tensors have matching dimensions.
    """

    # Reverse the shapes to compare from the last dimension backward
    reversed_shape1 = reversed(shape1)
    reversed_shape2 = reversed(shape2)

    # Compare the shapes from the last dimension backward
    for dim1, dim2 in zip(reversed_shape1, reversed_shape2):
        if dim1 != dim2:
            return False  # Not compatible if the dimensions differ

    # If no conflicts were found, they are compatible
    return True


def _smart_reshape(tensor, desired_shape):
    """
    Reshape a tensor to match the desired shape. If the number of elements in the tensor is greater than
    the number of elements in the desired shape, add an extra leading dimension.

    Args:
        tensor (Tensor): The input tensor to reshape.
        desired_shape (tuple): The desired shape (can have fewer elements than the input tensor).

    Returns:
        Tensor: The reshaped tensor, possibly with an extra leading dimension if necessary.
    """
    # Calculate the number of elements in the tensor and the desired shape
    tensor_num_elements = tensor.numel()
    desired_num_elements = torch.prod(torch.tensor(desired_shape)).item()

    if _are_shapes_compatible(tensor.shape, desired_shape):
        # If shapes already match, do nothing
        return tensor

    elif tensor_num_elements == desired_num_elements:
        # If the number of elements match, reshape directly
        return tensor.reshape(desired_shape)

    elif tensor_num_elements > desired_num_elements:
        # If the tensor has more elements, prepend a leading dimension
        extra_dim = tensor_num_elements // desired_num_elements
        new_shape = (extra_dim,) + desired_shape
        return tensor.reshape(new_shape)

    else:
        raise ValueError(
            "The tensor has fewer elements than the desired shape. Cannot reshape."
        )


def load_state_dict(
    module,
    state_dict: StateDict,
    path="",
    prev_state_dict: StateDict = {},
    strict_shapes: bool = True,
):
    """Sets model params to samples from e.g. approximate posterior.

    Args:
        module: torch module instance
        state_dict: dictionary {parameter_name/path: sample_value (tensor)}
        prev_state_dict: output dictionary containing previous values of the updated parameters
        strict_shapes: allow reshaping new values to match the originals (number of elements must match)
    """
    for name, m in module._modules.items():
        load_state_dict(
            m,
            state_dict,
            path=f"{path}.{name}",
            prev_state_dict=prev_state_dict,
            strict_shapes=strict_shapes,
        )

    for name in module._parameters.keys():
        sample_path = f"{path}.{name}"[1:]  # skip the leading dot

        if sample_path not in state_dict:
            logging.debug(f"[load_state_dict] No update for {sample_path}.")
            continue
        new_value = state_dict[sample_path]

        shape, new_shape = module._parameters[name].shape, new_value.shape
        assert (strict_shapes and new_shape == shape) or (
            not strict_shapes
            and (
                new_shape.numel() == shape.numel()
                or _are_shapes_compatible(new_shape, shape)
            )
        ), (f"sample_path={sample_path} shape={new_shape} " f"current shape={shape}")

        prev_state_dict[sample_path] = module._parameters[name]  # save the old values
        module._parameters[name] = (
            new_value if new_shape == shape else _smart_reshape(new_value, shape)
        )

    return module


def sample_parameters(
    parameters2sampler: Samplers, n_samples: int = 1
) -> Tuple[StateDict, NLLs]:
    """Samples model parameters using predefined samplers.

    Args:
        parameters2sampler: pairs: parameter_name(s) => sampling function

    Returns:
        Two dictionaries: {parameter_name: samples} and {parameter_name(s): NLLs}
    """
    if hasattr(parameters2sampler, "items"):
        parameters2sampler = parameters2sampler.items()

    samples, nlls = {}, {}
    for parameters, sampler in parameters2sampler:
        parameters_samples, parameters_nlls = sampler(n_samples)

        nlls[parameters] = parameters_nlls

        if isinstance(parameters, str):
            samples[parameters] = parameters_samples

        elif isinstance(parameters_samples, dict):
            samples.update(parameters_samples)

        elif isinstance(parameters_samples, Iterable):
            parameters_samples = list(parameters_samples)
            assert len(parameters) == len(parameters_samples)
            for name, value in zip(parameters, parameters_samples):
                samples[name] = value

        else:
            raise Exception(
                f"I don't how to handle samples for parameters={parameters}!"
            )

    return samples, nlls


def take_parameters_sample(parameters_samples: StateDict):
    """Yields state dictionaries {parameter_name: next(parameters_sample)}.

    Args:
        parameters_samples: dictionary {parameter_name: tensor with samples in axis=0}
    """

    def _samples_len(samples):
        n_samples = min(len(v) for _, v in samples.items())
        if not all((len(v) == n_samples) for _, v in samples.items()):
            warnings.warn(
                "Not all samples have the same length. "
                f"Setting n_samples to the minimum = {n_samples}."
            )
        return n_samples

    for sample_no in range(_samples_len(parameters_samples)):
        yield {p: s[sample_no] for p, s in parameters_samples.items()}


def is_parameter_handled(parameters2sampler: Samplers, parameter_name: str) -> bool:
    if hasattr(parameters2sampler, "items"):
        parameters2sampler = parameters2sampler.items()

    for parameters, _ in parameters2sampler:
        if parameter_name in parameters or parameter_name == parameters:
            return True

    return False


def estimate_parameters_nll(
    parameters2nllfunc: DensityEsimators,
    state_dict: StateDict,
    reduce_over_params: bool = False,
) -> NLLs:
    """Returns dictionary {parameter name: NLLs for samples from state_dict}.

    If reduce_over_params is True,
    returns NLLs for samples from state_dict
    but totaled over all parameters.
    """
    if hasattr(parameters2nllfunc, "items"):
        parameters2nllfunc = parameters2nllfunc.items()

    parameters2nll = {}
    for parameters, nll_estimator in parameters2nllfunc:

        if isinstance(parameters, str):
            parameters_value = state_dict[parameters]

        elif isinstance(parameters, Iterable):
            # extract multiple parameters and pass them as a list
            parameters_value = [state_dict[parameter] for parameter in parameters]

        else:
            raise Exception(f"I don't how to handle parameters={parameters}!")

        parameters2nll[parameters] = nll_estimator(parameters_value)

    if reduce_over_params:
        # total over all parameters => (n_samples, )
        nlls = list(parameters2nll.values())
        return torch.stack(nlls).sum(0)
    else:
        # calc priors for separate params => {param_name: (n_samples, )}
        return parameters2nll
