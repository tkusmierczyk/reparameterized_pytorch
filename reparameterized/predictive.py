"""Sampling from predictive distribution."""
import torch

from .parameters import StateDict, take_parameters_sample, load_state_dict

from typing import Callable


def sample_predictive(
    input_x: torch.Tensor,
    model: torch.nn.Module,
    parameters_samples: StateDict,
    sampler: Callable = lambda logits, _: logits[None, ...],
    n_samples: int = 1,
    flatten_samples_dims: bool = True,
    parameters_strict_shapes: bool = True,
    **sampler_kwargs,
) -> torch.Tensor:
    """Samples from model predictive distribution.

    First, takes latent parameters' samples (=parameters_samples) and loades them in model.
    Then, for the new values of parameters (e.g. weights and biases) pushes inputs (input_x).
    Finally, the obtained model output logits are pushed through sampler to get predictive samples.

    Args:
        input_x (torch.Tensor), model (torch.nn.Module): logits = model.forward(input_x)
        parameters_samples (StateDict): _description_
        sampler (Callable, optional): Takes logits and outputs n_samples samples from predictive distribution. 
            Default sampler = identity i.e. just returns logits with additional dimension but without any sampling.
        n_samples (int, optional): An argument passed to sampler. Defaults to 1.
        flatten_samples_dims (bool, optional): Whether to flatten output
            (i.e. remove additional dimension due to sampling with sampler). Defaults to True.
        parameters_strict_shapes (bool, optional): If shapes in parameters_samples must match shapes in model. 
            Defaults to True.

    Returns:
        torch.Tensor: n_samples samples from sampler obtained for each parameter from parameters_samples
    """
    predictive_samples = []
    prev_state_dict = {}
    for s, state_dict in enumerate(take_parameters_sample(parameters_samples)):

        model = load_state_dict(
            model,
            state_dict,
            prev_state_dict=prev_state_dict if s == 0 else {},  # store original values
            strict_shapes=parameters_strict_shapes,
        )
        logits = model.forward(input_x)

        predictive_samples1 = sampler(logits, n_samples=n_samples, **sampler_kwargs)
        assert predictive_samples1.shape == torch.Size([n_samples] + list(logits.shape))

        predictive_samples.append(predictive_samples1)
    model = load_state_dict(model, prev_state_dict)  # restore model to the original state
    # stack and move samples of y to dim=0 and samples of theta to dim=1:
    predictive_samples = torch.stack(predictive_samples).transpose(0, 1)

    if flatten_samples_dims:
        predictive_samples = predictive_samples.flatten(start_dim=0, end_dim=1)
    return predictive_samples


def predictive_likelihoods(
    input_x: torch.Tensor,
    output_y: torch.Tensor,
    model: torch.nn.Module,
    parameters_samples: StateDict,
    likelihood_func: Callable,
    **likelihood_func_kwargs,
) -> torch.Tensor:
    batch_dim = torch.Size([len(output_y)])
    logliks = []
    for state_dict in take_parameters_sample(parameters_samples):

        model = load_state_dict(model, state_dict)
        logits = model.forward(input_x)

        logliks1 = likelihood_func(logits, output_y, **likelihood_func_kwargs)
        assert logliks1.shape == batch_dim

        logliks.append(logliks1)
    logliks = torch.stack(logliks)
    return logliks
