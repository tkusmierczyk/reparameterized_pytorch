import torch
from ..parameters import take_parameters_sample, load_state_dict


def categorical_log_prob(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = y.flatten()
    assert len(logits) == len(y)
    return torch.distributions.Categorical(logits=logits).log_prob(y)


def categorical_posterior_probs(model, *inputs, samples):
    all_probs = []
    for s in take_parameters_sample(samples):
        load_state_dict(model, s)
        logits = model(*inputs)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        all_probs.append(probs)
    all_probs = torch.stack(all_probs, dim=0)
    return all_probs
