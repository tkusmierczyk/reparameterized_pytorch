from typing import Callable
import torch
from torch.distributions import Normal


def create_gaussian_nll(event_shape: torch.Size, loc=0.0, scale=1.0) -> Callable:
    loc = torch.zeros(event_shape) + loc
    scale = torch.ones(event_shape) * scale
    p = Normal(loc, scale)

    def get_nll(parameter):
        log_prob = p.log_prob(parameter)
        for _ in range(len(event_shape)):
            log_prob = log_prob.sum(-1)
        return -log_prob

    return get_nll
