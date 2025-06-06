import torch
from torch import nn
from torch.nn import Sequential, Linear, LeakyReLU, Tanh
from torch.distributions import MultivariateNormal

import logging


class RealNVP(nn.Module):
    """Based on implementation by Jakub Tomczak
    https://jmtomczak.github.io/blog/3/3_flows.html
    """

    def __init__(self, net_s, net_t, num_layers, dim, rezero_trick=False):
        super().__init__()
        self.register_buffer("prior_loc", torch.zeros(dim))
        self.register_buffer("prior_cov", torch.eye(dim))

        self.t = nn.ModuleList([net_t() for _ in range(num_layers)])
        self.s = nn.ModuleList([net_s() for _ in range(num_layers)])
        self.num_flows = num_layers
        self.alpha = (
            nn.Parameter(torch.zeros(len(self.s)))
            if rezero_trick
            else torch.ones((len(self.s)))
        )
        self.beta = (
            nn.Parameter(torch.zeros(len(self.t)))
            if rezero_trick
            else torch.ones((len(self.t)))
        )

    @property
    def prior(self):
        return MultivariateNormal(self.prior_loc, self.prior_cov)

    def coupling(self, x, index, forward=True):
        (xa, xb) = torch.chunk(x, 2, 1)
        s = self.alpha[index] * self.s[index](xa)
        t = self.beta[index] * self.t[index](xa)
        if forward:
            yb = (xb - t) * torch.exp(-s)
        else:
            yb = s.exp() * xb + t
        return torch.cat((xa, yb), 1), s, t

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in range(self.num_flows):
            z, s, _ = self.coupling(z, i, forward=True)
            z = z.flip(1)
            log_det_J = log_det_J - s.sum(dim=1)
        return z, log_det_J

    def f_inv(self, z, device=None):
        x = z
        log_det_J = x.new_zeros(x.shape[0])
        for i in reversed(range(self.num_flows)):
            x = x.flip(1)
            x, s, _ = self.coupling(x, i, forward=False)
            log_det_J = log_det_J + s.sum(dim=1)
        return x, log_det_J

    def forward(self, x):
        z, log_det_J = self.f(x)
        return self.prior.log_prob(z) + log_det_J

    def sample(self, batchSize, D, calculate_nll=False):
        z = self.prior.sample((batchSize,))
        x, log_det_J = self.f_inv(z)
        if calculate_nll:
            log_prob_z = self.prior.log_prob(z)
            nll = -(log_prob_z - log_det_J)
            return x.view(-1, D), nll
        return x.view(-1, D)


def build_realnvp(
    output_dim,
    realnvp_m=128,
    realnvp_num_layers=4,
    realnvp_rezero_trick=False,
    realnvp_activation=LeakyReLU(),
    **ignored_kwargs,
):
    logging.info(
        f"[build_realnvp] output_dim={output_dim} realnvp_m={realnvp_m} "
        f"realnvp_num_layers={realnvp_num_layers} realnvp_rezero_trick={realnvp_rezero_trick} "
        f"realnvp_activation={realnvp_activation}"
    )

    d = output_dim
    m = realnvp_m

    net_s = lambda: Sequential(
        Linear(d - d // 2, m),
        realnvp_activation,
        Linear(m, m),
        realnvp_activation,
        Linear(m, d // 2),
        Tanh(),
    )
    net_t = lambda: Sequential(
        Linear(d - d // 2, m),
        realnvp_activation,
        Linear(m, m),
        realnvp_activation,
        Linear(m, d // 2),
    )
    realnvp = RealNVP(
        net_s, net_t, realnvp_num_layers, d, rezero_trick=realnvp_rezero_trick
    )

    return realnvp
