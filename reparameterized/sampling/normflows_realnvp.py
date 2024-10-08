import normflows as nf
import logging
import torch
from .normflows_common import (
    NormFlowWrapper,
    train_nfm,
)


# Define flows
def build_realnvp_flow(
    output_dim,
    realnvp_flow_K=16,
    realnvp_init_zeros=True,
    realnvp_trainable_prior=True,
    parameters_shapes=None,
    pretrain_flow_target=None,
):
    logging.debug(
        f"[build_realnvp] output_dim={output_dim} " f"realnvp_flow_K={realnvp_flow_K} "
    )
    if parameters_shapes:
        logging.warning(
            "[build_realnvp] parameters_shapes is not None, but is not used by the flow!"
        )
    latent_size = output_dim
    K = realnvp_flow_K

    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP(
            [latent_size, 2 * latent_size, latent_size], init_zeros=realnvp_init_zeros
        )
        t = nf.nets.MLP(
            [latent_size, 2 * latent_size, latent_size], init_zeros=realnvp_init_zeros
        )
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]

    # Set target and q0
    q0 = nf.distributions.DiagGaussian(latent_size, trainable=realnvp_trainable_prior)

    # Construct flow model
    nfm = NormFlowWrapper(q0=q0, flows=flows)
    logging.debug(f"[build_realnvp] nfm={nfm}")

    # Force building a flow with batchSize>1, so layers with the right shapes are built
    nfm.sample(2)

    if pretrain_flow_target:
        # Initialize flow to produce parameter samples from some default distributions
        train_nfm(nfm, pretrain_flow_target)

    return nfm
