import normflows as nf
import logging
from .normflows_common import NormFlowWrapper


# Define flows
def build_residual(
    output_dim,
    residual_flow_K=16,
    residual_hidden_units=128,
    residual_hidden_layers=3,
    residual_init_zeros=True,
    residual_lipschitz_const=0.9,
    residual_trainable_prior=False,
    parameters_shapes=None,
):
    logging.debug(
        f"[build_residual] output_dim={output_dim} "
        f"residual_flow_K={residual_flow_K} "
    )
    if parameters_shapes:
        logging.warning(
            "[build_residual] parameters_shapes is not None, but is not used by the flow!"
        )
    latent_size = output_dim
    K = residual_flow_K
    hidden_units = residual_hidden_units
    hidden_layers = residual_hidden_layers

    flows = []
    for i in range(K):
        net = nf.nets.LipschitzMLP(
            [latent_size] + [hidden_units] * (hidden_layers - 1) + [latent_size],
            init_zeros=residual_init_zeros,
            lipschitz_const=residual_lipschitz_const,
        )
        flows += [nf.flows.Residual(net, reduce_memory=True)]
        flows += [nf.flows.ActNorm(latent_size)]

    # Set target and q0
    q0 = nf.distributions.DiagGaussian(latent_size, trainable=residual_trainable_prior)

    # Construct flow model
    nfm = NormFlowWrapper(q0=q0, flows=flows)
    logging.debug(f"[build_residual] nfm={nfm}")
    return nfm
