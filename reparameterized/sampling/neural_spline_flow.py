import normflows as nf
import logging
from .normflows_common import NormFlowWrapper


# Define flows
def build_spline_flow(
    output_dim,
    spline_flow_K=16,
    spline_flow_hidden_units=128,
    spline_flow_hidden_layers=2,
    spline_flow_layer_cls=nf.flows.AutoregressiveRationalQuadraticSpline,
    parameters_shapes=None,
    **layer_args,
):
    logging.debug(
        f"[build_spline_flow] output_dim={output_dim} "
        f"spline_flow_K={spline_flow_K} "
        f"spline_flow_hidden_units={spline_flow_hidden_units} "
        f"spline_flow_hidden_layers={spline_flow_hidden_layers} "
        f"spline_flow_layer_cls={spline_flow_layer_cls} "
        f"layer_args={layer_args} "
    )
    if parameters_shapes:
        logging.warning(
            "[build_spline_flow] parameters_shapes is not None, but is not used by the flow!"
        )
    latent_size = output_dim
    K = spline_flow_K
    hidden_units = spline_flow_hidden_units
    hidden_layers = spline_flow_hidden_layers

    flows = []
    for _ in range(K):

        if layer_args:
            flows += [
                spline_flow_layer_cls(
                    latent_size, hidden_layers, hidden_units, layer_args
                )
            ]

        else:
            flows += [spline_flow_layer_cls(latent_size, hidden_layers, hidden_units)]

        flows += [nf.flows.LULinearPermute(latent_size)]

    # Set base distribuiton
    q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
    logging.debug(f"[build_spline_flow] q0={q0}")

    # Construct flow model
    nfm = NormFlowWrapper(q0=q0, flows=flows)
    logging.debug(f"[build_spline_flow] nfm={nfm}")
    return nfm
