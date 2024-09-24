import normflows as nf
import logging
from .normflows_common import NormFlowWrapper, WeightsInitializer, train_nfm


# Define flows
def build_spline_flow(
    output_dim,
    spline_flow_K=16,
    spline_flow_hidden_units=128,
    spline_flow_hidden_layers=2,
    spline_flow_layer_cls=nf.flows.AutoregressiveRationalQuadraticSpline,
    spline_trainable_prior=False,
    parameters_shapes=None,
    pretrain_flow=False,
    pretrain_flow_1D_parameters="zeros",  # biases
    pretrain_flow_2D_parameters="xavier_uniform",  # weights
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
    q0 = nf.distributions.DiagGaussian(latent_size, trainable=spline_trainable_prior)
    logging.debug(f"[build_spline_flow] q0={q0}")

    # Construct flow model
    nfm = NormFlowWrapper(q0=q0, flows=flows)
    logging.debug(f"[build_spline_flow] nfm={nfm}")

    # Force building a flow with batchSize>1, so layers with the right shapes are built
    nfm.sample(2)

    if pretrain_flow:
        if parameters_shapes is None:
            raise ValueError("parameters_shapes must be provided if init_flow is True!")
        # Initialize flow to produce parameter samples from some default distributions
        target = WeightsInitializer(
            parameters_shapes,
            weight_init=pretrain_flow_2D_parameters,
            bias_init=pretrain_flow_1D_parameters,
        )
        train_nfm(nfm, target)

    return nfm
