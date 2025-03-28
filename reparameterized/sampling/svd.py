import torch
from typing import Iterable, Tuple, Callable, Dict

from .multiparameter import create_multiparameter_sampler_dict, separate_parameters

import logging


def svd_projection(parameter, **kwargs):
    full_matrices = kwargs.pop("full_matrices", False)
    u, s, vh = torch.linalg.svd(parameter, full_matrices=full_matrices, **kwargs)

    s = torch.diag(s)

    u = u.clone().detach()
    s = s.clone().detach()
    vh = vh.clone().detach()

    return u, s, vh


def svd_inv(
    r_samples: torch.Tensor,
    u: torch.Tensor,
    vh: torch.Tensor,
    residuals: bool = False,
    sigma: torch.Tensor = None,
):
    if residuals and sigma is not None:
        samples = torch.stack(
            [
                torch.einsum(
                    "ij,jk,kl->il",
                    u,
                    r1.reshape((u.shape[1], vh.shape[0])) + sigma,
                    vh,
                ).T
                for r1 in r_samples
            ],
            dim=0,
        )

    else:
        samples = torch.stack(
            [
                torch.einsum(
                    "ij,jk,kl->il", u, r1.reshape((u.shape[1], vh.shape[0])), vh
                ).T
                for r1 in r_samples
            ],
            dim=0,
        )

    return samples


def create_svd_sampler(
    parameter: torch.Tensor,
    svd_create_inner_matrix_sampler: Callable,
    svd_residuals: bool = False,
    **create_sampler_kwargs,
) -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    if len(parameter.shape) != 2:
        logging.warning(
            "[create_svd_sampler] "
            f"Currently SVD is supported only for 2D parameters! Failed for {parameter.shape}. "
            f"Falling back to sampling directly with {svd_create_inner_matrix_sampler}."
        )
        return svd_create_inner_matrix_sampler(parameter, **create_sampler_kwargs)

    u, s, vh = svd_projection(parameter)

    r_sampler, variational_params, aux_objs = svd_create_inner_matrix_sampler(
        s, **create_sampler_kwargs
    )

    aux_objs["u"] = u
    aux_objs["vh"] = vh
    aux_objs["s"] = s

    def sampler(n_samples=1, **inner_sampler_kwargs):
        r_samples, nlls = r_sampler(n_samples, **inner_sampler_kwargs)
        assert (
            len(r_samples.shape) == 3
        ), f"Wrong shape of the inner matrix samples: {r_samples.shape}"
        assert r_samples.shape[0] == n_samples
        assert r_samples.shape[1] == s.shape[0]
        assert r_samples.shape[2] == s.shape[0]

        sample = svd_inv(r_samples, u, vh, svd_residuals, s).transpose(1, 2)
        assert sample.shape[0] == n_samples
        assert (
            sample.shape[1:] == parameter.shape
        ), f"sample.shape={sample.shape} != parameter.shape={parameter.shape}"

        return sample, nlls

    return sampler, variational_params, aux_objs


def create_multiparameter_svd_sampler(
    svd_create_inner_matrix_sampler: Callable,
    named_parameters: Iterable[Tuple[str, torch.Tensor]],
    svd_residuals: bool = False,
    **create_sampler_kwargs,
) -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    if hasattr(named_parameters, "items"):
        named_parameters = named_parameters.items()
    named_parameters = list(named_parameters)

    aux_objs = {}
    named_inner_parameters = []
    for name, parameter in named_parameters:
        if len(parameter.shape) == 2:
            logging.debug(
                f"[create_multiparameter_svd_sampler] Creating SVD projection for parameter={name}"
            )
            u, s, vh = svd_projection(parameter)

            named_inner_parameters.append((name, s))

            aux_objs[name + ".u"] = u
            aux_objs[name + ".s"] = s
            aux_objs[name + ".vh"] = vh

        else:
            logging.warning(
                f"[create_multiparameter_svd_sampler] Failed creating SVD projection for parameter={name}. "
                f"Falling back to sampling directly with {svd_create_inner_matrix_sampler}."
            )
            named_inner_parameters.append((name, parameter))

    # named_parameters = dict(named_parameters)

    # create a joint sampler for all inner matrices considered together
    multiparameter_sampler, variational_params, aux_objs_inner = (
        create_multiparameter_sampler_dict(
            svd_create_inner_matrix_sampler,
            named_inner_parameters,
            **create_sampler_kwargs,
        )
    )
    aux_objs.update(aux_objs_inner)

    def sampler(n_samples=1, **inner_sampler_kwargs):
        samples, nlls = multiparameter_sampler(n_samples, **inner_sampler_kwargs)

        for name, r_samples in samples.items():
            if (name + ".u") in aux_objs:  # SVD projection
                u = aux_objs[name + ".u"]
                s = aux_objs[name + ".s"]
                vh = aux_objs[name + ".vh"]

                assert (
                    len(r_samples.shape) == 3
                ), f"Wrong shape of the inner matrix samples: {r_samples.shape}"
                assert r_samples.shape[0] == n_samples
                assert r_samples.shape[1] == s.shape[0]
                assert r_samples.shape[2] == s.shape[0]

                samples1 = svd_inv(r_samples, u, vh, svd_residuals, s).transpose(1, 2)
                assert samples1.shape[0] == n_samples
                # assert samples1.shape[1:] == named_parameters[name].shape

                samples[name] = samples1

        return samples, nlls

    return sampler, variational_params, aux_objs
