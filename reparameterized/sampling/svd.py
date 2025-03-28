import torch
from typing import Tuple, Callable, Dict


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
    **kwargs,
) -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    assert len(parameter.shape) == 2, "Currently only 2D parameters are supported."

    u, s, vh = torch.linalg.svd(parameter, full_matrices=False)

    u = u.clone().detach()
    vh = vh.clone().detach()
    s = s.clone().detach()

    r_sampler, variational_params, aux_objs = svd_create_inner_matrix_sampler(
        torch.diag(s), **kwargs
    )

    aux_objs["u"] = u
    aux_objs["vh"] = vh
    aux_objs["s"] = s

    def sampler(n_samples=1, u=u, vh=vh, s=s, svd_residuals=svd_residuals):
        r_samples, nlls = r_sampler(n_samples)
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
