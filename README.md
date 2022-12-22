# reparametrized pytorch
Learning with reparametrized gradients for native pytorch modules.

## Introduction
Mathematical formulations of learning with samples from reparametrized distributions separate posteriors $q$ from structures of likelihoods (networks) $f$. For example, for ELBO $\mathcal{L} = E_q \left( \log p(y|f(x|w)) + \log p(w) - \log q(w|\lambda) \right) \approx \frac{1}{S} \sum_{w \sim q(w|\lambda)} \left( \log p(f(y,x|w)) + \log p(w) - \log q(w|\lambda) \right)$, $f$ takes parameters (weights) $w$ as an argument but is not tied anyhow to the sampling distribution $q$.
At the same time, all the available pytorch libraries (for example, [bayesian torch](https://github.com/IntelLabs/bayesian-torch)) work by replacing pytorch native layers with custom layers. As a consequence, it is impossible to sample jointly for multiple layers or pass additional information to the sampling code.

We achieve full separation of sampling procedures from network structures by implementing [custom procedure](reparametrized/parameters.py) for loading state dictionary (pytorch's default *load_state_dict* loses gradients of samples) for an arbitrary network's parameters. 

## Installation

The library can be installed using:
`pip install git+https://github.com/tkusmierczyk/reparametrized_pytorch.git#egg=reparametrized`

## Limitations

For native pytorch modules it is impossible to pass at once multiple sampled parameter sets for a network. Hence, when we sampled more than one set, we need to loop over them using *take_parameters_sample*. In each iteration the *forward* operation is then repeated, which makes execution slower.

## Demos
6. [Learn Normalizing Flow for Bayesian linear regression](notebooks/bayesian_linear_regression_bnn_wrapper.ipynb) (13 dimensions; [using BNN wrapper class](reparametrized/bnn_wrapper.py))
5. [Learn full-rank Normal for Bayesian linear regression](notebooks/bayesian_linear_regression_full_rank.ipynb)
4. [Learn factorized Normal for Bayesian linear regression](notebooks/bayesian_linear_regression_mfvi.ipynb)
3. [Minimize KL(q|p) for q modeled as Bayesian Hypernetwork](notebooks/bayesian_hypernet_matching_full_rank_gaussian_prior.ipynb)
2. [Minimize KL(q|p) for q modeled as RealNVP flow](notebooks/realnvp_matching_full_rank_gaussian_prior.ipynb)
1. [Minimize KL(q|p) for q and p being factorized Normal](notebooks/matching_gaussian_prior.ipynb)

## Credits
RealNVP implementation is based on [code](https://jmtomczak.github.io/blog/3/3_flows.html) from Jakub Tomczak.
Code for flows includes contributions by Bartosz Wójcik [*bartwojc(AT)gmail.com*] and Marcin Sendera [*marcin.sendera(AT)gmail.com*].

