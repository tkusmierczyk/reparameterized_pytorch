# reparametrized pytorch
Learning with reparametrized gradients for native pytorch modules.


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
