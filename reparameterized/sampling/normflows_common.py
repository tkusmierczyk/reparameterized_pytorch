import normflows as nf
import logging
import torch
from tqdm import tqdm
import torch.nn.init as init


class NormFlowWrapper(nf.NormalizingFlow):
    def sample_raw_nf(self, *args, **kwargs):
        return super().sample(*args, **kwargs)

    def sample(self, batchSize: int, D=None, calculate_nll=False):
        x, log_prob = super().sample(batchSize)
        if D:
            x = x.reshape(-1, D)
        return (x, -log_prob) if calculate_nll else x


class ZerosSampler:
    def __init__(self, dim) -> None:
        self.dim = dim

    def sample(self, num_samples=1):
        return torch.zeros(num_samples, self.dim)


class WeightsInitializer:
    """
    A class for generating initializations for model weights and biases.

    This class takes a list of tensor shapes and generates initializations
    for weights and biases according to selected initialization algorithms.
    It can return the initializations either as a flattened tensor or as a
    list of tensors following the order of the provided shapes.

    Parameters:
        shapes (list of torch.Size): A list of shapes for the model parameters.
        flatten (bool, optional): If True (default), the initializations are
            flattened into a tensor of shape (n_samples, total_params). If False,
            the initializations are returned as a list of tensors.
        weight_init (str, optional): The initialization method for weights.
            Supported methods:
                - 'xavier_uniform'
                - 'xavier_normal'
                - 'kaiming_uniform'
                - 'kaiming_normal'
                - 'orthogonal'
                - 'normal'
                - 'uniform'
                - 'zeros' (initializes weights to zero)
        bias_init (str, optional): The initialization method for biases.
            Supported methods:
                - 'zeros' (default)
                - 'ones'
                - 'uniform'
                - 'normal'

    Methods:
        sample(n_samples):
            Generates n_samples initializations for the weights and biases
            according to the selected initialization methods.

            Parameters:
                n_samples (int): The number of samples to generate.

            Returns:
                If flatten is True:
                    A tensor of shape (n_samples, total_params) containing
                    the flattened parameter initializations.
                If flatten is False:
                    A list of tensors, each of shape (n_samples, *shape), following
                    the order of the provided shapes.

    Example:
        shapes = [torch.Size([48, 2]), torch.Size([48]), torch.Size([1, 48]), torch.Size([1])]
        initializer = WeightInitializer(
            shapes,
            flatten=False,
            weight_init='xavier_uniform',
            bias_init='zeros'
        )
        initializations = initializer.sample(n_samples=5)
    """

    def __init__(
        self, shapes, weight_init="xavier_uniform", bias_init="zeros", flatten=True
    ):
        self.shapes = shapes
        self.flatten = flatten
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.params_shapes = shapes  # Keep the order
        self.weights_indices = []
        self.biases_indices = []
        for idx, shape in enumerate(shapes):
            if len(shape) > 1:
                self.weights_indices.append(idx)
            else:
                self.biases_indices.append(idx)

    def sample(self, n_samples):
        init_params = []
        for shape in self.params_shapes:
            param_init = torch.empty((n_samples, *shape))
            init_params.append(param_init)

        # Initialize weights
        for idx in self.weights_indices:
            param = init_params[idx]
            for i in range(n_samples):
                self.init_weight(param[i])

        # Initialize biases
        for idx in self.biases_indices:
            param = init_params[idx]
            for i in range(n_samples):
                self.init_bias(param[i])

        if self.flatten:
            total_params = sum(
                torch.prod(torch.tensor(shape)) for shape in self.params_shapes
            )
            inits = torch.empty((n_samples, total_params))
            for i in range(n_samples):
                params = [
                    init_params[idx][i].flatten()
                    for idx in range(len(self.params_shapes))
                ]
                inits[i] = torch.cat(params)
            return inits
        else:
            # Return list of initializations following the order of the shapes
            return [init_params[idx] for idx in range(len(self.params_shapes))]

    def init_bias(self, param):
        if self.bias_init == "zeros":
            init.zeros_(param)
        elif self.bias_init == "ones":
            init.ones_(param)
        elif self.bias_init == "uniform":
            init.uniform_(param)
        elif self.bias_init == "normal":
            init.normal_(param)
        elif self.bias_init == "normal_small":
            init.normal_(param, std=0.01)
        else:
            raise ValueError(f"Unsupported bias initializer: {self.bias_init}")

    def init_weight(self, param):
        if self.weight_init == "xavier_uniform":
            init.xavier_uniform_(param)
        elif self.weight_init == "xavier_normal":
            init.xavier_normal_(param)
        elif self.weight_init == "kaiming_uniform":
            init.kaiming_uniform_(param, nonlinearity="relu")
        elif self.weight_init == "kaiming_normal":
            init.kaiming_normal_(param, nonlinearity="relu")
        elif self.weight_init == "orthogonal":
            init.orthogonal_(param)
        elif self.weight_init == "normal":
            init.normal_(param)
        elif self.weight_init == "normal_small":
            init.normal_(param, std=0.01)
        elif self.weight_init == "uniform":
            init.uniform_(param)
        elif self.weight_init == "zeros":
            init.zeros_(param)
        else:
            raise ValueError(f"Unsupported weight initializer: {self.weight_init}")


def train_nfm(
    model,
    target,
    mask=None,
    max_iter=10000,
    num_samples=10,
    lr=5e-4,
    weight_decay=1e-5,
    early_stopping_n_iters=30,
):
    # Train model

    logging.info(f"[train_nfm] Training NF to match = {target}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss, best_it = float("inf"), -early_stopping_n_iters
    for iteration in tqdm(range(max_iter)):
        optimizer.zero_grad()

        # Get training samples
        x = target.sample(num_samples)

        # Compute loss
        loss = model.forward_kld(x)
        if loss.item() < best_loss:
            best_loss, best_it = loss.item(), iteration

        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        if iteration > best_it + early_stopping_n_iters:
            logging.info(
                f"[train_nfm] Early stopping due to no improvement in {early_stopping_n_iters} iterations."
            )
            break
