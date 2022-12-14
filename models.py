import torch
import gpytorch
import math
from typing import List

# from blitz.modules import BayesianLinear
# from blitz.utils import variational_estimator


class LSTMFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 0,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super(LSTMFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.encoder = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            proj_size=output_size,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor):
        output, self.hidden_state = self.encoder(x)
        return output


class MLPFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: List[int],
        output_size: int,
        dropout: float = 0.0,
    ):
        super(MLPFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = len(hidden_size)
        self.dropout = dropout

        self.input_layer = torch.nn.ReLU(
            torch.nn.Linear(self.input_size, self.hidden_size[0])
        )

        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_size[i], self.hidden_size[i + 1])
                for i in range(self.num_layers - 1)
            ]
        )

        self.output_layer = torch.nn.tanh(
            torch.nn.Linear(self.hidden_size[-1], self.output_size)
        )

        if dropout > 0:
            self.dropout_layer = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.input_layer(x)
        for linear in self.linears:
            x = linear(x)
            if self.dropout > 0:
                x = self.dropout_layer(x)
            x = torch.nn.ReLU(x)
        x = self.output_layer(x)
        return x


# @variational_estimator
# class BayesFeatureExtractor(torch.nn.Module):
#     def __init__(
#         self,
#         input_size: int,
#         hidden_size: List[int],
#         output_size: int,
#         num_layers: int = 1,
#     ):
#         super(BayesFeatureExtractor, self).__init__()

#         self.input_layer = torch.nn.ReLU(BayesianLinear(input_size, hidden_size[0]))

#         self.blinears = torch.nn.ModuleList(
#             [
#                 BayesianLinear(hidden_size[i], hidden_size[i + 1])
#                 for i in range(num_layers)
#             ]
#         )

#         self.output_layer = torch.nn.tanh(BayesianLinear(hidden_size[-1], output_size))

#     def forward(self, x):
#         x = self.input_layer(x)
#         for blinear in self.blinears:
#             x = torch.nn.ReLU(blinear(x))
#         x = self.output_layer(x)
#         return x


class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10.0, 10.0), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = (
            gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.GridInterpolationVariationalStrategy(
                    self,
                    grid_size=grid_size,
                    grid_bounds=[grid_bounds],
                    variational_distribution=variational_distribution,
                ),
                num_tasks=num_dim,
            )
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)