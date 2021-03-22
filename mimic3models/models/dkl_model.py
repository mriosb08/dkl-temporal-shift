import torch.nn.functional as F
from torch import nn
import torch
import os
import gpytorch
import math

class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_dim=1, grid_size=200, Xu=None): #TODO try 150 or 200
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a MultitaskVariationalStrategy so that our output is a vector-valued GP
        # TODO change multitask becasue it trains a matrix A over different clases
        variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim
        )
        #variational_strategy = gpytorch.variational.GridInterpolationVariationalStrategy(
        #    self, grid_size=grid_size, grid_bounds=[grid_bounds], 
        #    variational_distribution=variational_distribution,
        #    )
        #variational_strategy = gpytorch.variational.VariationalStrategy(
        #    self, Xu, variational_distribution, learn_inducing_locations=True
        #)

        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
            #gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, batch_size=torch.Size([num_dim]), ard_num_dims=1)
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-100., 100.), grid_size=200, grid_dim=1, Xu=None):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_size=grid_size, grid_bounds=grid_bounds, grid_dim=grid_dim, Xu=Xu)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        features = self.feature_extractor(x)
        features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        # This next line makes it so that we learn a GP for each feature
        # for multitask!!
        features = features.transpose(-1, -2).unsqueeze(-1)
        #print(features.size())
        #features = features.unsqueeze(-1) # for grid add a dim? .unsqueeze(-1)
        #print(self.gp_layer.variational_strategy.inducing_points)
        res = self.gp_layer(features)
        return res


class GPClassificationModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, feature_extractor):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.feature_extractor = feature_extractor

    def forward(self, x):
        x = self.feature_extractor(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred



