# -- coding: utf-8 --
import torch
from torch import nn
import gpytorch as gpy
import numpy as np
import time
from .nn import NN


class HybridModel(gpy.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood, layers, activation='tanh', kernel='rbf'):
        self.kernel = kernel
        self.activation = activation
        self.x_train = torch.Tensor(x_train)
        self.y_train = torch.Tensor(y_train)
        self.layers = layers
        self.ard_num_dims = self.layers[-1]
        super(HybridModel, self).__init__(self.x_train, self.y_train, likelihood)
        self.feature_extractor = NN(layers, self.activation)
        self.mean_module = gpy.means.ConstantMean()
        self.covar_module = gpy.kernels.ScaleKernel(self._kernel(self.kernel))

        # self.scale_to_bounds = gpy.utils.grid.ScaleToBounds(-1., 1.)
        self.time = 0
        self.name = 'Hybrid'
        self.length_scale = None
        self.output_scale = None
        self.noise_variance = None
        self.constant_mean = None
        self.alpha = None
        self.weights = []
        self.biases = []
        self.features = None

    def forward(self, x):
        x = self.feature_extractor(x)
        # x = self.scale_to_bounds(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, callback=None, batch_size=10, epochs=1000, learning_rate=1e-2, weight_decay=0.0):
        loss_func = gpy.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # 获取所有需要训练的参数
        trainable_params = {name: param for name, param in self.named_parameters()}

        # 打印所有训练参数的名字和大小
        for name, param in trainable_params.items():
            print(f'Parameter: {name}, Size: {param.size()}')

        self.train()
        self.likelihood.train()
        start_time = time.time()
        train_loss = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            permutation = torch.randperm(len(self.x_train))
            for i in range(0, len(self.x_train), batch_size):
                idx = permutation[i:i + batch_size]
                x_batch, y_batch = self.x_train[idx], self.y_train[idx]
                predictions = self.forward(x_batch)
                loss = -loss_func(predictions, y_batch)
                loss = loss.mean()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()

            epoch_loss /= len(self.x_train) / batch_size
            if callback:
                callback(epoch, epoch_loss)
            train_loss.append(epoch_loss)
        end_time = time.time()
        self.time = end_time - start_time
        self.save_params()
        # 输出所有训练参数
        print("All training parameters after training:")
        for name, param in trainable_params.items():
            print(f'Parameter: {name}, Value: {param.data}')

    def predict(self, x, return_std=False):
        x = torch.from_numpy(x).type(torch.float32)
        # print(f'Input shape: {x.shape}')  # Debug information
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # print(f'Transformed input shape: {x_transformed.shape}')  # Debug information
            y = self(x).mean  # Access mean prediction from the GP model
            if return_std:
                y_var = self(x).variance  # Access variance prediction from the GP model
                return y.numpy(), y_var.numpy()
            else:
                return y.numpy()

    def _kernel(self, kernel):

        if kernel == 'rbf':
            kernel = gpy.kernels.RBFKernel()
        elif kernel == 'linear':
            kernel = gpy.kernels.LinearKernel()
        elif kernel == 'polynomial':
            kernel = gpy.kernels.PolynomialKernel(power=2)
        elif kernel == 'RationalQuadratic':
            kernel = gpy.kernels.RQKernel()
        elif kernel == 'ExpSineSquared':
            kernel = gpy.kernels.CosineKernel()
        elif kernel == 'Matern':
            kernel = gpy.kernels.MaternKernel(nu=1.5)
        elif kernel == 'Sum_RBF':
            kernel = gpy.kernels.RBFKernel() + gpy.kernels.RBFKernel()
        elif kernel == 'Sum_RQ':
            kernel = gpy.kernels.RQKernel() + gpy.kernels.RQKernel()
        return kernel

    def save_params(self):
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                self.weights.append(layer.weight.data.numpy())
                self.biases.append(layer.bias.data.numpy())

        if self.kernel == 'rbf':
            self.length_scale = self.covar_module.base_kernel.lengthscale.item()
        self.output_scale = self.covar_module.outputscale.item()
        self.noise_variance = self.likelihood.noise
        self.constant_mean = self.mean_module.constant

        self.eval()
        with torch.no_grad():
            self.features = self.feature_extractor(self.x_train)
            K = self.covar_module(self.features).to_dense() + torch.eye(self.features.size(0)) * self.noise_variance
            K_inv = torch.inverse(K)
            self.alpha = torch.matmul(K_inv, self.y_train - self.constant_mean).cpu().detach().numpy()

