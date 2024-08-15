import numpy as np
import torch
from torch import nn
import time

import matplotlib.pyplot as plt


class NN(nn.Sequential):
    def __init__(self, layers, activation='tanh', is_classifier=False):
        if is_classifier:
            self.name = 'NNClf'
        else:
            self.name = 'NN'
        self.is_classifier = is_classifier
        self.layers = layers
        self.activation = activation
        self.weights = []
        self.biases = []
        self.time = 0
        super().__init__(*self._build_layers(layers))

    def fit(
        self, 
        x,
        y,
        callback=None,
        batch_size=10, 
        epochs=1000,
        learning_rate=1e-2, 
        weight_decay=0.0,
        loss_func=nn.MSELoss(), 
        iprint=False
    ):
        if self.name == 'NNClf':
            loss_func = nn.BCEWithLogitsLoss()
        x_train, y_train = torch.Tensor(x), torch.Tensor(y).unsqueeze(1)
        optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train()
        start_time = time.time()
        train_loss = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            permutation = torch.randperm(len(x_train))
            for i in range(0, len(x_train), batch_size):
                idx = permutation[i:i+batch_size]
                x_batch, y_batch = x_train[idx], y_train[idx]
                predictions = self.forward(x_batch)
                loss = loss_func(predictions, y_batch)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
            epoch_loss /= len(x_train)/batch_size
            if callback:
                callback(epoch, epoch_loss)
            train_loss.append(epoch_loss)
        end_time = time.time()
        # plt.plot(range(1, epochs + 1), train_loss, label='Training Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('Training Loss vs. Epochs')
        # plt.legend()
        # plt.show()
        self._get_params()
        self.time = end_time - start_time
        if iprint:
            print('{} model fitted! Time elapsed {:.5f} s'.format(self.name, end_time - start_time))

    def predict(self, x, return_proba=False, return_class=False, threshold=0.5):
        x = torch.Tensor(x)
        self.eval()
        y = self.forward(x).detach()
        c = torch.max(y, torch.tensor([0.]))
        proba = torch.sigmoid(y).detach()
        c[proba > threshold] = 1
        if return_class and return_proba:
            return y.numpy(), proba.numpy(), c.numpy()
        elif return_class:
            return y.numpy(), c.numpy()
        elif return_proba:
            return y.numpy(), proba.numpy()
        elif self.is_classifier:
            return proba.numpy()
        else:
            return y.numpy()
        
    def formulation(self, x):
        output = np.zeros((x.shape[0], 1))

        for ind, x_val in enumerate(x):

            w = self.weights
            b = self.biases
            af = self._af_selector()
            n = {i: set(range(w[i].shape[1])) for i in range(len(w))}
            a = {key: lambda x: af(x) for key in range(len(w) - 1)}
            a[len(a)] = lambda x: x

            def f(i):
                if i == -1:
                    return x_val
                return a[i](sum(torch.from_numpy(w[i])[:, j] * f(i-1)[j] for j in n[i]) + torch.from_numpy(b[i]))

            output_val = f(len(self.weights) - 1)
            output[ind] = output_val.numpy()

        return output
    
    def _get_params(self):
        for layer in self:
            if isinstance(layer, nn.Linear):
                self.weights.append(layer.weight.data.numpy())
                self.biases.append(layer.bias.data.numpy())
    
    def _af_selector(self):
        if self.activation == 'tanh':
            def f(x):
                return 1 - 2 / (np.exp( 2 * x ) + 1)
        
        elif self.activation == 'sigmoid':
            def f(x):
                return 1 / (1 + np.exp( -x ))
        elif self.activation == 'softplus':
            def f(x):
                return np.log(1 + np.exp(x))

        elif self.activation == 'relu':
            def f(x):
                return np.maximum(0, x)
        
        elif self.activation == 'linear':
            def f(x):
                return x

        elif self.activation == 'hardsigmoid':
            def f(x):
                y = np.zeros_like(x)
                for i in range(len(y)):
                    if x[i] >= 3:
                        y[i] = 1
                    elif x[i] <= -3:
                        y[i] = 0
                    else:
                        y[i] = x[i] / 6 + 0.5
                return y
        
        return f

    def _activation_selector(self):
        if self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'softplus':
            return nn.Softplus()
        elif self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'hardsigmoid':
            return nn.Hardsigmoid()
        elif self.activation == 'linear':
            return nn.Identity()
        elif self.activation == 'leaky relu':
            return nn.LeakyReLU()

    def _build_layers(self, layers):
        torch_layers = []
        for i in range(len(layers) - 1):
            torch_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:  # Apply activation for all but the last layer
                torch_layers.append(self._activation_selector())
        print(torch_layers)
        return torch_layers
