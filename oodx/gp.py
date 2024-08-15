import numpy as np
from numpy.linalg import inv, slogdet
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, RationalQuadratic, ExpSineSquared, Matern, Sum, \
    ConstantKernel as con
import time


class GPR(GaussianProcessRegressor):
    def __init__(self, kernel='rbf', noise=1e-10, porder=2, n_restarts_optimizer=300):
        self.name = 'GPR'
        self.kernel_name = kernel
        self.noise = noise
        self.x_train = None
        self.length_scale = None
        self.length_scale_1 = None
        self.constant_value = None
        self.sigma_0 = None
        self.inv_K = None
        self.porder = porder
        self.time = None
        self.scale_mixture = None
        self.scale_mixture_1 = None
        self.periodicity = None
        self.nu = None
        super().__init__(kernel=self._kernel(kernel), alpha=noise, n_restarts_optimizer=n_restarts_optimizer)

    def _kernel(self, kernel):
        if kernel == 'rbf':
            kernel = con(1.0, (1e-5, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e5))
        elif kernel == 'linear':
            kernel = con(1.0, (1e-5, 1e6)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-4, 1e5))
        elif kernel == 'polynomial':
            kernel = con(1.0, (1e-5, 1e6)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-4, 1e5)) ** self.porder
        elif kernel == 'RationalQuadratic':
            kernel = con(1.0, (1e-5, 1e6)) * RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-4, 1e5))
        elif kernel == 'ExpSineSquared':
            kernel = con(1.0, (1e-5, 1e6)) * ExpSineSquared(length_scale=1.0, length_scale_bounds=(1e-4, 1e5))
        elif kernel == 'Matern':
            kernel = con(1.0, (1e-5, 1e6)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e5), nu=1.5)
        elif kernel == 'Sum_RBF':
            kernel = con(1.0, (1e-5, 1e6)) * Sum(RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e5)),
                                                 RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e5)))
        elif kernel == 'Sum_RQ':
            kernel = con(1.0, (1e-5, 1e6)) * Sum(RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-4, 1e5)),
                                                 RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-4, 1e5)))
        return kernel

    def fit(self, x, y, iprint=False):
        self.x_train = x
        with np.errstate(divide='ignore'):
            start_time = time.time()
            super().fit(x, y)
            end_time = time.time()
        self._save_params()
        self.time = end_time - start_time
        if iprint:
            print('{} model fitted! Time elapsed {:.5f} s'.format(self.name, end_time - start_time))

    def _save_params(self):
        params = self.kernel_.get_params()
        self.constant_value = params['k1__constant_value']
        if self.kernel_name == 'rbf':
            self.length_scale = params['k2__length_scale']
        elif self.kernel_name == 'linear':
            self.sigma_0 = params['k2__sigma_0']
        elif self.kernel_name == 'polynomial':
            self.sigma_0 = params['k2__kernel__sigma_0']
        elif self.kernel_name == 'RationalQuadratic':
            self.scale_mixture = params['k2__alpha']
            self.length_scale = params['k2__length_scale']
        elif self.kernel_name == 'ExpSineSquared':
            self.length_scale = params['k2__length_scale']
            self.periodicity = params['k2__periodicity']
        elif self.kernel_name == 'Matern':
            self.length_scale = params['k2__length_scale']
            self.nu = params['k2__nu']
        elif self.kernel_name == 'Sum_RBF':
            self.length_scale = params['k2__k1__length_scale']
            self.length_scale_1 = params['k2__k2__length_scale']
        elif self.kernel_name == 'Sum_RQ':
            self.length_scale = params['k2__k1__length_scale']
            self.scale_mixture = params['k2__k1__alpha']
            self.length_scale_1 = params['k2__k2__length_scale']
            self.scale_mixture_1 = params['k2__k2__alpha']
        self.alpha = self.alpha_.ravel()
        K = self.kernel_(self.x_train, self.x_train) + np.eye(self.x_train.shape[0]) * self.noise
        self.inv_K = inv(K)

    def predict(self, x, return_std=False, return_cov=False):
        if return_std:
            return super().predict(x, return_std=True)
        elif return_cov:
            return super().predict(x, return_cov=True)
        else:
            return super().predict(x, return_std=False)

    def formulation(self, x, return_std=False):
        n = self.x_train.shape[0]  # number of training samples
        m = self.x_train.shape[1]  # number of input dimensions
        # squared exponential kernel evaluated at training and new inputs

        if self.kernel_name == 'rbf':
            k = self.constant_value * np.exp(
                -sum(0.5 / self.length_scale ** 2 * (
                        x[:, j].reshape(1, -1) - self.x_train[:, j].reshape(-1, 1)
                ) ** 2 for j in range(m))
            )
        if self.kernel_name == 'linear':
            k = self.constant_value * (
                    self.sigma_0 ** 2 + sum(
                x[:, j].reshape(1, -1) * self.x_train[:, j].reshape(-1, 1) for j in range(m))
            )
        if self.kernel_name == 'polynomial':
            k = self.constant_value * (
                    self.sigma_0 ** 2 + sum(
                x[:, j].reshape(1, -1) * self.x_train[:, j].reshape(-1, 1) for j in range(m))
            ) ** self.porder
        # linear predictor of mean function
        pred = sum(k[i] * self.alpha[i] for i in range(n)).reshape(-1, 1)
        if return_std:
            # vector-matrix-vector product of k^T K^-1 k
            vMv = sum(
                k[i] * sum(
                    self.inv_K[i, j] * k[j] for j in range(n)
                ) for i in range(n)
            )
            # variance and std at new input
            if self.kernel_name == 'rbf':
                k_ss = np.array(self.constant_value).reshape(1, 1)
            elif self.kernel_name == 'linear':
                k_ss = self.constant_value * (
                        self.sigma_0 ** 2 + sum(x[:, j].reshape(1, -1) * x[:, j].reshape(-1, 1) for j in range(m))
                )
            elif self.kernel_name == 'polynomial':
                k_ss = self.constant_value * (
                        self.sigma_0 ** 2 + sum(x[:, j].reshape(1, -1) * x[:, j].reshape(-1, 1) for j in range(m))
                ) ** self.porder
            var = np.diag(k_ss) - vMv
            std = np.sqrt(var)
            return pred, std
        else:
            return pred


class GPC:
    def __init__(self):
        self.name = 'GPC'
        self.x_train = None
        self.t_train = None
        self.l = None
        self.sigma_f = None
        self.delta = None
        self.inv_P = None
        self.time = None

    def _kernel(self, x1, x2):
        sq_dist = sum(
            (x1[:, j].reshape(1, -1) - x2[:, j].reshape(-1, 1)) ** 2 for j in range(x1.shape[1])
        )
        sq_exp = self.sigma_f ** 2 * np.exp(- 0.5 / self.l ** 2 * sq_dist)
        return sq_exp

    def fit(self, x, t, iprint=False):
        self.x_train = x
        self.t_train = t
        self._calculate_params(iprint=iprint)

    def predict(self, x, return_std=False, return_class=False, threshold=0.5):
        a = self._posterior_mode()
        k_s = self._kernel(x, self.x_train)
        mu = k_s.T.dot(self.t_train - self._sigmoid(a))
        var = self.sigma_f ** 2 - k_s.T.dot(self.inv_P).dot(k_s)
        var = np.diag(var).clip(min=0).reshape(-1, 1)
        beta = np.sqrt(1 + 3.1416 / 8 * var)
        prediction = self._sigmoid(mu / beta.ravel())
        if return_class:
            c = prediction.copy()
            c[c >= threshold] = 1
            c[c < threshold] = 0
            return prediction, c
        elif return_std:
            return prediction, np.sqrt(var)
        else:
            return prediction

    def formulation(self, x):
        n = self.x_train.shape[0]
        m = self.x_train.shape[1]
        sq_exp = np.exp(
            -sum(0.5 / self.l ** 2 * (
                    x[:, j].reshape(1, -1) - self.x_train[:, j].reshape(-1, 1)) ** 2 for j in range(m))
        )
        mu = self.sigma_f ** 2 * sum(self.delta[i] * sq_exp[i] for i in range(n))
        var = self.sigma_f ** 2 * (1 - sum(
            sq_exp[i] * self.sigma_f ** 2 * sum(
                sq_exp[i_] * self.inv_P[i, i_] for i_ in range(n)) for i in range(n)))
        beta = np.sqrt(1 + 3.1416 / 8 * var)
        prediction = 1 / (1 + np.exp(- mu / beta))
        return prediction.reshape(-1, 1)

    def _posterior_mode(self, max_iter=10, tol=1e-9):
        K = self._kernel(self.x_train, self.x_train)
        a = np.zeros_like(self.t_train)
        I = np.eye(self.x_train.shape[0])
        for i in range(max_iter):
            W = self._sigmoid(a) * (1 - self._sigmoid(a))
            W = np.diag(W.ravel())
            inv_Q = inv(I + W @ K)
            a_new = (K @ inv_Q).dot(self.t_train - self._sigmoid(a) + W.dot(a))
            a_diff = np.abs(a_new - a)
            a = a_new
            if not np.any(a_diff > tol):
                break
        return a

    def _calculate_params(self, iprint):
        start_time = time.time()
        params = minimize(
            fun=self._opt_fun,
            x0=[1.0, 1.0],
            bounds=[(1e-6, None), (1e-6, None)],
            method='L-BFGS-B',
            options={'iprint': -1})
        end_time = time.time()
        self.l = params.x[0]
        self.sigma_f = params.x[1]
        a = self._posterior_mode()
        W = self._sigmoid(a) * (1 - self._sigmoid(a))
        I = np.eye(self.x_train.shape[0])
        W = np.diag(W.ravel()) + 1e-5 * I
        K = self._kernel(self.x_train, self.x_train)
        P = inv(W) + K
        self.inv_P = inv(P)
        self.delta = self.t_train - self._sigmoid(a)
        self.time = end_time - start_time
        if iprint:
            print('{} model fitted! Time elapsed {:.5f} s'.format(self.name, end_time - start_time))

    def _opt_fun(self, theta):
        I = np.eye(self.x_train.shape[0])
        self.l = theta[0]
        self.sigma_f = theta[1]
        K = self._kernel(self.x_train, self.x_train) + 1e-5 * I
        inv_K = inv(K)
        a = self._posterior_mode()
        W = self._sigmoid(a) * (1 - self._sigmoid(a))
        W = np.diag(W.ravel())
        ll = self.t_train.T.dot(a) - np.sum(np.log(1.0 + np.exp(a))) - 0.5 * (
                a.T.dot(inv_K).dot(a) +
                slogdet(K)[1] +
                slogdet(W + inv_K)[1])
        return -ll.ravel()

    @staticmethod
    def _sigmoid(a):
        return 1 / (1 + np.exp(-a))
