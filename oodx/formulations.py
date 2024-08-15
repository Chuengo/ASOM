import pyomo.environ as pyo
import numpy as np
from scipy.special import kv, gamma


class OODXBlock:

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.formulation = None

    def get_formulation(self, return_std=False):
        if self.model.name == 'NN' or self.model.name == 'NNClf':
            if self.model.activation == 'relu':
                self.formulation = pyo.Block(rule=self._nn_relu_rule)
            elif self.model.activation == 'tanh':
                self.formulation = pyo.Block(rule=self._nn_tanh_rule)
            elif self.model.activation == 'softplus':
                self.formulation = pyo.Block(rule=self._nn_softplus_rule)
            elif self.model.activation == 'sigmoid':
                self.formulation = pyo.Block(rule=self._nn_sigmoid_rule)
            elif self.model.activation == 'hardsigmoid':
                self.formulation = pyo.Block(rule=self._nn_hardsigmoid_rule)
            elif self.model.activation == 'linear':
                self.formulation = pyo.Block(rule=self._nn_linear_rule)
            elif self.model.activation == 'leakyrelu':
                self.formulation = pyo.Block(rule=self._nn_leakyrelu_rule)

        elif self.model.name == 'GPR':
            if self.model.kernel_name == 'rbf':
                if return_std:
                    self.formulation = pyo.Block(rule=self._gpr_rbf_std_rule)
                else:
                    self.formulation = pyo.Block(rule=self._gpr_rbf_rule)

            elif self.model.kernel_name == 'linear':
                if return_std:
                    self.formulation = pyo.Block(rule=self._gpr_linear_std_rule)
                else:
                    self.formulation = pyo.Block(rule=self._gpr_linear_rule)

            elif self.model.kernel_name == 'polynomial':
                if return_std:
                    self.formulation = pyo.Block(rule=self._gpr_polynomial_std_rule)
                else:
                    self.formulation = pyo.Block(rule=self._gpr_polynomial_rule)

            elif self.model.kernel_name == 'RationalQuadratic':
                self.formulation = pyo.Block(rule=self._gpr_rq_rule)

            elif self.model.kernel_name == 'ExpSineSquared':
                self.formulation = pyo.Block(rule=self._gpr_ess_rule)

            elif self.model.kernel_name == 'Matern':
                self.formulation = pyo.Block(rule=self._gpr_matern_rule)

            elif self.model.kernel_name == 'Sum_RBF':
                self.formulation = pyo.Block(rule=self._gpr_sum_rbf_rule)
            elif self.model.kernel_name == 'Sum_RQ':
                self.formulation = pyo.Block(rule=self._gpr_sum_rq_rule)

        elif self.model.name == 'GPC':
            self.formulation = pyo.Block(rule=self._gpc_rule)

        elif self.model.name == 'Hybrid':
            self.formulation = pyo.Block(rule=self._hybrid_rule)

        return self.formulation

    def _gpr_rbf_rule(self, m):
        # declare parameters
        x_train = self.model.x_train
        length_scale = self.model.length_scale
        constant_value = self.model.constant_value
        alpha = self.model.alpha

        # declare sets
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))
        n_outputs = set(range(1))

        # declare variables
        m.inputs = pyo.Var(n_inputs)
        m.outputs = pyo.Var(n_outputs)

        prediction = sum(alpha[i] * constant_value * pyo.exp(
            -sum(
                0.5 / length_scale ** 2 * ((m.inputs[j] - self.data.x_train_mean[j]) / self.data.x_train_std[j]  # scale
                                           - x_train[i, j]) ** 2 for j in n_inputs)
        ) for i in n_samples)

        prediction = prediction * self.data.y_train_std[0] + self.data.y_train_mean[0]

        # gpr constraint
        m.gpr = pyo.Constraint(expr=
                               m.outputs[0] == prediction
                               )

    def _gpr_linear_rule(self, m):
        # declare parameters
        x_train = self.model.x_train
        sigma_0 = self.model.sigma_0
        constant_value = self.model.constant_value
        alpha = self.model.alpha

        # declare sets
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))
        n_outputs = set(range(1))

        # declare variables
        m.inputs = pyo.Var(n_inputs)
        m.outputs = pyo.Var(n_outputs)

        prediction = sum(alpha[i] * constant_value * (
                sigma_0 ** 2 + sum((m.inputs[j] - self.data.x_train_mean[j]) / self.data.x_train_std[j]  # scale
                                   * x_train[i, j] for j in n_inputs)
        ) for i in n_samples)

        prediction = prediction * self.data.y_train_std[0] + self.data.y_train_mean[0]

        # gpr constraint        
        m.gpr = pyo.Constraint(expr=
                               m.outputs[0] == prediction
                               )

    def _gpr_polynomial_rule(self, m):
        # declare parameters
        x_train = self.model.x_train
        sigma_0 = self.model.sigma_0
        constant_value = self.model.constant_value
        alpha = self.model.alpha
        porder = self.model.porder

        # declare sets
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))
        n_outputs = set(range(1))

        # declare variables
        m.inputs = pyo.Var(n_inputs)
        m.outputs = pyo.Var(n_outputs)

        prediction = sum(alpha[i] * constant_value * (
                sigma_0 ** 2 + sum((m.inputs[j] - self.data.x_train_mean[j]) / self.data.x_train_std[j]  # scale
                                   * x_train[i, j] for j in n_inputs)
        ) ** porder for i in n_samples)
        prediction = prediction * self.data.y_train_std[0] + self.data.y_train_mean[0]

        # gpr constraint        
        m.gpr = pyo.Constraint(expr=
                               m.outputs[0] == prediction
                               )

    def _gpr_rq_rule(self, m):
        # declare parameters
        x_train = self.model.x_train
        alpha = self.model.alpha
        constant_value = self.model.constant_value
        length_scale = self.model.length_scale
        scale_mixture = self.model.scale_mixture

        # declare sets
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))
        n_outputs = set(range(1))

        # declare variables
        m.inputs = pyo.Var(n_inputs)
        m.outputs = pyo.Var(n_outputs)

        prediction = sum(
            alpha[i] * constant_value * (
                    1 + sum(
                (
                        (m.inputs[j] - self.data.x_train_mean[j]) / self.data.x_train_std[j]  # scale
                        - x_train[i, j]
                ) ** 2 / (2 * scale_mixture * length_scale ** 2) for j in n_inputs
            )
            ) ** (-scale_mixture) for i in n_samples)
        prediction = prediction * self.data.y_train_std[0] + self.data.y_train_mean[0]

        # gpr constraint
        m.gpr = pyo.Constraint(expr=
                               m.outputs[0] == prediction
                               )

    def _gpr_ess_rule(self, m):
        # declare parameters
        x_train = self.model.x_train
        alpha = self.model.alpha
        constant_value = self.model.constant_value
        length_scale = self.model.length_scale
        periodicity = self.model.periodicity

        # declare sets
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))
        n_outputs = set(range(1))

        # declare variables
        m.inputs = pyo.Var(n_inputs)
        m.outputs = pyo.Var(n_outputs)

        prediction = sum(
            alpha[i] * constant_value * pyo.exp(
                -2 * pyo.sin(
                    np.pi / periodicity * pyo.sqrt(
                        sum(
                            (
                                    (m.inputs[j] - self.data.x_train_mean[j]) / self.data.x_train_std[j]
                                    - x_train[i, j]
                            ) ** 2 for j in n_inputs
                        )
                    )
                ) ** 2

            ) for i in n_samples
        )
        prediction = prediction * self.data.y_train_std[0] + self.data.y_train_mean[0]

        # gor constraint
        m.gpr = pyo.Constraint(expr=
                               m.outputs[0] == prediction
                               )

    def _gpr_matern_rule(self, m):
        # declare parameters
        x_train = self.model.x_train
        length_scale = self.model.length_scale
        nu = self.model.nu
        constant_value = self.model.constant_value
        alpha = self.model.alpha

        # declare sets
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))
        n_outputs = set(range(1))

        # declare variables
        m.inputs = pyo.Var(n_inputs)
        m.outputs = pyo.Var(n_outputs)

        def matern_kernel(i):
            distance = pyo.sqrt(sum(((m.inputs[j] - self.data.x_train_mean[j]) / self.data.x_train_std[j]
                                     - x_train[i, j]) ** 2 for j in n_inputs))
            factor = np.sqrt(2 * nu) * distance / length_scale
            if nu == 0.5:
                return pyo.exp(-factor)
            elif nu == 1.5:
                return (1 + factor) * pyo.exp(-factor)
            elif nu == 2.5:
                return (1 + factor + factor ** 2 / 3) * pyo.exp(-factor)
            else:
                # Here we can't use scipy's kv directly, so we need to precompute or use another approach
                # For simplicity, let's assume nu is one of the handled values
                # In real scenarios, handle or precompute kv values for non-handled nu
                raise NotImplementedError("nu value not handled in this implementation")

        prediction = sum(
            alpha[i] * constant_value * matern_kernel(i)
            for i in n_samples
        )

        prediction = prediction * self.data.y_train_std[0] + self.data.y_train_mean[0]

        # gpr constraint
        m.gpr = pyo.Constraint(expr=
                               m.outputs[0] == prediction
                               )

    def _gpr_sum_rbf_rule(self, m):
        # declare parameters
        x_train = self.model.x_train
        length_scale = self.model.length_scale
        length_scale_1 = self.model.length_scale_1
        constant_value = self.model.constant_value
        alpha = self.model.alpha

        # declare sets
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))
        n_outputs = set(range(1))

        # declare variables
        m.inputs = pyo.Var(n_inputs)
        m.outputs = pyo.Var(n_outputs)

        prediction = sum(alpha[i] * constant_value * (pyo.exp(
            -sum(
                0.5 / length_scale ** 2 * ((m.inputs[j] - self.data.x_train_mean[j]) / self.data.x_train_std[j]  # scale
                                           - x_train[i, j]) ** 2 for j in n_inputs)
        ) + pyo.exp(
            -sum(
                0.5 / length_scale_1 ** 2 * (
                        (m.inputs[j] - self.data.x_train_mean[j]) / self.data.x_train_std[j]  # scale
                        - x_train[i, j]) ** 2 for j in n_inputs)
        )) for i in n_samples)

        prediction = prediction * self.data.y_train_std[0] + self.data.y_train_mean[0]

        # gpr constraint
        m.gpr = pyo.Constraint(expr=
                               m.outputs[0] == prediction
                               )

    def _gpr_sum_rq_rule(self, m):
        # declare parameters
        x_train = self.model.x_train
        alpha = self.model.alpha
        constant_value = self.model.constant_value
        length_scale = self.model.length_scale
        length_scale_1 = self.model.length_scale_1
        scale_mixture = self.model.scale_mixture
        scale_mixture_1 = self.model.scale_mixture_1
        # declare sets
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))
        n_outputs = set(range(1))

        # declare variables
        m.inputs = pyo.Var(n_inputs)
        m.outputs = pyo.Var(n_outputs)

        prediction = sum(
            alpha[i] * constant_value * ((
                                                 1 + sum(
                                             (
                                                     (m.inputs[j] - self.data.x_train_mean[j]) / self.data.x_train_std[
                                                 j]  # scale
                                                     - x_train[i, j]
                                             ) ** 2 / (2 * scale_mixture * length_scale ** 2) for j in n_inputs
                                         )
                                         ) ** (-scale_mixture) + (
                                                 1 + sum(
                                             (
                                                     (m.inputs[j] - self.data.x_train_mean[j]) / self.data.x_train_std[
                                                 j]  # scale
                                                     - x_train[i, j]
                                             ) ** 2 / (2 * scale_mixture_1 * length_scale_1 ** 2) for j in n_inputs
                                         )
                                         ) ** (-scale_mixture_1)) for i in n_samples)
        prediction = prediction * self.data.y_train_std[0] + self.data.y_train_mean[0]

        # gpr constraint
        m.gpr = pyo.Constraint(expr=
                               m.outputs[0] == prediction
                               )

    def _gpr_rbf_std_rule(self, m):
        # declare parameters
        x_train = self.model.x_train
        length_scale = self.model.length_scale
        constant_value = self.model.constant_value
        inv_K = self.model.inv_K

        # declare sets
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))
        n_outputs = set(range(1))

        # declare variables
        m.inputs = pyo.Var(n_inputs)
        m.outputs = pyo.Var(n_outputs)

        # gpr constraint representing -k^T K^-1 k in the std calc
        m.gpr_std = pyo.Constraint(expr=
                                   m.outputs[0] == - sum(
                                       constant_value * pyo.exp(-sum(
                                           0.5 / length_scale ** 2 * (
                                                   m.inputs[j] - x_train[i, j]
                                           ) ** 2 for j in n_inputs
                                       )) * sum(
                                           inv_K[i, k] * constant_value * pyo.exp(-sum(
                                               0.5 / length_scale ** 2 * (
                                                       m.inputs[j] - x_train[k, j]
                                               ) ** 2 for j in n_inputs
                                           )) for k in n_samples
                                       ) for i in n_samples
                                   )
                                   )

    def _gpr_linear_std_rule(self, m):
        # declare parameters
        x_train = self.model.x_train
        sigma_0 = self.model.sigma_0
        constant_value = self.model.constant_value
        inv_K = self.model.inv_K

        # declare sets
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))
        n_outputs = set(range(1))

        # declare variables
        m.inputs = pyo.Var(n_inputs)
        m.outputs = pyo.Var(n_outputs)

        # gpr constraint representing -k^T K^-1 k in the std calc
        m.gpr_std = pyo.Constraint(expr=
                                   m.outputs[0] == - sum(
                                       constant_value * (
                                               sigma_0 ** 2 + sum(m.inputs[j] * x_train[i, j] for j in range(m)))
                                       * sum(inv_K[i, k] * constant_value * (
                                               sigma_0 ** 2 + sum(m.inputs[j] * x_train[k, j] for j in range(m)))
                                             for k in n_samples) for i in n_samples
                                   )
                                   )

    def _gpr_polynomial_std_rule(self, m):
        # declare parameters
        x_train = self.model.x_train
        sigma_0 = self.model.sigma_0
        constant_value = self.model.constant_value
        inv_K = self.model.inv_K
        porder = self.model.porder

        # declare sets
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))
        n_outputs = set(range(1))

        # declare variables
        m.inputs = pyo.Var(n_inputs)
        m.outputs = pyo.Var(n_outputs)

        # gpr constraint representing -k^T K^-1 k in the std calc
        m.gpr_std = pyo.Constraint(expr=
                                   m.outputs[0] == - sum(
                                       constant_value * (sigma_0 ** 2 + sum(
                                           m.inputs[j] * x_train[i, j] for j in range(m))) ** porder
                                       * sum(inv_K[i, k] * constant_value * (sigma_0 ** 2 + sum(
                                           m.inputs[j] * x_train[k, j] for j in range(m))) ** porder
                                             for k in n_samples) for i in n_samples
                                   )
                                   )

    def _gpc_rule(self, m):
        # declare parameters
        x_train = self.model.x_train
        length_scale = self.model.l
        constant_value = self.model.sigma_f ** 2
        delta = self.model.delta
        invP = self.model.inv_P

        # declare sets
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))
        n_outputs = set(range(1))

        # declare variables
        m.inputs = pyo.Var(n_inputs)
        m.outputs = pyo.Var(n_outputs)

        # gpc constraint
        m.gpc = pyo.Constraint(expr=
                               m.outputs[0] ==
                               1 / (1 + pyo.exp(-constant_value * sum(
                                   delta[j] * pyo.exp(-sum(
                                       0.5 / length_scale ** 2 * (m.inputs[i] - x_train[j, i]) ** 2
                                       for i in n_inputs)) for j in n_samples) / pyo.sqrt(
                                   1 + 3.1416 / 8 * constant_value * (1 - sum(
                                       pyo.exp(-sum(0.5 / length_scale ** 2 * (
                                               m.inputs[i] - x_train[j, i]) ** 2
                                                    for i in n_inputs)) * constant_value * sum(
                                           pyo.exp(-sum(0.5 / length_scale ** 2 * (
                                                   m.inputs[j] - x_train[k, j]) ** 2
                                                        for j in n_inputs)) * invP[j, k]
                                           for k in n_samples) for j in n_samples))))))

    def _nn_general(self, m):
        # declare parameters
        W = self.model.weights
        b = self.model.biases

        # declare sets
        m.layers = list(range(len(self.model.layers)))
        m.nodes = {layer: set(range(nodes)) for layer, nodes in enumerate(self.model.layers)}

        # declare variables
        m.inputs = pyo.Var(m.nodes[0])
        m.z = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.a = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.outputs = pyo.Var(m.nodes[len(self.model.layers) - 1])

        # constraints
        m.c = pyo.ConstraintList()

        for n in m.nodes[1]:
            m.c.add(m.z[(1, n)] == sum(W[0][n, k] * m.inputs[k] for k in m.nodes[0]) + b[0][n])

        for l in m.layers[1:]:
            for n in m.nodes[l]:
                m.c.add(m.z[(l, n)] == sum(W[l - 1][n, k] * m.a[(l - 1, k)] for k in m.nodes[l - 1]) + b[l - 1][n])

        for n in m.nodes[len(self.model.layers) - 1]:
            m.c.add(m.outputs[n] == m.z[(len(self.model.layers) - 1, n)])

    def _nn_linear_rule(self, m):
        W = self.model.weights
        b = self.model.biases

        m.layers = list(range(len(self.model.layers)))
        m.nodes = {layer: set(range(nodes)) for layer, nodes in enumerate(self.model.layers)}

        last = len(m.layers) - 1

        m.inputs = pyo.Var(m.nodes[0])
        m.outputs = pyo.Var(m.nodes[len(self.model.layers) - 1])

        m.z = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.a = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.c = pyo.ConstraintList()

        for n in m.nodes[1]:
            m.c.add(m.z[(1, n)] == sum(W[0][n, k] * (m.inputs[k] - self.data.x_train_mean[k]) / self.data.x_train_std[k]
                                       for k in m.nodes[0]) + b[0][n])
            m.c.add(m.a[(1, n)] == m.z[(1, n)])

        for l in m.layers[2:]:
            for n in m.nodes[l]:
                m.c.add(m.z[(l, n)] == sum(W[l - 1][n, k] * m.a[(l - 1, k)] for k in m.nodes[l - 1]) + b[l - 1][n])
                m.c.add(m.a[(l, n)] == m.z[(l, n)])

        for n in m.nodes[last]:
            m.c.add(m.outputs[n] == m.z[(last, n)] * self.data.y_train_std[n] + self.data.y_train_mean[n])

        # # retrieve general  model
        # m. = pyo.Block(rule=self.__general)
        #
        # # declare variables
        # m.inputs = pyo.Var(m..nodes[0])
        # m.outputs = pyo.Var(m..nodes[len(self.model.layers) - 1])
        #
        # # constraints
        # m.c = pyo.ConstraintList()
        #
        # # activated layers return tanh of linear outputs
        # for l in m..layers[1:]:
        #     for n in m..nodes[l]:
        #         m.c.add(m..a[(l, n)] == m..z[(l, n)])
        #
        # # coect inputs to general model
        # for i in m..nodes[0]:
        #     m.c.add(m.inputs[i] == m..inputs[i])
        #
        # # coect outputs to general model
        # for i in m..nodes[len(self.model.layers) - 1]:
        #     m.c.add(m.outputs[i] == m..outputs[i])

    def _nn_tanh_rule(self, m):
        W = self.model.weights
        b = self.model.biases

        m.layers = list(range(len(self.model.layers)))

        m.nodes = {layer: set(range(nodes)) for layer, nodes in enumerate(self.model.layers)}

        last = len(m.layers) - 1

        m.inputs = pyo.Var(m.nodes[0])
        m.outputs = pyo.Var(m.nodes[len(self.model.layers) - 1])

        m.z = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.a = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.c = pyo.ConstraintList()

        for n in m.nodes[1]:
            m.c.add(m.z[(1, n)] == sum(W[0][n, k] * (m.inputs[k] - self.data.x_train_mean[k]) / self.data.x_train_std[k]
                                       for k in m.nodes[0]) + b[0][n])
            m.c.add(m.a[(1, n)] == 1 - 2 / (pyo.exp(2 * m.z[(1, n)]) + 1))

        for l in m.layers[2:]:
            for n in m.nodes[l]:
                m.c.add(m.z[(l, n)] == sum(W[l - 1][n, k] * m.a[(l - 1, k)] for k in m.nodes[l - 1]) + b[l - 1][n])
                m.c.add(m.a[(l, n)] == 1 - 2 / (pyo.exp(2 * m.z[(l, n)]) + 1))

        for n in m.nodes[last]:
            m.c.add(m.outputs[n] == m.z[(last, n)] * self.data.y_train_std[n] + self.data.y_train_mean[n])

        # z = {(i, j): 0.0 for i in m.layers for j in m.nodes[i]}
        # a = {(i, j): 0.0 for i in m.layers for j in m.nodes[i]}
        #
        # for n in m.nodes[1]:
        #     z[(1, n)] = sum(W[0][n, k] * (m.inputs[k] - self.data.x_train_mean[k]) / self.data.x_train_std[k]
        #                     for k in m.nodes[0]) + b[0][n]
        #     a[(1, n)] = 1 - 2 / (pyo.exp(2 * z[(1, n)]) + 1)
        #
        # for l in m.layers[2:]:
        #     for n in m.nodes[l]:
        #         z[(l, n)] = sum(W[l - 1][n, k] * a[(l - 1, k)] for k in m.nodes[l - 1]) + b[l - 1][n]
        #         a[(l, n)] = 1 - 2 / (pyo.exp(2 * z[(l, n)]) + 1)
        #
        # m.c = pyo.Constraint(expr=m.outputs[0] == z[(last, 0)] * self.data.y_train_std[0] + self.data.y_train_mean[0])

        # for n in m.nodes[1]:
        #     m.z[(1, n)] = sum(W[0][n, k] * m.inputs[k] for k in m.nodes[0]) + b[0][n]
        #     m.a[(1, n)] = 1 - 2 / (pyo.exp(2 * m.z[(1, n)]))
        #
        # for l in m.layers[1:]:
        #     for n in m.nodes[l]:
        #         print(l, n)

        # # retrieve general  model
        # m. = pyo.Block(rule=self.__general)
        #
        # # declare variables
        # m.inputs = pyo.Var(m..nodes[0])
        # m.outputs = pyo.Var(m..nodes[len(self.model.layers) - 1])
        #
        # # constraints
        # m.c = pyo.ConstraintList()
        #
        # # activated layers return tanh of linear outputs
        # for l in m..layers[1:]:
        #     for n in m..nodes[l]:
        #         m.c.add(m..a[(l, n)] == 1 - 2 / (pyo.exp(2 * m..z[(l, n)]) + 1))
        #
        # # coect inputs to general model
        # for i in m..nodes[0]:
        #     m.c.add(m.inputs[i] == m..inputs[i])
        #
        # # coect outputs to general model
        # for i in m..nodes[len(self.model.layers) - 1]:
        #     m.c.add(m.outputs[i] == m..outputs[i])

    def _nn_sigmoid_rule(self, m):
        W = self.model.weights
        b = self.model.biases

        m.layers = list(range(len(self.model.layers)))
        m.nodes = {layer: set(range(nodes)) for layer, nodes in enumerate(self.model.layers)}

        last = len(m.layers) - 1

        m.inputs = pyo.Var(m.nodes[0])
        m.outputs = pyo.Var(m.nodes[len(self.model.layers) - 1])

        m.z = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.a = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.c = pyo.ConstraintList()

        for n in m.nodes[1]:
            m.c.add(m.z[(1, n)] == sum(W[0][n, k] * (m.inputs[k] - self.data.x_train_mean[k]) / self.data.x_train_std[k]
                                       for k in m.nodes[0]) + b[0][n])
            m.c.add(m.a[(1, n)] == 1 / (1 + pyo.exp(-m.z[(1, n)])))

        for l in m.layers[2:]:
            for n in m.nodes[l]:
                m.c.add(m.z[(l, n)] == sum(W[l - 1][n, k] * m.a[(l - 1, k)] for k in m.nodes[l - 1]) + b[l - 1][n])
                m.c.add(m.a[(l, n)] == 1 / (1 + pyo.exp(-m.z[(l, n)])))

        for n in m.nodes[last]:
            m.c.add(m.outputs[n] == m.z[(last, n)] * self.data.y_train_std[n] + self.data.y_train_mean[n])
        # # retrieve general  model
        # m. = pyo.Block(rule=self.__general)
        #
        # # declare variables
        # m.inputs = pyo.Var(m..nodes[0])
        # m.outputs = pyo.Var(m..nodes[len(self.model.layers) - 1])
        #
        # # constraints
        # m.c = pyo.ConstraintList()
        #
        # # activated layers return sigmoid of linear outputs
        # for l in m..layers[1:]:
        #     for n in m..nodes[l]:
        #         m.c.add(m..a[(l, n)] == 1 / (1 + pyo.exp(-m..z[(l, n)])))
        #
        # # coect inputs to general model
        # for i in m..nodes[0]:
        #     m.c.add(m.inputs[i] == m..inputs[i])
        #
        # # coect outputs to general model
        # for i in m..nodes[len(self.model.layers) - 1]:
        #     m.c.add(m.outputs[i] == m..outputs[i])

    def _nn_softplus_rule(self, m):
        W = self.model.weights
        b = self.model.biases

        m.layers = list(range(len(self.model.layers)))
        m.nodes = {layer: set(range(nodes)) for layer, nodes in enumerate(self.model.layers)}

        last = len(m.layers) - 1

        m.inputs = pyo.Var(m.nodes[0])
        m.outputs = pyo.Var(m.nodes[len(self.model.layers) - 1])

        m.z = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.a = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.c = pyo.ConstraintList()

        for n in m.nodes[1]:
            m.c.add(m.z[(1, n)] == sum(W[0][n, k] * (m.inputs[k] - self.data.x_train_mean[k]) / self.data.x_train_std[k]
                                       for k in m.nodes[0]) + b[0][n])
            m.c.add(m.a[(1, n)] == pyo.log(1 + pyo.exp(m.z[(1, n)])))

        for l in m.layers[2:]:
            for n in m.nodes[l]:
                m.c.add(m.z[(l, n)] == sum(W[l - 1][n, k] * m.a[(l - 1, k)] for k in m.nodes[l - 1]) + b[l - 1][n])
                m.c.add(m.a[(l, n)] == pyo.log(1 + pyo.exp(m.z[(l, n)])))

        for n in m.nodes[last]:
            m.c.add(m.outputs[n] == m.z[(last, n)] * self.data.y_train_std[n] + self.data.y_train_mean[n])
        # # retrieve general  model
        # m. = pyo.Block(rule=self.__general)
        #
        # # declare variables
        # m.inputs = pyo.Var(m..nodes[0])
        # m.outputs = pyo.Var(m..nodes[len(self.model.layers) - 1])
        #
        # # constraints
        # m.c = pyo.ConstraintList()
        #
        # # activated layers return softplus of linear outputs
        # for l in m..layers[1:]:
        #     for n in m..nodes[l]:
        #         m.c.add(m..a[(l, n)] == pyo.log(1 + pyo.exp(m..z[(l, n)])))
        #
        # # coect inputs to general model
        # for i in m..nodes[0]:
        #     m.c.add(m.inputs[i] == m..inputs[i])
        #
        # # coect outputs to general model
        # for i in m..nodes[len(self.model.layers) - 1]:
        #     m.c.add(m.outputs[i] == m..outputs[i])

    def _nn_relu_rule(self, m):
        W = self.model.weights
        b = self.model.biases

        m.layers = list(range(len(self.model.layers)))
        m.nodes = {layer: set(range(nodes)) for layer, nodes in enumerate(self.model.layers)}

        last = len(m.layers) - 1

        m.inputs = pyo.Var(m.nodes[0])
        m.outputs = pyo.Var(m.nodes[len(self.model.layers) - 1])
        m.y = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]), domain=pyo.Binary)

        m.z = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.a = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.c = pyo.ConstraintList()

        for n in m.nodes[1]:
            m.c.add(m.z[(1, n)] == sum(W[0][n, k] * (m.inputs[k] - self.data.x_train_mean[k]) / self.data.x_train_std[k]
                                       for k in m.nodes[0]) + b[0][n])
            m.c.add(m.a[(1, n)] >= 0)
            m.c.add(m.a[(1, n)] >= m.z[(1, n)])
            m.c.add(m.a[(1, n)] <= 1e6 * m.y[(1, n)])
            m.c.add(m.a[(1, n)] <= m.z[(1, n)] + 1e6 * (1 - m.y[(1, n)]))

        for l in m.layers[2:]:
            for n in m.nodes[l]:
                m.c.add(m.z[(l, n)] == sum(W[l - 1][n, k] * m.a[(l - 1, k)] for k in m.nodes[l - 1]) + b[l - 1][n])
                m.c.add(m.a[(l, n)] >= 0)
                m.c.add(m.a[(l, n)] >= m.z[(l, n)])
                m.c.add(m.a[(l, n)] <= 1e6 * m.y[(l, n)])
                m.c.add(m.a[(l, n)] <= m.z[(l, n)] + 1e6 * (1 - m.y[(l, n)]))

        for n in m.nodes[last]:
            m.c.add(m.outputs[n] == m.z[(last, n)] * self.data.y_train_std[n] + self.data.y_train_mean[n])
        # # retrieve general  model
        # m. = pyo.Block(rule=self.__general)
        #
        # # declare variables
        # m.inputs = pyo.Var(m..nodes[0])
        # m.outputs = pyo.Var(m..nodes[len(self.model.layers) - 1])
        # m.y = pyo.Var(set([(i, j) for i in m..nodes for j in m..nodes[i]]), domain=pyo.Binary)
        #
        # # constraints
        # m.c = pyo.ConstraintList()
        #
        # # activated layers return ReLU of linear outputs, big-M formulation
        # for l in m..layers[1:]:
        #     for n in m..nodes[l]:
        #         m.c.add(m..a[(l, n)] >= 0)
        #         m.c.add(m..a[(l, n)] >= m..z[(l, n)])
        #         m.c.add(m..a[(l, n)] <= 1e6 * m.y[(l, n)])
        #         m.c.add(m..a[(l, n)] <= m..z[(l, n)] + 1e6 * (1 - m.y[(l, n)]))
        #
        # # coect inputs to general model
        # for i in m..nodes[0]:
        #     m.c.add(m.inputs[i] == m..inputs[i])
        #
        # # coect outputs to general model
        # for i in m..nodes[len(self.model.layers) - 1]:
        #     m.c.add(m.outputs[i] == m..outputs[i])

    def _nn_hardsigmoid_rule(self, m):

        W = self.model.weights
        b = self.model.biases

        m.layers = list(range(len(self.model.layers)))
        m.nodes = {layer: set(range(nodes)) for layer, nodes in enumerate(self.model.layers)}

        last = len(m.layers) - 1

        m.inputs = pyo.Var(m.nodes[0])
        m.outputs = pyo.Var(m.nodes[len(self.model.layers) - 1])

        m.z = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.a = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.p = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]), domain=pyo.Binary)
        m.q = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]), domain=pyo.Binary)
        m.c = pyo.ConstraintList()

        for n in m.nodes[1]:
            m.c.add(m.z[(1, n)] == sum(W[0][n, k] * (m.inputs[k] - self.data.x_train_mean[k]) / self.data.x_train_std[k]
                                       for k in m.nodes[0]) + b[0][n])
            m.c.add(m.a[(1, n)] <= m.p[(1, n)])
            m.c.add(
                m.a[(1, n)] >=
                m.z[(1, n)] / 6 + 0.5 - 1e6 * (1 - m.p[(1, n)] + m.q[(1, n)])
            )
            m.c.add(
                m.a[(1, n)] <=
                m.z[(1, n)] / 6 + 0.5 + 1e6 * (1 - m.p[(1, n)] + m.q[(1, n)])
            )
            m.c.add(m.a[(1, n)] >= m.q[(1, n)])
            m.c.add(m.z[(1, n)] - 1e6 * m.p[(1, n)] <= -3)
            m.c.add(m.z[(1, n)] + 1e6 * (1 - m.p[(1, n)]) >= -3)
            m.c.add(m.z[(1, n)] - 1e6 * m.q[(1, n)] <= 3)
            m.c.add(m.z[(1, n)] + 1e6 * (1 - m.q[(1, n)]) >= 3)

        for l in m.layers[2:]:
            for n in m.nodes[l]:
                m.c.add(m.z[(l, n)] == sum(W[l - 1][n, k] * m.a[(l - 1, k)] for k in m.nodes[l - 1]) + b[l - 1][n])
                m.c.add(m.a[(l, n)] <= m.p[(l, n)])
                m.c.add(
                    m.a[(l, n)] >=
                    m.z[(l, n)] / 6 + 0.5 - 1e6 * (1 - m.p[(l, n)] + m.q[(l, n)])
                )
                m.c.add(
                    m.a[(l, n)] <=
                    m.z[(l, n)] / 6 + 0.5 + 1e6 * (1 - m.p[(l, n)] + m.q[(l, n)])
                )
                m.c.add(m.a[(l, n)] >= m.q[(l, n)])
                m.c.add(m.z[(l, n)] - 1e6 * m.p[(l, n)] <= -3)
                m.c.add(m.z[(l, n)] + 1e6 * (1 - m.p[(l, n)]) >= -3)
                m.c.add(m.z[(l, n)] - 1e6 * m.q[(l, n)] <= 3)
                m.c.add(m.z[(l, n)] + 1e6 * (1 - m.q[(l, n)]) >= 3)

        for n in m.nodes[last]:
            m.c.add(m.outputs[n] == m.z[(last, n)] * self.data.y_train_std[n] + self.data.y_train_mean[n])

        # # retrieve general  model
        # m. = pyo.Block(rule=self.__general)
        #
        # # declare variables
        # m.inputs = pyo.Var(m..nodes[0])
        # m.outputs = pyo.Var(m..nodes[len(self.model.layers) - 1])
        # m.p = pyo.Var(set([(i, j) for i in m..nodes for j in m..nodes[i]]), domain=pyo.Binary)
        # m.q = pyo.Var(set([(i, j) for i in m..nodes for j in m..nodes[i]]), domain=pyo.Binary)
        #
        # # constraints
        # m.c = pyo.ConstraintList()
        #
        # # activated layers return HardSigmoid of linear outputs, big-M formulation
        # for l in m..layers[1:]:
        #     for n in m..nodes[l]:
        #         m.c.add(m.nn.a[(l, n)] <= m.p[(l, n)])
        #         m.c.add(
        #             m.nn.a[(l, n)] >=
        #             m.nn.z[(l, n)] / 6 + 0.5 - 1e6 * (1 - m.p[(l, n)] + m.q[(l, n)])
        #         )
        #         m.c.add(
        #             m.nn.a[(l, n)] <=
        #             m.nn.z[(l, n)] / 6 + 0.5 + 1e6 * (1 - m.p[(l, n)] + m.q[(l, n)])
        #         )
        #         m.c.add(m.nn.a[(l, n)] >= m.q[(l, n)])
        #         m.c.add(m.nn.z[(l, n)] - 1e6 * m.p[(l, n)] <= -3)
        #         m.c.add(m.nn.z[(l, n)] + 1e6 * (1 - m.p[(l, n)]) >= -3)
        #         m.c.add(m.nn.z[(l, n)] - 1e6 * m.q[(l, n)] <= 3)
        #         m.c.add(m.nn.z[(l, n)] + 1e6 * (1 - m.q[(l, n)]) >= 3)
        #
        # # coect inputs to general model
        # for i in m..nodes[0]:
        #     m.c.add(m.inputs[i] == m..inputs[i])
        #
        # # coect outputs to general model
        # for i in m..nodes[len(self.model.layers) - 1]:
        #     m.c.add(m.outputs[i] == m..outputs[i])

    def _nn_leakyrelu_rule(self, m):
        W = self.model.weights
        b = self.model.biases

        m.layers = list(range(len(self.model.layers)))
        m.nodes = {layer: set(range(nodes)) for layer, nodes in enumerate(self.model.layers)}

        last = len(m.layers) - 1

        m.inputs = pyo.Var(m.nodes[0])
        m.outputs = pyo.Var(m.nodes[len(self.model.layers) - 1])
        m.y = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]), domain=pyo.Binary)

        m.z = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.a = pyo.Var(set([(i, j) for i in m.nodes for j in m.nodes[i]]))
        m.c = pyo.ConstraintList()

        for n in m.nodes[1]:
            m.c.add(m.z[(1, n)] == sum(W[0][n, k] * (m.inputs[k] - self.data.x_train_mean[k]) / self.data.x_train_std[k]
                                       for k in m.nodes[0]) + b[0][n])
            m.c.add(m.a[(1, n)] >= 1e-2 * m.z[(1, n)])
            m.c.add(m.a[(1, n)] >= m.z[(1, n)])
            m.c.add(m.a[(1, n)] <= m.z[(1, n)] + 1e6 * (1 - m.y[(1, n)]))
            m.c.add(m.a[(1, n)] <= 1e-2 * m.z[(1, n)] + 1e6 * m.y[(1, n)])

        for l in m.layers[2:]:
            for n in m.nodes[l]:
                m.c.add(m.z[(l, n)] == sum(W[l - 1][n, k] * m.a[(l - 1, k)] for k in m.nodes[l - 1]) + b[l - 1][n])
                m.c.add(m.a[(l, n)] >= 1e-2 * m.z[(l, n)])
                m.c.add(m.a[(l, n)] >= m.z[(l, n)])
                m.c.add(m.a[(l, n)] <= m.z[(l, n)] + 1e6 * (1 - m.y[(l, n)]))
                m.c.add(m.a[(l, n)] <= 1e-2 * m.z[(l, n)] + 1e6 * m.y[(l, n)])

        for n in m.nodes[last]:
            m.c.add(m.outputs[n] == m.z[(last, n)] * self.data.y_train_std[n] + self.data.y_train_mean[n])

    def _hybrid_rule(self, m):

        if self.model.activation == 'relu':
            m.nn = pyo.Block(rule=self._nn_relu_rule)
        elif self.model.activation == 'tanh':
            m.nn = pyo.Block(rule=self._nn_tanh_rule)
        elif self.model.activation == 'softplus':
            m.nn = pyo.Block(rule=self._nn_softplus_rule)
        elif self.model.activation == 'sigmoid':
            m.nn = pyo.Block(rule=self._nn_sigmoid_rule)
        elif self.model.activation == 'hardsigmoid':
            m.nn = pyo.Block(rule=self._nn_hardsigmoid_rule)
        elif self.model.activation == 'linear':
            m.nn = pyo.Block(rule=self._nn_linear_rule)
        elif self.model.activation == 'leakyrelu':
            m.nn = pyo.Block(rule=self._nn_leakyrelu_rule)

        m.inputs = pyo.Var(m.nn.nodes[0])
        m.outputs = pyo.Var(set(range(1)))
        m.feature_extractor = pyo.Var(m.nn.nodes[len(self.model.layers) - 1])
        m.c = pyo.ConstraintList()
        for i in m.nn.nodes[0]:
            m.c.add(m.inputs[i] == m.nn.inputs[i])

        for i in m.nn.nodes[len(self.model.layers) - 1]:
            m.c.add(m.feature_extractor[i] == m.nn.z[(len(self.model.layers) - 1, i)])

        output_scale = self.model.output_scale
        alpha = self.model.alpha
        # features = self.model.features
        x_train = self.model.x_train.numpy()
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))
        prediction = None
        if self.model.kernel == 'rbf':
            length_scale = self.model.length_scale
            prediction = sum(alpha[i] * output_scale * pyo.exp(
                -sum(
                    0.5 / length_scale ** 2 * (
                        m.feature_extractor[j] - x_train[i, j]) ** 2 for j in n_inputs)
            ) for i in n_samples)

        prediction = prediction * self.data.y_train_std[0] + self.data.y_train_mean[0]

        # gpr constraint
        m.c.add(m.outputs[0] == prediction)
