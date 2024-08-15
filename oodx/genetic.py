# -- coding: utf-8 --
from pymoo.core.problem import ElementwiseProblem
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
import matplotlib.pyplot as plt


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data = []

    def notify(self, algorithm):
        # 记录每一代的最优目标值
        best_F = algorithm.pop.get("F").min()
        self.data.append(best_F)
        print(f"Generation: {algorithm.n_gen}, Best F: {best_F}")


class MyProblem(ElementwiseProblem):

    def __init__(self, trained_model, n_var, xl, xu, data):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=xl, xu=xu)
        self.model = trained_model
        self.data = data

    def _evaluate(self, x, out, *args, **kwargs):
        # 将输入变量转化为 GPR 模型可接受的格式
        x = np.array(x).reshape(1, -1)
        x = self.data.scale_x(x)
        # 使用 GPR 模型进行预测
        prediction = self.data.inv_scale_y(self.model.predict(x))
        # 返回预测结果作为目标值
        out["F"] = - prediction[0, 0]


class Genetic:

    def __init__(self, trained_model, n_var, xl, xu, pop_size, generations, data):
        self.problem = MyProblem(trained_model, n_var, xl, xu, data)
        self.pop_size = pop_size
        self.generations = generations

    def solve(self):
        callback = MyCallback()

        res = minimize(
            self.problem,
            NSGA2(pop_size=self.pop_size),
            ('n_gen', self.generations),
            seed=1,
            verbose=True,
            callback=callback
        )

        plt.plot(callback.data)
        plt.xlabel('Generation')
        plt.ylabel('Best F')
        plt.title('Convergence Plot')
        plt.show()

        return res.X, res.F

