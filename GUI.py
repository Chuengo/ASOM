# -- coding: utf-8 --
import ast
import os
import sys
import math
import re
import numpy as np
import gpytorch as gpy
import time
import tempfile
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier

os.environ['QT_PLUGIN_PATH'] = r'F:\anaconda3\envs\new_omo\Lib\site-packages\PyQt6\Qt6\plugins\platforms'
from PySide6.QtGui import QGuiApplication
from PySide6.QtCore import QFile, QIODevice, Slot, QThread, Signal
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import *
from SBO_GUI import Ui_SBO
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from oodx import DataHandler, GPR, GPC, NN, HybridModel, OODXBlock, Genetic
from sklearn.metrics import *
import mplcursors
import pyomo.environ as pyo


class PlotDialog(QDialog):
    def __init__(self, func, name, y_test, y_test_predict, y_train, y_train_predict):
        super().__init__()
        self.name = name
        self.y_test = y_test
        self.y_test_predict = y_test_predict
        self.y_train = y_train
        self.y_train_predict = y_train_predict
        self.setWindowTitle(f"{self.name} Performance")
        layout = QGridLayout()
        self.canvas1 = FigureCanvas(Figure())
        self.canvas2 = FigureCanvas(Figure())
        layout.addWidget(self.canvas1, 0, 0)
        layout.addWidget(self.canvas2, 0, 1)
        self.setLayout(layout)
        if func == 'regression':
            self.plot_regression()
        else:
            self.plot_classification()

    def plot_regression(self):
        ax1 = self.canvas1.figure.add_subplot(111)
        ax1.clear()

        min_val = math.floor(min(np.min(self.y_test), np.min(self.y_test_predict)).item())
        max_val = math.ceil(max(np.max(self.y_test), np.max(self.y_test_predict)).item())
        ax1.set_xlim(min_val, max_val)
        ax1.set_ylim(min_val, max_val)

        ax1.scatter(self.y_test, self.y_test_predict, c='r', label='test data', s=1)
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', label="y=x")
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Scatter Plot of Test Data')

        step_size = 0.5
        x_ticks = np.arange(min_val, max_val, step_size)
        y_ticks = np.arange(min_val, max_val, step_size)
        ax1.set_xticks(x_ticks)
        ax1.set_yticks(y_ticks)
        ax1.legend()

        ax2 = self.canvas2.figure.add_subplot(111)
        ax2.clear()
        mse_train = mean_squared_error(self.y_train, self.y_train_predict)
        mse_test = mean_squared_error(self.y_test, self.y_test_predict)

        bars = ax2.bar(['Train MSE', 'Test MSE'], [mse_train, mse_test], color=['blue', 'orange'])
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('MSE')
        ax2.set_title('MSE Comparison: Train vs Test')
        cursor = mplcursors.cursor(bars, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(f'MSE: {sel.target[1]:.4f}')

        self.canvas1.draw()
        self.canvas2.draw()

    def plot_classification(self):
        y_predict_binary = (self.y_test_predict > 0.5).astype(int)
        cm = confusion_matrix(self.y_test, y_predict_binary, labels=[0.0, 1.0])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        ax1 = self.canvas1.figure.add_subplot(111)
        ax1.clear()
        disp.plot(ax=ax1)
        ax1.set_title('Confusion Matrix')

        log_loss_train = log_loss(self.y_train, self.y_train_predict, labels=[0.0, 1.0])
        log_loss_test = log_loss(self.y_test, self.y_test_predict, labels=[0.0, 1.0])

        self.canvas1.draw()
        ax2 = self.canvas2.figure.add_subplot(111)
        ax2.clear()
        bars = ax2.bar(['Train Log Loss', 'Test Log Loss'], [log_loss_train, log_loss_test], color=['blue', 'orange'])
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Log Loss')
        ax2.set_title('Log Loss Comparison: Train vs Test')

        cursor = mplcursors.cursor(bars, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(f'Log Loss: {sel.target[1]:.4f}')

        self.canvas2.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load UI
        self.ui = Ui_SBO()
        self.ui.setupUi(self)
        self.data = DataHandler()
        self.model = 'Gaussian Process(Regression)'
        self.activation = 'tanh'
        self.kernel = 'rbf'
        self.hb_activation = 'tanh'
        self.hb_kernel = 'rbf'
        self.is_regression = True
        self.dglayer = None
        self.input_for_prediction = None
        self.opt_method = 'Genetic Algorithm'
        self.n_individuals = None
        self.n_generations = None
        self.selected_dimension = None
        self.trained_model = None
        self.prediction = None
        self.nn_layers = None
        self.batch_size = 10
        self.epochs = 100
        self.solver = 'BARON'

        # Dataset Upload
        self.ui.pushButton_Input_Data.clicked.connect(self.upload_input)
        self.ui.pushButton_Space.clicked.connect(self.upload_space)
        self.ui.pushButton_Output_Data.clicked.connect(self.upload_output)
        self.ui.pushButto_Process.clicked.connect(self.upload_process)
        self.ui.pushButton_Scale.clicked.connect(self.data_preprocess)
        self.ui.pushButton_Performance.clicked.connect(self.show_performance)

        # Model Choosing and Training
        self.ui.comboBox_Model.currentIndexChanged.connect(self.model_update)
        self.ui.comboBox_Kernel.currentIndexChanged.connect(self.kernel_update)
        self.ui.comboBox_HB_Gaukernel.currentIndexChanged.connect(self.HB_kernel_update)
        self.ui.comboBox_HB_NNactivate.currentIndexChanged.connect(self.HB_activation_update)
        self.ui.comboBox_Activation.currentIndexChanged.connect(self.activation_update)
        self.ui.pushButton_Fit.clicked.connect(self.model_fit)

        # Predict
        self.ui.pushButton_Predict.clicked.connect(self.model_predict)

        # Optimisation
        self.ui.comboBox_Optimise_Method.currentIndexChanged.connect(self.optimisation_method_update)
        self.ui.pushButton_Optimise.clicked.connect(self.model_optimise)
        self.ui.comboBox_Solvers.currentIndexChanged.connect(self.solver_update)

    def upload_input(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input File", "",
                                                   "All Files (*);;Text Files (*.txt);;CSV Files (*.csv)",
                                                   options=options)
        if file_path:
            self.ui.textEdit_Input_Data.setText(os.path.basename(file_path))
            self.data.x = np.loadtxt(file_path)

    def upload_space(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Space File", "",
                                                   "All Files (*);;Text Files (*.txt);;CSV Files (*.csv)",
                                                   options=options)
        space = []
        pattern = re.compile(r'\((.*?)\)')
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                match = pattern.search(line)
                if match:
                    tuple_str = match.group(1)
                    tuple_vals = tuple(map(float, tuple_str.split(',')))
                    space.append(tuple_vals)
        self.data.space = space
        if file_path:
            self.ui.textEdit_Space.setText(os.path.basename(file_path))

    def upload_output(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Output File", "",
                                                   "All Files (*);;Text Files (*.txt);;CSV Files (*.csv)",
                                                   options=options)
        if file_path:
            self.ui.textEdit_Output_Data.setText(os.path.basename(file_path))
            self.data.y = np.loadtxt(file_path)

    def upload_process(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Convergence File", "",
                                                   "All Files (*);;Text Files (*.txt);;CSV Files (*.csv)",
                                                   options=options)
        if file_path:
            self.ui.textEdit_Process.setText(os.path.basename(file_path))
            self.data.t = np.loadtxt(file_path)

    def data_preprocess(self):
        if self.data.t is None:
            self.data.t = np.ones((self.data.y.shape[0], 1))
        elif self.data.y is None:
            self.data.y = np.ones((self.data.t.shape[0], 1))
        self.data.split(test_size=0.2)
        self.data.scale()
        self.ui.textEdit_Results.append("Data is ready!\n")

    def model_update(self):
        self.model = self.ui.comboBox_Model.currentText()
        if self.model in ["Gaussian Process(Regression)", "Neural Network(Regression)", "Hybrid(Regression)"]:
            self.is_regression = True
        else:
            self.is_regression = False

    def kernel_update(self):
        self.kernel = self.ui.comboBox_Kernel.currentText()

    def activation_update(self):
        self.activation = self.ui.comboBox_Activation.currentText()

    def HB_kernel_update(self):
        self.hb_kernel = self.ui.comboBox_HB_Gaukernel.currentText()

    def HB_activation_update(self):
        self.hb_activation = self.ui.comboBox_HB_NNactivate.currentText()

    def solver_update(self):
        self.solver = self.ui.comboBox_Solvers.currentText()

    def optimisation_method_update(self):
        self.opt_method = self.ui.comboBox_Optimise_Method.currentText()

    def model_fit(self):
        self.ui.textEdit_Results.append("Model fitting......\n")
        QApplication.processEvents()

        if self.model == "Gaussian Process(Regression)":
            gpr = GPR(kernel=self.kernel)
            gpr.fit(self.data.x_train_, self.data.y_train_[:, 0], iprint=True)
            self.trained_model = gpr
            self.ui.textEdit_Results.append(
                '{} model fitted! Time elapsed {:.5f} s'.format(self.trained_model.name, self.trained_model.time))
            kernel_params = self.trained_model.kernel_.get_params()
            self.ui.textEdit_Results.append("Kernel Parameters:")
            for param, value in kernel_params.items():
                self.ui.textEdit_Results.append(f"{param}: {value}")
            self.ui.textEdit_Results.append("\n")

        elif self.model == "Gaussian Process(Classification)":
            gpc = GPC()
            gpc.fit(self.data.x_train_, self.data.t_train)
            self.trained_model = gpc
            self.ui.textEdit_Results.append(
                '{} model fitted! Time elapsed {:.5f} s'.format(self.trained_model.name, self.trained_model.time))
            self.ui.textEdit_Results.append("Kernel Parameters:")
            self.ui.textEdit_Results.append(f"Kernel:RBF")
            self.ui.textEdit_Results.append(f"Constant:{self.trained_model.sigma_f ** 2}")
            self.ui.textEdit_Results.append(f"Length-scale:{self.trained_model.l}")

        elif self.model in ("Neural Network(Regression)", "Neural Network(Classification)"):
            layers = self.ui.lineEdit_Nodes.text()
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', layers)
            layers = np.array([int(num) for num in numbers])
            batch_size = int(self.ui.lineEdit_Batch_Size.text())
            epochs = int(self.ui.lineEdit_Epochs.text())
            learning_rate = float(self.ui.lineEdit_Learning_Rate.text())
            decay = float(self.ui.lineEdit_Weight_Decay.text())

            def callback(epoch, loss):
                self.ui.textEdit_Results.append(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}\n")
                QApplication.processEvents()

            if self.model == "Neural Network(Regression)":
                nn = NN(layers, activation=self.activation)
                nn.fit(self.data.x_train_, self.data.y_train_[:, 0], callback=callback, batch_size=batch_size,
                       learning_rate=learning_rate, epochs=epochs, weight_decay=decay)
            else:
                nn = NN(layers, self.activation, is_classifier=True)
                nn.fit(self.data.x_train_, self.data.t_train,
                       callback=callback, batch_size=batch_size, learning_rate=learning_rate,
                       epochs=epochs, weight_decay=decay)
            self.trained_model = nn
            self.ui.textEdit_Results.append(
                '{} model fitted! Time elapsed {:.5f} s'.format(self.trained_model.name, self.trained_model.time))

        elif self.model == "Hybrid(Regression)":
            layers = self.ui.lineEdit_HB_NNnodes.text()
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', layers)
            layers = np.array([int(num) for num in numbers])
            batch_size = int(self.ui.lineEdit_HB_NNbatchSize.text())
            epochs = int(self.ui.lineEdit_HB_NNepochs.text())
            learning_rate = float(self.ui.lineEdit_HB_NNlearnrate.text())
            decay = float(self.ui.lineEdit_HB_NNweightdecay.text())

            def callback(epoch, loss):
                self.ui.textEdit_Results.append(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}\n")
                QApplication.processEvents()

            hb_model = HybridModel(self.data.x_train_, self.data.y_train_[:, 0], gpy.likelihoods.GaussianLikelihood(),
                                   layers, self.hb_activation, self.hb_kernel)

            hb_model.fit(callback=callback, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs,
                         weight_decay=decay)
            self.trained_model = hb_model
            self.ui.textEdit_Results.append(
                '{} model fitted! Time elapsed {:.5f} s'.format(self.trained_model.name, self.trained_model.time))

    def model_predict(self):
        text = self.ui.textEdit_Input_Predict.toPlainText()
        try:
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text)
            array_for_pre = np.array([float(num) for num in numbers]).reshape(-1, self.data.x.shape[1])
            self.input_for_prediction = self.data.scale_x(array_for_pre)
            self.prediction = self.trained_model.predict(self.input_for_prediction)
            if self.is_regression:
                self.prediction = self.data.inv_scale_y(self.prediction)
            self.ui.textEdit_Results.append(f"The Prediction for the Input is: {self.prediction}\n")
        except Exception as e:
            self.ui.textEdit_Results.append(f"Invalid input. Please enter a valid list format. Error: {e}\n")

    def model_optimise(self):
        if self.opt_method == "Surrogate Model":
            omo = pyo.ConcreteModel()
            omo.n_inputs = set(range(len(self.data.space)))
            omo.inputs = pyo.Var(omo.n_inputs, bounds=self.data.space)
            omo.output = pyo.Var()
            omo.obj = pyo.Objective(expr=omo.output, sense=pyo.maximize)
            omo.block = OODXBlock(self.trained_model, self.data).get_formulation()
            omo.c = pyo.ConstraintList()
            omo.c.add(omo.output == omo.block.outputs[0])
            for i in omo.n_inputs:
                omo.c.add(omo.inputs[i] == omo.block.inputs[i])

            # solve
            solver = None
            if self.solver == "BARON":
                solver = pyo.SolverFactory('BARON')
            elif self.solver == "ipopt":
                solver = pyo.SolverFactory('ipopt')
            elif self.solver == "bonmin":
                solver = pyo.SolverFactory('bonmin')
            elif self.solver == "Couenne":
                solver = pyo.SolverFactory('couenne')

            self.ui.textEdit_Results.append(f"{self.solver} is solving the problem\n")
            QApplication.processEvents()
            st_time = time.time()
            results = solver.solve(omo, tee=True)
            ed_time = time.time()
            print("Solver Status:", results.solver.status)
            print("Soling time:", ed_time-st_time)
            print("Solver Termination Condition:", results.solver.termination_condition)
            for i in omo.n_inputs:
                self.ui.textEdit_Results.append(f"Optimal solution of x[{i}]: {pyo.value(omo.inputs[i])}")
            self.ui.textEdit_Results.append(f"Optimal value:{pyo.value(omo.output)}\n")

        else:
            pop_size = int(self.ui.lineEdit_Num_Individuals.text())
            generations = int(self.ui.lineEdit_Num_Generations.text())
            n_var = len(self.data.space)
            xl = [t[0] for t in self.data.space]
            xu = [t[1] for t in self.data.space]
            omo = Genetic(self.trained_model, n_var, xl, xu, pop_size, generations, self.data)
            st_time = time.time()
            self.ui.textEdit_Results.append(f"genetic algorithm is solving the problem\n")
            x, y = omo.solve()
            ed_time = time.time()
            print("Soling time:", ed_time - st_time)
            if len(y) == 1:
                solution_0 = x[0]
                solution_1 = x[1]
                value = -y[0]
                self.ui.textEdit_Results.append(f"Optimal solution: [{solution_0},{solution_1}]")
                self.ui.textEdit_Results.append(f"Optimal value: {value}\n")
            else:
                for i in range(len(x)):
                    solution = x[i]
                    value = -y[i]
                    self.ui.textEdit_Results.append(f"Optimal solution {i + 1}: {solution}")
                    self.ui.textEdit_Results.append(f"Optimal value {i + 1}: {value}\n")

    @Slot()
    def show_performance(self):
        test_predict = self.trained_model.predict(self.data.x_test_)

        print(self.data.x_test)

        print(test_predict)

        train_predict = self.trained_model.predict(self.data.x_train_)

        if self.is_regression:
            dialog = PlotDialog(func='regression', name=self.model, y_test=self.data.y_test_[:, 0],
                                y_test_predict=test_predict, y_train=self.data.y_train_[:, 0],
                                y_train_predict=train_predict)
        else:
            dialog = PlotDialog(func='classification', name=self.model, y_test=self.data.t_test,
                                y_test_predict=test_predict, y_train=self.data.t_train, y_train_predict=train_predict)

        dialog.exec()


if __name__ == "__main__":
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec()
