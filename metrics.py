import numpy as np
from second_derivative_utils import CI_coverage_probability

class IntervalMetrics():
    def __init__(self, N_simulations, alphas, N_test):
        self.alphas = alphas
        self.CI_correct = np.zeros((N_simulations, len(alphas), N_test))
        self.CI_width = np.zeros((N_simulations, len(alphas), N_test))
        self.PI_correct = np.zeros((N_simulations, len(alphas), N_test))
        self.PI_width = np.zeros((N_simulations, len(alphas), N_test))
        self.bias = 1e3 * np.ones((N_simulations, N_test))
        self.error = 1e3 * np.ones((N_simulations, N_test))

    def update_CI(self, CI, f_test, simulation, alpha):
        self.CI_correct[simulation, alpha] = CI_coverage_probability(CI, f_test)
        self.CI_width[simulation, alpha] = (CI[:, 1] - CI[:, 0])


    def update_PI(self, PI, f_test, var_true, distribution, simulation, alpha):
        self.PI_correct[simulation, alpha] = PI_coverage_probability(PI, f_test, var_true, distribution)
        self.PI_width[simulation, alpha] = (PI[:, 1] - PI[:, 0])

    def Brier_score(self, type):
        if type == 'PI':
            BS = []
            for i, alpha in enumerate(self.alphas):
                BS.append(np.mean((np.mean(self.PI_correct, axis=0)[i] - (1 - self.alphas[i])) ** 2))
            return BS
        if type == 'CI':
            BS = []
            for i, alpha in enumerate(self.alphas):
                BS.append(np.mean((np.mean(self.CI_correct, axis=0)[i] - (1 - self.alphas[i])) ** 2))
            return BS

    def update_bias(self, prediction, f_test, simulation):
        self.bias[simulation] = prediction - f_test

    def update_error(self, prediction, Y_test, simulation):
        self.error[simulation] = prediction - Y_test

    def RMSE(self):
        return np.sqrt(np.mean((self.error)**2))


