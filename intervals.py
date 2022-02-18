import numpy as np
from scipy import stats
from utils import get_E, get_second_derivative_matrix
import scipy


def CI_intervals(model, X, X_train, Y_train, alpha, D, covinv):
    CI = np.zeros((len(X), 2))
    predictions = model.predict(X_train)
    mu_hats = predictions[:, 0]
    start_values = model.predict(X)[:, 0]
    sigma_hats = np.exp(predictions[:, 1]) + 1e-3
    for j, x in enumerate(X):
        l, u = CI_x(x, X_train, Y_train, mu_hats, sigma_hats, start_values[j], alpha, D, covinv)
        CI[j, 0] = l
        CI[j, 1] = u
    return CI

def CI_intervals_2(model, X, X_train, Y_train, alpha, rho):
    CI = np.zeros((len(X), 2))
    predictions = model.predict(X_train)
    mu_hats = predictions[:, 0]
    start_values = model.predict(X)[:, 0]
    sigma_hats = np.exp(predictions[:, 1]) + 1e-3
    for j, x in enumerate(X):
        A = get_second_derivative_matrix(model, x)
        E = get_E(A, rho)
        l, u = CI_x_2(x, X_train, Y_train, mu_hats, sigma_hats, start_values[j], alpha, E)
        CI[j, 0] = l
        CI[j, 1] = u
    return CI


def CI_x(x_0, X_train, Y_train, mu_hats, sigma_hats, start_value, alpha, D, covinv, stepsize=0.025):
    upperbound = start_value
    accepting = True
    n = 1
    while accepting:
        accepting = accept(X_train, Y_train, mu_hats, sigma_hats, x_0, stepsize * n, alpha, D, covinv, positive=1)
        if accepting:
            upperbound += stepsize
            n += 1
    accepting = True
    lowerbound = start_value
    n = 1
    while accepting:
        accepting = accept(X_train, Y_train, mu_hats, sigma_hats, x_0, stepsize * n, alpha, D, covinv, positive=0)
        if accepting:
            lowerbound -= stepsize
            n += 1
    return lowerbound, upperbound

def CI_x_2(x_0, X_train, Y_train, mu_hats, sigma_hats, start_value, alpha, E, stepsize=0.025):
    upperbound = start_value
    accepting = True
    n = 1
    while accepting:
        accepting = accept_2(X_train, Y_train, mu_hats, sigma_hats, x_0, stepsize * n, alpha, E=E, positive=1)
        if accepting:
            upperbound += stepsize
            n += 1
    accepting = True
    lowerbound = start_value
    n = 1
    while accepting:
        accepting = accept_2(X_train, Y_train, mu_hats, sigma_hats, x_0, stepsize * n, alpha, E=E, positive=0)
        if accepting:
            lowerbound -= stepsize
            n += 1
    return lowerbound, upperbound


def accept(X, Y, mu_hats, sigma_hats, x_0, y_0, alpha, D, covinv=None, positive=True):
    g = get_perturbation(x_0, y_0, D, covinv)
    if positive:
        mu_0 = mu_hats + np.array(list(map(g, X)))
    if not positive:
        mu_0 = mu_hats - np.array(list(map(g, X)))
    ratio = LR(Y, mu_0, mu_hats, sigma_hats)
  #  q = scipy.stats.chi2(1).ppf(1 - alpha / 2)
    C = get_C(sigma_hats, g, X)
    critical_value = scipy.stats.norm(loc=-C, scale=2*np.sqrt(C)).ppf(1 - alpha)
    # print(f'q={q}, ratio={ratio}')
    if ratio > critical_value:
        return 0
    else:
        return 1

def accept_2(X, Y, mu_hats, sigma_hats, x_0, y_0, alpha, E, positive=True):
    g = get_perturbation_2(x_0, y_0, E)
    if positive:
        mu_0 = mu_hats + np.array(list(map(g, X)))
    if not positive:
        mu_0 = mu_hats - np.array(list(map(g, X)))
    ratio = LR(Y, mu_0, mu_hats, sigma_hats)
    C = get_C(sigma_hats, g, X)
    critical_value = scipy.stats.norm(loc=-C, scale=2*np.sqrt(C)).ppf(1 - alpha) # todo: Think about alpha or alpha/2
    if ratio > critical_value:
        return 0
    else:
        return 1


def get_perturbation(x_0, y_0, D, convinv=None):
    shape_x_0 = np.shape(x_0)
    def g(x):
        assert np.shape(x) == shape_x_0
        d = np.sqrt(np.abs(y_0) / D)
        z = np.sqrt(np.sum((x - x_0) ** 2))
        if convinv is not None:
            z = np.sqrt(np.transpose((x - x_0) @ convinv @ (x-x_0)))
        if z < d:
            return y_0 - D / 2 * z**2
        elif d < z < (2 * d):
            return D / 2 * (z - 2 * d)**2
        else:
            return 0
    return g


def get_perturbation_2(x_0, y_0, E):
    shape_x_0 = np.shape(x_0)
    def g(x):
        assert np.shape(x) == shape_x_0
        ksi = np.sqrt(np.abs(y_0))
        z = np.sqrt(np.transpose(x - x_0) @ E @ (x-x_0))
        if z < ksi:
            return y_0 - 1 / 2 * z**2
        elif ksi < z < (2 * ksi):
            return 1 / 2 * (z - 2 * ksi)**2
        else:
            return 0
    return g



def get_C(sigma, g, X):
    C = np.sum(np.array(list(map(g, X)))**2 / sigma**2) + 1e-3
    return C


def LR(y, mu_0, mu_hat, sigma):
    return 2 * (- 0.5 * np.sum(((y - mu_hat) / sigma)**2) + 0.5 * np.sum(((y - mu_0) / sigma)**2))

#

# S = np.linalg.inv(np.array([[1, 0.5],[0.5, 1]]))
# b = get_perturbation(np.array([0,0]), 1, 3, S)
# x = np.linspace(-2, 2, 40)
# y = np.linspace(-2,2, 40)
# z = np.zeros((len(x), len(x)))
# for i, xi in enumerate(x):
#     for j, yj in enumerate(y):
#         z[i, j] = b([xi,yj])
# plt.contour(x, y, z)
# plt.show()