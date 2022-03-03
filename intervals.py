import numpy as np
from scipy import stats
from second_derivative_utils import get_E, get_second_derivative_matrix
import scipy
from copy import deepcopy
import time


def timer(func):
    """A timer decorator to quickly check performance."""
    def wrapper(*args, **kwargs):
        begin_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - begin_time
        print(f'finished {func.__name__} in {duration} seconds')
        return value
    return wrapper



def CI_intervals(model, X, X_train, Y_train, alpha, rho):
    CI = np.zeros((len(X), 2))
    predictions = model.predict(X_train)
    mu_hats = predictions[:, 0]
    start_values = model.predict(X)[:, 0]
    sigma_hats = np.exp(predictions[:, 1]) + 1e-3
    Z_values = (mu_hats - Y_train) / sigma_hats
    for j, x in enumerate(X):
        A = get_second_derivative_matrix(model, x)
        E = get_E(A, rho)
        l, u = CI_x(x, X_train, Z_values, sigma_hats, start_values[j], alpha, E)
        CI[j, 0] = l
        CI[j, 1] = u
    return CI



def CI_x(x_0, X_train, Z, sigma_hats, start_value, alpha, E):
    perturbation_1 = golden_search(X_train, Z, sigma_hats, x_0, alpha, right_bound=0.5, E=E, positive=1, tollerance=0.01)
    perturbation_2 = golden_search(X_train, Z, sigma_hats, x_0, alpha, right_bound=0.5, E=E, positive=0, tollerance=0.01)
    upperbound = start_value + perturbation_1
    lowerbound = start_value - perturbation_2
    return lowerbound, upperbound



def golden_search(X_train, Z, sigma_hats, x_0, alpha, E, right_bound=0.5, positive=1, tollerance=0.01):
    accepting = accept(X_train, Z, sigma_hats, x_0, right_bound, alpha, E=E, positive=positive)
    n = 0
    left_bound = 0
    while accepting: # If the start_value gets accepted, we need to move the upper bound.
        n += 1
        left_bound = deepcopy(right_bound)  # We already know that we do not need to look left of the current right_bound
        right_bound *= 2
        accepting = accept(X_train, Z, sigma_hats, x_0, right_bound, alpha, E=E, positive=positive)
    difference = 2 * tollerance
    while difference > tollerance:
        n += 1
        difference = 0.62 * (right_bound - left_bound)
        new_bound = left_bound + difference
        if accept(X_train, Z, sigma_hats, x_0, new_bound, alpha, E=E, positive=positive):
            left_bound = new_bound  # If the new bound gets accepted, this is the new lowerbound
        else:
            right_bound = new_bound
    return (right_bound + left_bound) / 2




def accept(X_train, Z, sigma_hats, x_0, delta, alpha, E, positive=True):
    g = get_perturbation_function(x_0, delta, E)
    C_values = np.array(list(map(g, X_train))) / sigma_hats
    if positive:
        constant = np.sum(C_values ** 2)
        T = np.sum(C_values * Z) + constant
    if not positive:
        constant = np.sum(C_values ** 2)
        T = -np.sum(C_values * Z) + constant
    assert np.shape(C_values) == np.shape(Z)
    if constant == 0.0:
        return 1
    critical_value = scipy.stats.norm(loc=0, scale=np.sqrt(constant)).ppf(1 - alpha)
    if T > critical_value:
        return 0
    else:
        return 1


def get_perturbation_function(x_0, delta, E):
    shape_x_0 = np.shape(x_0)
    if shape_x_0 is ():
        x_0 = np.array([x_0])
        shape_x_0 = np.shape(x_0)
    def g(x):
        if np.shape(x) is ():
            x = np.array([x])
        assert np.shape(x) == shape_x_0
        ksi = np.sqrt(np.abs(delta))
        z = np.sqrt(np.transpose(x - x_0) @ E @ (x-x_0))
        if z < ksi:
            return delta - 1 / 2 * z ** 2
        elif ksi < z < (2 * ksi):
            return 1 / 2 * (z - 2 * ksi)**2
        else:
            return 0
    return g

#
# def get_C(sigma, g, X):
#     C = np.sum(np.array(list(map(g, X)))**2 / sigma**2) + 1e-9
#     return C
#
#
# def LR(y, mu_0, mu_hat, sigma):
#     return 2 * (- 0.5 * np.sum(((y - mu_hat) / sigma)**2) + 0.5 * np.sum(((y - mu_0) / sigma)**2))


# def CI_x(x_0, X_train, Y_train, mu_hats, sigma_hats, start_value, alpha, D, covinv, stepsize=0.025):
#     upperbound = start_value
#     accepting = True
#     n = 1
#     while accepting:
#         accepting = accept(X_train, Y_train, mu_hats, sigma_hats, x_0, stepsize * n, alpha, D, covinv, positive=1)
#         if accepting:
#             upperbound += stepsize
#             n += 1
#     accepting = True
#     lowerbound = start_value
#     n = 1
#     while accepting:
#         accepting = accept(X_train, Y_train, mu_hats, sigma_hats, x_0, stepsize * n, alpha, D, covinv, positive=0)
#         if accepting:
#             lowerbound -= stepsize
#             n += 1
#     return lowerbound, upperbound

# def CI_x_2(x_0, X_train, Y_train, mu_hats, sigma_hats, start_value, alpha, E, stepsize=0.025):
#     upperbound = start_value
#     accepting = True
#     n = 1
#     while accepting:
#         accepting = accept_2(X_train, Y_train, mu_hats, sigma_hats, x_0, stepsize * n, alpha, E=E, positive=1)
#         if accepting:
#             upperbound += stepsize
#             n += 1
#     accepting = True
#     lowerbound = start_value
#     n = 1
#     while accepting:
#         accepting = accept_2(X_train, Y_train, mu_hats, sigma_hats, x_0, stepsize * n, alpha, E=E, positive=0)
#         if accepting:
#             lowerbound -= stepsize
#             n += 1
#     return lowerbound, upperbound

# def get_perturbation(x_0, y_0, D, convinv=None):
#     shape_x_0 = np.shape(x_0)
#     def g(x):
#         assert np.shape(x) == shape_x_0
#         d = np.sqrt(np.abs(y_0) / D)
#         z = np.sqrt(np.sum((x - x_0) ** 2))
#         if convinv is not None:
#             z = np.sqrt(np.transpose((x - x_0) @ convinv @ (x-x_0)))
#         if z < d:
#             return y_0 - D / 2 * z**2
#         elif d < z < (2 * d):
#             return D / 2 * (z - 2 * d)**2
#         else:
#             return 0
#     return g

# def CI_intervals(model, X, X_train, Y_train, alpha, D, covinv):
#     CI = np.zeros((len(X), 2))
#     predictions = model.predict(X_train)
#     mu_hats = predictions[:, 0]
#     start_values = model.predict(X)[:, 0]
#     sigma_hats = np.exp(predictions[:, 1]) + 1e-3
#     for j, x in enumerate(X):
#         l, u = CI_x(x, X_train, Y_train, mu_hats, sigma_hats, start_values[j], alpha, D, covinv)
#         CI[j, 0] = l
#         CI[j, 1] = u
#     return CI

# def accept(X, Y, mu_hats, sigma_hats, x_0, y_0, alpha, D, covinv=None, positive=True):
#     g = get_perturbation(x_0, y_0, D, covinv)
#     if positive:
#         mu_0 = mu_hats + np.array(list(map(g, X)))
#     if not positive:
#         mu_0 = mu_hats - np.array(list(map(g, X)))
#     ratio = LR(Y, mu_0, mu_hats, sigma_hats)
#   #  q = scipy.stats.chi2(1).ppf(1 - alpha / 2)
#     C = get_C(sigma_hats, g, X)
#     critical_value = scipy.stats.norm(loc=-C, scale=2*np.sqrt(C)).ppf(1 - alpha)
#     # print(f'q={q}, ratio={ratio}')
#     if ratio > critical_value:
#         return 0
#     else:
#         return 1

