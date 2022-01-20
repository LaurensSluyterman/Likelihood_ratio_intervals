import numpy as np
from scipy import stats
from utils import normalize
import scipy


def CI_intervals(model, X, X_train, Y_train, alpha, D):
    CI = np.zeros((len(X), 2))
    predictions = model.predict(X_train)
    mu_hats = model.predict(X_train)[:, 0]
    start_values = model.predict(X)[:, 0]
    sigma_hats = np.exp(predictions[:, 1]) +1e-3
    for j, x in enumerate(X):
        l, u = CI_x(x, X_train, Y_train, mu_hats, sigma_hats, start_values[j], alpha, D)
        CI[j, 0] = l
        CI[j, 1] = u
    return CI


def CI_x(x_0, X_train, Y_train, mu_hats, sigma_hats, start_value, alpha, D, stepsize=0.01):
    upperbound = start_value
    accepting = True
    n = 1
    while accepting:
        print(upperbound)
        accepting = accept(X_train, Y_train, mu_hats, sigma_hats, x_0, stepsize * n, D, alpha, positive=1)
        if accepting:
            upperbound += stepsize
            n += 1
    accepting = True
    lowerbound = start_value
    n = 1
    while accepting:
        print(lowerbound)
        accepting = accept(X_train, Y_train, mu_hats, sigma_hats, x_0, stepsize * n, D, alpha, positive=0)
        if accepting:
            lowerbound -= stepsize
            n += 1
    print('--')
    return lowerbound, upperbound


def accept(X, Y, mu_hats, sigma_hats, x_0, y_0, D, alpha, positive=True):
    g = get_perturbation(x_0, y_0, D)
    if positive:
        mu_0 = mu_hats + np.array(list(map(g, X)))
    if not positive:
        mu_0 = mu_hats - np.array(list(map(g, X)))
    ratio = LR(Y, mu_0, mu_hats, sigma_hats)
    q = scipy.stats.chi2(1).ppf(1 - alpha / 2)
    C = get_C(sigma_hats, g, X)
 #   q = scipy.stats.norm(loc=-C, scale=2*np.sqrt(C)).ppf(1 - alpha / 2)
    print(f'q={q}')
    if ratio > q:
        return 0
    else:
        return 1


def get_perturbation(x_0, y_0, D):
    try:
        n_dim = len(x_0)
    except TypeError:
        n_dim = 1
        x_0 = [x_0]
    def g(x):
        if n_dim == 1:
            x = [x]
        d = np.sqrt(np.abs(y_0) / (D*n_dim))
        result = 0
        for i in range(n_dim):
            if -d < (x[i]-x_0[i]) < d:
                result += y_0 / n_dim - D / (2*n_dim) * (x[i] - x_0[i])**2
            elif d < (x[i] - x_0[i]) < (2 * d):
                result += D / (2*n_dim) * (x[i] - (x_0[i] + 2 * d))**2
            elif (-2 * d) < (x[i]-x_0[i]) < (-d):
                result += D / (2*n_dim) * (x[i] - (x_0[i] - 2 * d))**2
            else:
                result += 0.
        return result
    return g


def get_C(sigma, g, X):
    C = np.sum(np.array(list(map(g, X)))**2 / sigma**2) + 1e-3
    return C


def LR(y, mu_0, mu_hat, sigma):
    return 2 * (- 0.5 * np.sum(((y - mu_hat) / sigma)**2) + 0.5 * np.sum(((y - mu_0) / sigma)**2))

#
# y = list(map(a, x))
# b = get_perturbation(np.array([0.1, 2]), 0.1, 10)
# y_2 = list(map(b, x))
# x = np.linspace(1.5, 2.5, 100)
#
# b([0.1, 20])
# plt.plot(x, y)
# plt.plot(x, y_2)
# plt.show()