import numpy as np
from scipy import stats
import scipy


def CI_intervals(network, X, X_train, Y_train, alpha, D):
    CI = np.zeros((len(X), 2))
    for j, x in enumerate(X):
        l, u = CI_x(x, network, X_train, Y_train, alpha, D)
        CI[j, 0] = l
        CI[j, 1] = u
    return CI


def CI_x(x_0, model, X_train, Y_train, alpha, D, stepsize=0.005):
    upperbound = model.f([x_0])
    accepting = True
    n = 1
    while accepting:
        y = upperbound + stepsize
        accepting = accept(model, X_train, Y_train, x_0, stepsize * n, D, alpha, positive=1)
        if accepting:
            upperbound = y
            n += 1
    accepting = True
    lowerbound = model.f([x_0])
    n = 1
    while accepting:
        y = lowerbound - stepsize
        accepting = accept(model, X_train, Y_train, x_0, stepsize * n, D, alpha, positive=0)
        if accepting:
            lowerbound = y
            n += 1
    return lowerbound[0], upperbound[0]


def accept(model, X, Y, x_0, y_0, D, alpha, positive=True):
    g = get_perturbation(x_0, y_0, D)
    mu_hat = model.f(X)
    sigma = model.sigma(X)
    if positive:
        mu_0 = mu_hat + np.array(list(map(g, X)))
    if not positive:
        mu_0 = mu_hat - np.array(list(map(g, X)))
    ratio = LR(Y, mu_0, mu_hat, sigma)
 #   q = scipy.stats.chi2(1).ppf(1 - alpha / 2)
    C = get_C(model, g, X)
    q = scipy.stats.norm(loc=-C, scale=2*np.sqrt(C)).ppf(1 - alpha / 2)
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


def get_C(model, g, X):
    sigma = model.sigma(X)
    C = np.sum(np.array(list(map(g, X)))**2 / sigma**2) + 1e-3
    return C


def LR(y, mu_0, mu_hat, sigma):
    return 2 * (- 0.5 * np.sum(((y - mu_hat) / sigma)**2) + 0.5 * np.sum(((y - mu_0) / sigma)**2))


y = list(map(a, x))
b = get_perturbation(np.array([0.1, 2]), 0.1, 10)
y_2 = list(map(b, x))
x = np.linspace(1.5, 2.5, 100)

b([0.1, 20])
plt.plot(x, y)
plt.plot(x, y_2)
plt.show()