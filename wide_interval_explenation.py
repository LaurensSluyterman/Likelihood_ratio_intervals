import importlib

import numpy as np
from utils import load_data, normalize, CI_coverage_probability, reverse_normalized
from neural_networks import LikelihoodNetwork
from sklearn.model_selection import train_test_split
from target_simulation import create_true_function_and_variance, gen_new_targets
from second_derivative_utils import get_E, get_second_derivative_matrix
from intervals import get_perturbation_function, CI_intervals
from wald_intervals import CI_wald_x
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
import scipy
import keras.backend as K
import tensorflow as tf

importlib.reload(plt)

data_directory = 'bostonHousing'
X, Y = load_data(data_directory)
distribution = 'Gaussian'
f, var_true = create_true_function_and_variance(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
X_train_n = normalize(X_train)
X_test_n = normalize(X_test, np.mean(X_train, axis=0), np.std(X_train, axis=0))
X_n = np.vstack((X_train_n, X_test_n))
N_test = len(X_test)
alpha = 0.2

# pca = KernelPCA(n_components=1)
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

Y_train_new = gen_new_targets(X_train, f, var_true, dist=distribution)

Y_test_new = gen_new_targets(X_test, f, var_true, dist=distribution)
Y_train_n = normalize(Y_train_new)
network = LikelihoodNetwork(X_train, Y_train_new, np.array([40, 30, 20]), n_epochs=80, verbose=True, normalization=True,
                            get_rho_constant=True)

CI, predictions = network.CI(X_test[0:5], X_train, Y_train_new, alpha=0.2, rho=network.rho, method='Wald')
# CI_n, E_matrices = CI_intervals(network.model, X_test_n, X_train_n, Y_train_n, alpha, rho=network.rho,
#                                 give_E_matrices=True, method='Wald')
# CI = CI_n * np.std(Y_train_new) + np.mean(Y_train_new) * np.ones(np.shape(CI_n))
# widths = (CI_n[:, 1] - CI_n[:, 0]) * np.std(Y_train_new)
widths = CI[:, 1] - CI[:, 0]
skewness = (network.f(X_test) - CI[:, 0]) / (CI[:, 1] - network.f(X_test))
plt.hist(skewness)
plt.show()

lower = (network.f(X_test) - CI[:, 0]) > (CI[:, 1] - network.f(X_test))
actually_lower = network.f(X_test) - f(X_test) > 0
print(f'bias fraction is {np.mean(lower == actually_lower)}')
print(f'coverage={np.mean(CI_coverage_probability(CI, f(X_test)))}')
widths = CI[:, 1] - CI[:, 0]
print(f'mean width {np.mean(widths)}')
indices = [np.where(widths == i)[0][0] for i in sorted(widths, reverse=True)]  # Indices of intervals in descending width

np.sqrt(np.mean((np.mean(CI, axis=1) - Y_test)**2))
np.sqrt(np.mean((predictions[:,0] - Y_test)**2))
np.sqrt(np.mean((network.f(X_test) - Y_test)**2))

plt.hist((predictions[:,0] - Y_test)**2)
plt.hist((network.f(X_test) - Y_test)**2, alpha=0.5)
plt.show()
plt.hist(np.abs(bias_2) - np.abs(bias_1))
plt.xlabel('vanilla - CI')
plt.show()
bias_1 = (np.mean(CI, axis=1) - f(X_test))
bias_2 = (network.f(X_test) - f(X_test))
plt.plot(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
plt.xlabel('CI')
plt.ylabel('vanilla')
plt.scatter(bias_1, bias_2)
plt.show()

plt.hist((network.f(X_test) - CI[:,0]) / (CI[:,1] - network.f(X_test)))
plt.show()

xs = np.linspace(0.00001, 20, 100)
plt.plot(xs, -xs + 2*np.sqrt(xs) * 1.6)
plt.xlabel('C')
plt.ylabel('q')
plt.show()



plt.hist(widths)
plt.xlabel('width')
plt.show()



number_of_points_hit = np.zeros((len(X_test), 2))
for i, x in enumerate(X_test_n):
    print(i+1)
    # A = get_second_derivative_matrix(network.model, x)
    # E = get_E(A, network.rho)
    E = E_matrices[i]
    delta_1 = network.model.predict(np.array([x]))[:, 0] - CI_n[i, 0] + 0.01
    delta_2 = CI_n[i, 1] - network.model.predict(np.array([x]))[:, 0] + 0.01
    g_1 = get_perturbation_function(x, delta_1, E)
    g_2 = get_perturbation_function(x, delta_2, E)
    hits_1 = list(map(g_1, X_train_n))
    hits_2 = list(map(g_2, X_train_n))
    hit_indices_1 = []
    hit_indices_2 = []
    for j, hit in enumerate(hits_1):
        if hit != 0:
            hit_indices_1.append(j)
    for j, hit in enumerate(hits_2):
        if hit != 0:
            hit_indices_2.append(j)
    number_of_points_hit[i, 0] = len(hit_indices_1)
    number_of_points_hit[i, 1] = len(hit_indices_2)

plt.scatter(np.sum(number_of_points_hit, axis =1), widths)
plt.xlabel('number of  points hit')
plt.ylabel('width')
plt.show()
number_of_points_hit

def p(delta, network, X, Y, E, positive=1):
    mu_hats = network.model.predict(X)[:, 0]
    sigma_hats = np.exp(network.model.predict(X)[:, 1]) + 1e-3
    g = get_perturbation_function(x, delta, E)
    C_values = np.array(list(map(g, X))) / sigma_hats * positive
    C = np.sum(C_values**2)
    if C == 0:
        return 1
    Z_values = (mu_hats - normalize(Y)) / sigma_hats
    return 1-scipy.stats.norm.cdf((np.sum(C_values * Z_values) + C) / np.sqrt(C))

x = X_test_n[94]
A = get_second_derivative_matrix(network.model, x, B=500)
E = get_E(A, network.rho)
deltas = [x / 100 for x in range(0,300)]
positiveps = [p(delta, network, X_train_n, Y_train, E, positive=1) for delta in deltas]
negavitveps = [p(delta, network, X_train_n, Y_train, E, positive=-1) for delta in deltas]
plt.plot(deltas, positiveps, label='+')
plt.plot(deltas, negavitveps, label='-')
plt.axhline(0.1)
plt.xlabel('delta')
plt.ylabel('p')
plt.legend()
plt.show()

def LLR(x, delta, network, X, Y, E, positive=1):
    g = get_perturbation_function(x, delta, E)
    mu_1 = network.model.predict(X)[:, 0]
    g_values = np.array(list(map(g, X))) * positive
    mu_0 = mu_1 + g_values
    sigma_hats = np.exp(network.model.predict(X)[:, 1]) + 1e-3
    R = np.sum(-0.5 * ((Y - mu_0) / sigma_hats)**2 + 0.5 * ((Y - mu_1) / sigma_hats)**2)
    return R

def cL(x, delta, X, Y, mu_hats, sigma_hats, E):
    g = get_perturbation_function(x, np.abs(delta), E)
    mu_delta = mu_hats + np.array(list(map(g, X))) * np.sign(delta)
    return - np.sum((Y - mu_delta)**2 / sigma_hats**2)

mu_hats = a.model.predict(X_n)[:,0]
sigma_hats = np.exp(a.model.predict(X_n)[:, 1]) + 1e-3
cL(x_test_n[1], 0.8, X_n, Y_n, mu_hats, sigma_hats, np.array([[3]]))

deltas = np.linspace(-1, 1, 100)
cLs = [cL(0.0, delta, X_n, Y_n, mu_hats, sigma_hats, np.array([[5]])) for delta in deltas]
values = np.max(cLs) - cLs
q = scipy.stats.chi2(1).ppf(1 - alpha / 2)
indices = np.where(values < q)[0]
lowerbound, upperbound = deltas[indices[0]], deltas[indices[-1]]
print(lowerbound, upperbound)
delta = 0.5
g = get_perturbation_function(0.1, delta, np.array([[5]]))
plt.plot(X_n, Y_n, 'o')
plt.plot(np.sort(x_test_n),  a.model.predict(np.sort(x_test_n))[:,0] + np.array(list(map(g, x_test_n))))
plt.plot(np.sort(x_test_n), a.model.predict(np.sort(x_test_n))[:,0])
plt.show()



plt.plot(deltas, values)
plt.show()



def get_CI_new(x, X, Y, mu_hats, sigma_hats, E):
    deltas = np.linspace(-3, 3, 100)
    cLs = [cL(x, delta, X, Y, mu_hats, sigma_hats, E) for delta in deltas]
    values = np.max(cLs) - cLs
    print(np.round(values, 2))
    q = scipy.stats.chi2(1).ppf(1 - alpha / 2)
    indices = np.where(values < q)[0]
    print(f'delta_star={deltas[np.where(values==0)]}')
    lowerbound, upperbound = deltas[indices[0]], deltas[indices[-1]]
    return lowerbound, upperbound

get_CI_new(x_test_n[60], X_n, Y_n, mu_hats, sigma_hats, np.array([[5]]))

CI_new = np.zeros((len(x_test), 2))
for i, x in enumerate(x_test_n):
    print(i)
    CI_new[i] = CI_wald_x(model=a.model, x0=x, X_train=X_n, Y_train=X_n, mu_hats=mu_hats,
              sigma_hats=sigma_hats, E=np.array([[5]]), alpha=0.2)
  #  CI_new[i] = get_CI_new(x, X_n, Y_n, mu_hats, sigma_hats, np.array([[5]]))

CI_new[:, 0] = CI_new[:, 0] + a.model.predict(x_test_n)[:, 0]
CI_new[:, 1] = CI_new[:, 1] + a.model.predict(x_test_n)[:, 0]
CI_new_un = CI_new * np.std(Y) + np.mean(Y) * np.ones(np.shape(CI_new))
CI_3 = CI_new_un
widths_new = (CI_new[:, 1] - CI_new[:, 0]) * np.std(Y_train_new)
Y_train_n = normalize(Y_train_new)
LLR(x, delta_1 + 0.4, network, X_train_n, Y_train_n, E, positive=1)


x = X_test_n[22]
E = E_matrices[22]
deltas = [x / 100 for x in range(0,200)]
LLS_p = [LLR(x, delta, network, X_train_n, Y_train_n, E, positive=1) for delta in deltas]
LLS_n = [LLR(x, delta, network, X_train_n, Y_train_n, E, positive=-1) for delta in deltas]
plt.plot(deltas, LLS_p, label='positive')
plt.plot(deltas, LLS_n, label='negative')
plt.axhline(np.log(alpha / (1-alpha)), linestyle='--')
plt.ylim((np.log(alpha / (1-alpha))-1, 1.5))
plt.xlabel('delta')
plt.ylabel('log(LL(H0) / LL(H1))')
plt.legend()
plt.show()




hit_indices = []
for i, hit in enumerate(hits):
    if hit != 0:
        hit_indices.append(i)
hits.reshape((len(hits), 1)) != 0
np.shape(hits)

all_eigen_values = np.zeros(np.shape(X_test_n))

for i, x in enumerate(X_test_n):
    print(f'{i+1} of {len(X_test_n)}')
    E = E_matrices[i]
    eigen_values, eigenvectors = np.linalg.eigh(E)
    all_eigen_values[i] = eigen_values

for i in range(len(all_eigen_values[0])):
    plt.scatter(all_eigen_values[:, i], widths)
    plt.title(f'eigenvalue {i+1}')
    plt.show()


for i in range(len(X_test_n[0])):
    plt.scatter(X_test_n[:, i], widths)
    plt.title(f'Covariate {i+1}')
    plt.show()


pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_n)
X_test_pca = pca.transform(X_test_n)

n_large = 10
n_small = 10
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], label='Training data', s=20)
plt.scatter(X_test_pca[:, 0][indices[0:n_large]], X_test_pca[:, 1][indices[0:n_large]],
            label=f'{n_large} largest intervals', s=50)
plt.scatter(X_test_pca[:, 0][indices[-n_small:]], X_test_pca[:, 1][indices[-n_small:]],
            label=f'{n_small} smallest intervals', s=50)
plt.legend()
plt.show()

sigmas = network.sigma(X_train)



max(sigmas)
plt.scatter(np.sqrt(var_true(X_train)), sigmas)
plt.ylabel('predicted')
plt.xlabel('true')
plt.plot(np.linspace(0, 10, 20), np.linspace(0, 10, 20), color='r')
plt.xlim((0, 10))
plt.ylim((0, 20))
plt.show()

