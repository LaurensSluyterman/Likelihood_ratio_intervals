import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from intervals import get_perturbation_function
from intervals_2 import CI_NN
from neural_networks import LikelihoodNetwork
from mve_network import MVENetwork
from copy import deepcopy
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 0.2
importlib.reload(mve_network)

size = (40, 1)
X = np.hstack((np.random.uniform(-1, -0.2, size=40), np.random.uniform(0.2, 1, size=40)))
Y = np.random.normal(loc=2*X**2, scale=0.1)
X = np.reshape(X, (80, 1))
# X_n = (X - np.mean(X)) / np.std(X)
#
# Y_n = (Y - np.mean(Y)) /  np.std(Y)
#
# a = LikelihoodNetwork(X, Y, np.array([30, 20, 10]), n_epochs=100, verbose=1, normalization=True,
#                       get_rho_constant=False)

b = MVENetwork(X=X, Y=Y, n_hidden_mean=np.array([40, 30, 20]), n_hidden_var=np.array([5]),
                     n_epochs=100, verbose=1, normalization=True)
c = deepcopy(b)
x_star = 0
delta = 0.5
c.train_on(x_new=np.array(x_star), X_train=X, Y_train=Y, step=delta, n_epochs=200,
           positive=1, verbose=1, fraction=0.1)
model = a.model
model.evaluate(X_n, Y_n)

alpha = 0.2
x_test = np.linspace(-1,1,120)
x_test_n = (x_test - np.mean(X)) / np.std(X)
CI_3, predictions = a.CI(x_test, X, Y, alpha, rho=a.rho, method='Wald')
CI_4, predictions_2 = a.CI(x_test, X, Y, alpha / 10, rho=a.rho, method='Wald')

plt.figure(figsize=(9,6), dpi=120)
plt.title(r'$80\%$ CI')
# plt.fill_between(x_test, CI[:, 0], CI[:, 1], color='grey', alpha=0.2, label='D=3', linewidth=0.1)
plt.fill_between(x_test, CI_3[:, 0], CI_3[:, 1], color='blue', alpha=0.2, linewidth=0.1, label=0.2)
plt.fill_between(x_test, CI_4[:, 0], CI_4[:, 1], color='red', alpha=0.2, linewidth=0.1, label=0.02)
plt.plot(X, Y, 'o', alpha=0.2)
plt.plot(x_test, a.f(x_test), label=r'$\hat{\mu}$')
plt.plot(x_test, predictions, label=r'$\hat{\mu}_d$')
plt.plot(x_test, 2*x_test**2, linestyle='--', label=r'$\mu$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.tight_layout()
plt.show()
#%%
x_test = np.linspace(-1.2, 1.2, 100)
plt.title(f'delta={delta}')
plt.plot(x_test, c.f(x_test), label='perturbed')
plt.plot(x_test, b.f(x_test), label='original')
plt.axvline(x_star)
plt.plot(X, Y, 'o')
plt.legend()
plt.show()

#%%
x_test = np.linspace(-1.2, 1.2, 100)
plt.title(f'delta={delta}')
plt.plot(x_test, c.f(x_test) - b.f(x_test), label='perturbed')
plt.axvline(x_star)
plt.legend()
plt.show()

#%%
test = CI_NN(MVE_network=b, X=np.linspace(-1.2, 1.2, 20), X_train=X, Y_train=Y, alpha=0.2, n_steps=30,
             n_epochs=100, step=0.4)

#%%
x_test = np.linspace(-1.2, 1.2, 20)
plt.figure(figsize=(9,6), dpi=120)
plt.title(r'$80\%$ CI')
# plt.fill_between(x_test, CI[:, 0], CI[:, 1], color='grey', alpha=0.2, label='D=3', linewidth=0.1)
plt.fill_between(x_test, test[:, 0], test[:, 1], color='blue', alpha=0.2, linewidth=0.1, label=0.2)
plt.plot(X, Y, 'o', alpha=0.2)
plt.plot(x_test, b.f(x_test), label=r'$\hat{\mu}$')
plt.plot(x_test, 2*x_test**2, linestyle='--', label=r'$\mu$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.tight_layout()
plt.show()

#%%
x_test = np.linspace(-1.2, 1.2, 80)
N_simulations = 75
coverage_2 = np.zeros((N_simulations, len(x_test)))
alpha = 0.2
for i in range(N_simulations):
    print(f'simulation nr {i + 1}')
    Y_train_new = np.random.normal(loc=2*X**2, scale=0.1)
    network = LikelihoodNetwork(X, Y_train_new, np.array([30, 20, 10]), n_epochs=200,
                          verbose=0, normalization=True)
    CI = network.CI(x_test, X, Y_train_new, 0.2)
    coverage_2[i] = CI_coverage_probability(CI, 2*x_test**2)

plt.plot(x_test, np.mean(coverage_2, axis=0))
plt.ylim(0, 1.1)
plt.axhline(1-alpha, linestyle='--')
plt.plot(X, np.ones(len(X))*(1-alpha), 'o')
plt.xlabel('x')
plt.ylabel('coverage')
plt.show()


# Visualizing the perturbation
x = np.linspace(-1, 1, 100)
for rho in [1, 2, 8, 16]:
    g = get_perturbation_function(0.2, 0.1, np.array([[np.sqrt(rho)]]))
    perturbations = list(map(g, x))
    plt.plot(x, perturbations, label=r'$\rho=$'+f'{rho}')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.show()

for delta in [0.1, 0.2, 0.3, 0.4]:
    g = get_perturbation_function(0.2, delta, np.array([[2]]))
    perturbations = list(map(g, x))
    plt.plot(x, perturbations, label=r'$\delta=$'+f'{delta}')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.show()

