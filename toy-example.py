import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from intervals import get_perturbation_function
from neural_networks import LikelihoodNetwork
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 0.2



X = np.hstack((np.random.uniform(-1, -0.2, size=40), np.random.uniform(0.2, 1, size=40)))
X_n = (X - np.mean(X)) / np.std(X)
Y = np.random.normal(loc=2*X**2, scale=0.1)
Y_n = (Y - np.mean(Y)) /  np.std(Y)
a = LikelihoodNetwork(X, Y, np.array([30, 20, 10]), n_epochs=100, verbose=1, normalization=True,
                      get_rho_constant=True)

x_test = np.linspace(-1.5,1.5,120)
CI_3 = a.CI(x_test, X, Y, 0.2, rho=a.rho)

plt.figure(figsize=(9,6), dpi=120)
plt.title(r'$80\%$ CI')
# plt.fill_between(x_test, CI[:, 0], CI[:, 1], color='grey', alpha=0.2, label='D=3', linewidth=0.1)
plt.fill_between(x_test, CI_3[:, 0], CI_3[:, 1], color='blue', alpha=0.2, linewidth=0.1)
plt.plot(X, Y, 'o', alpha=0.2)
plt.plot(x_test, a.f(x_test), label=r'$\hat{\mu}$')
plt.plot(x_test, 2*x_test**2, linestyle='--', label=r'$\mu$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.tight_layout()
plt.show()

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

