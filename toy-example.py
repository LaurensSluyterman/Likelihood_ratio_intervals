import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from intervals_2 import CI_NN
from mve_network import MVENetwork
from metrics import CI_coverage_probability
matplotlib.use("TkAgg")
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 0.2


#%% Simulating data
np.random.seed(2)
size = (40, 1)
X = np.hstack((np.random.uniform(-1, -0.2, size=40), np.random.uniform(0.2, 1, size=40)))
Y = np.random.normal(loc=2*X**2, scale=0.1)
X = np.reshape(X, (80, 1))

#%% Training a model
model = MVENetwork(X=X, Y=Y, n_hidden_mean=np.array([40, 30, 20]), n_hidden_var=np.array([5]),
                     n_epochs=200, verbose=1, normalization=True)

#%% Visualizing the perturbation
x_test = np.linspace(-1.2, 1.2, 100)
plt.title(f'delta={delta}')
plt.plot(x_test, model.f_perturbed(x_test, positive=True), label='perturbed')
plt.plot(x_test, model.f(x_test), label='original')
plt.axvline(x_star)
plt.plot(X, Y, 'o')
plt.legend()
plt.show()


#%% Creating confidence intervals
x_test = np.linspace(-1.2, 1.2, 20)
CI = CI_NN(MVE_network=model, X=x_test, X_train=X, Y_train=Y, alpha=0.2,
           n_steps=30, n_epochs=100, step=0.4)

#%% Visualizing the confidence intervals
plt.figure(figsize=(9,6), dpi=120)
plt.title(r'$80\%$ CI')
plt.fill_between(x_test, CI[:, 0], CI[:, 1], color='blue', alpha=0.2, linewidth=0.1, label=0.2)
plt.plot(X, Y, 'o', alpha=0.2)
plt.plot(x_test, model.f(x_test), label=r'$\hat{\mu}$')
plt.plot(x_test, 2*x_test**2, linestyle='--', label=r'$\mu$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.tight_layout()
plt.show()

#%% Testing coverage
x_test = np.linspace(-1.2, 1.2, 40)
N_simulations = 75
coverage_2 = np.zeros((N_simulations, len(x_test)))
alpha = 0.2
for i in range(N_simulations):
    print(f'simulation nr {i + 1}')
    Y_train_new = np.random.normal(loc=2*X**2, scale=0.1)
    network = MVENetwork(X=X, Y=Y_train_new, n_hidden_mean=np.array([40, 30, 20]),
                       n_hidden_var=np.array([5]),
                       n_epochs=200, verbose=1, normalization=True)
    CI = CI_NN(MVE_network=model, X=x_test, X_train=X, Y_train=Y_train_new, alpha=0.2,
               n_steps=50, n_epochs=100, step=0.4)
    coverage_2[i] = CI_coverage_probability(CI, 2*x_test**2)

plt.plot(x_test, np.mean(coverage_2, axis=0))
plt.ylim(0, 1.1)
plt.axhline(1-alpha, linestyle='--')
plt.plot(X, np.ones(len(X))*(1-alpha), 'o')
plt.xlabel('x')
plt.ylabel('coverage')
plt.show()
