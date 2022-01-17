import numpy as np
from utils import get_second_derivative_constant, CI_coverage_probability
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from intervals import CI_intervals, CI_x
from neural_networks import LikelihoodNetwork



X = np.hstack((np.random.uniform(-1, -0.2, size=40), np.random.uniform(0.2, 1, size=40)))
Y = np.random.normal(loc=2*X**2, scale=0.1)

a = LikelihoodNetwork(X, Y, np.array([30, 20, 10]), n_epochs=300, verbose=1, normalization=False)
model = a.model



D = get_second_derivative_constant(X, model)

x_test = np.linspace(-1,1,200)
plt.plot(x_test, a.f(x_test))
plt.plot(x_test, 2 * x_test**2)
plt.tight_layout()
plt.show()



x_test = np.linspace(-1.5, 1.5, 70)
CI = CI_intervals(a, x_test, X, Y, 0.1, 3)
CI_2 = CI_intervals(a, x_test, X, Y, 0.1, 15)


plt.title(r'$90\%$ CI')
plt.fill_between(x_test, CI[:, 0], CI[:, 1], color='grey', alpha=0.2, label='D=3', linewidth=0.1)
plt.fill_between(x_test, CI_2[:, 0], CI_2[:, 1], color='blue', alpha=0.2, label='D=15', linewidth=0.1)
plt.plot(X, Y, 'o', alpha=0.2)
plt.plot(x_test, a.f(x_test), label=r'$\hat{\mu}$')
plt.plot(x_test, 2*x_test**2, linestyle='--', label=r'$\mu$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

x_test = np.linspace(-1.2, 1.2, 80)
N_simulations = 75
coverage_2 = np.zeros((N_simulations, len(x_test)))
alpha = 0.2
for i in range(N_simulations):
    print(f'simulation nr {i + 1}')
    Y_train_new = np.random.normal(loc=2*X**2, scale=0.1)
    network = LikelihoodNetwork(X, Y_train_new, np.array([30, 20, 10]), n_epochs=200,
                          verbose=0, normalization=False)
    D = get_second_derivative_constant(X, network.model)
    CI = CI_intervals(network, x_test, X, Y_train_new, alpha, D)
    coverage_2[i] = CI_coverage_probability(CI, 2*x_test**2)

plt.plot(x_test, np.mean(coverage_2, axis=0))
plt.ylim(0, 1.1)
plt.axhline(1-alpha, linestyle='--')
plt.plot(X, np.ones(len(X))*(1-alpha), 'o')
plt.xlabel('x')
plt.ylabel('coverage')
plt.show()
