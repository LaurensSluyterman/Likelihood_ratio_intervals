import numpy as np
from utils import load_data
# from neural_networks import LikelihoodNetwork
from mve_network import MVENetwork
from sklearn.model_selection import train_test_split
from target_simulation import create_true_function_and_variance, gen_new_targets
from metrics import IntervalMetrics
import matplotlib.pyplot as plt
from copy import deepcopy
from klepto.archives import dir_archive
from intervals_2 import CI_NN
import keras.backend as K
import gc
from importlib import reload
# import mve_network
# from importlib import reload
# reload(mve_network)
import intervals_2
reload(intervals_2)
#importlib.reload(packagename)
data_directory = 'bostonHousing'
X, Y = load_data(data_directory)
distribution = 'Gaussian'
np.random.seed(2)
f, var_true = create_true_function_and_variance(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=1)
N_test = len(X_test)

Y_train_new = gen_new_targets(X_train, f, var_true, dist=distribution)
# network = LikelihoodNetwork(X_train, Y_train_new, np.array([40, 30, 20]), n_epochs=120, verbose=1, normalization=True,
#                             get_rho_constant=True)
##
network = MVENetwork(X=X_train, Y=Y_train_new, n_hidden_mean=np.array([40, 30, 20]), n_hidden_var=np.array([5]),
                     n_epochs=100, verbose=1, normalization=True)
network_2 = deepcopy(network)
network_3 = deepcopy(network)
fraction=0.2
network_2.train_on(x_new=np.array([X_train[0]]), X_train=X_train,
                   Y_train=Y_train_new, n_epochs=50, step=0.5,
                   fraction=0.1, verbose=1, batch_size=None)
start_2 = network_2.f(X_test[0:5])
network_3.train_on(x_new=np.array([X_train[0]]), X_train=X_train,
                   Y_train=Y_train_new, n_epochs=50, step=0.5,
                   fraction=0.1, verbose=1, batch_size=None, positive=0)
plt.hist(network_2.f(X_train) - network.f(X_train))
plt.show()
network_2.f(X_train) - network.f(X_train)

start = network.f(X_test[0:5])
test = CI_NN(MVE_network=network, X=X_test[0:5], X_train=X_train,
             Y_train=Y_train_new, alpha=0.2, n_steps=30,
             n_epochs=100, step=0.5, fraction=0.1)
print(test)
f(X_test)[0:10]
test[:,1] - network.f(X_test[0:5])
test[:,0] - network.f(X_test[0:5])
correct = (test[:, 0] < f(X_test[0:5])) * (test[:, 1] > f(X_test[0:5]))
print(correct)
# %%

mu_hat = network.f(X_test)
network_2 = deepcopy(network)
train_on(network_2, np.array([X_test[0]]), X_train, Y_train_new, step=2, positive=False)
# model = network.model
# X_ood = np.random.normal(loc=np.mean(X_train, axis=0), scale=np.std(X_train, axis=0), size=(100,8))
CI_normal = network.CI(X_test, X_train, Y_train_new, alpha=0.2, rho=network.rho)
CI_wald = network.CI(X_test[0:10], X_train, Y_train_new, alpha=0.2, rho=network.rho, method='Wald')
# CI_ood = network.CI(X_ood, X_train, Y_train_new, alpha=0.2, D=network.D)
# CI_normal
# np.mean(CI_normal[:, 1] - CI_normal[:,0])
# network.f(X_test[1:5])
# f(X_test[0:5])

# %%
N_simulations = 30
metrics = IntervalMetrics(N_simulations, [0.2], N_test)
for i in range(0, N_simulations):
    print(f'Simulation {i + 1}/{N_simulations}')
    # Simulate new data
    Y_train_new = gen_new_targets(X_train, f, var_true, dist=distribution)
    Y_test_new = gen_new_targets(X_test, f, var_true, dist=distribution)
    f_test = f(X_test)
    # network = LikelihoodNetwork(X_train, Y_train_new, np.array([40, 30, 20]), n_epochs=80, verbose=0,
    #                             normalization=True,
    #                             get_rho_constant=True)
    network = MVENetwork(X=X_train, Y=Y_train_new,
                         n_hidden_mean=np.array([40, 30, 20]),
                         n_hidden_var=np.array([5]),
                         n_epochs=200, verbose=0, normalization=True)
    # CI, predictions = network.CI(X_test, X_train, Y_train_new, alpha=0.2, rho=network.rho, method='Wald')
    predictions = network.f(X_test)
    CI = CI_NN(MVE_network=network, X=X_test, X_train=X_train, Y_train=Y_train_new,
                 alpha=0.2, n_steps=30,
                 n_epochs=100, step=1)
    metrics.update_bias(predictions, f_test, i)
    metrics.update_error(predictions, Y_test_new, i)
    metrics.update_CI(CI, f_test, i, 0)
    del network
    gc.collect()
    K.clear_session()

#%% Saving and plotting the results
CICF_values = np.mean(metrics.CI_correct, axis=0)[0]
average_widths = np.mean(metrics.CI_width, axis=0)[0]

plt.hist(np.mean(metrics.CI_correct, axis=0)[0])
plt.xlim((0,1))
plt.show()

np.shape(metrics.CI_width)
_PLOT_LOCATION = "/Users/laurens/OneDrive/Onedrivedocs/PhD/Code/2022/Hessian_intervals/Results/" +  data_directory + '/plots/'
_METRICS_LOCATION = "/Users/laurens/OneDrive/Onedrivedocs/PhD/Code/2022/Hessian_intervals/Results/" + data_directory + "/raw_results/NNalternativesd1alpha02"
metrics_list = ['metrics']
metrics_dir = {metric: eval(metric) for metric in metrics_list}
results = dir_archive(_METRICS_LOCATION, metrics_dir, serialized=True)
results.dump()

fig = plt.figure()
plt.scatter(average_widths, CICF_values)
plt.axhline(0.8)
plt.xlabel('width')
plt.ylabel('CICF')
plt.ylim((0, 1))
plt.show()
fig.savefig(_PLOT_LOCATION + data_directory + f'widthCICF.png')

fig = plt.figure()
plt.violinplot(CICF_values)
plt.axhline(1 - 0.2, color='k', linestyle='dashed', linewidth=1)
plt.ylim((0, 1))
plt.ylabel('CICF')
plt.xlim((0, 2))
plt.title(f'BS={np.mean((CICF_values - 0.8)**2)}')
plt.show()
fig.savefig(_PLOT_LOCATION + data_directory + f'CICFviolin.png')

np.mean((CICF_values - 0.8)**2)

np.mean(metrics.CI_width)

bias = np.mean(metrics.bias, axis=0)
plt.scatter(bias, CICF_values)
plt.axhline(0.8)
plt.xlabel('bias')
plt.ylabel('CICF')
plt.show()

bias = np.mean(metrics.bias**2, axis=0)
plt.scatter(bias, average_widths)
plt.xlabel('mse (of function values)')
plt.ylabel('width')
plt.show()


a = predictions - CI[:, 0]
b = predictions - CI[:, 1]
c = a*b