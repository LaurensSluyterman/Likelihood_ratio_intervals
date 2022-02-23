import numpy as np
from utils import load_data
from neural_networks import LikelihoodNetwork
from sklearn.model_selection import train_test_split
from target_simulation import create_true_function_and_variance, gen_new_targets
from metrics import IntervalMetrics
import matplotlib.pyplot as plt
from klepto.archives import dir_archive

#importlib.reload(packagename)
data_directory = 'bostonHousing'
X, Y = load_data(data_directory)
distribution = 'Gaussian'
f, var_true = create_true_function_and_variance(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
N_test = len(X_test)

Y_train_new = gen_new_targets(X_train, f, var_true, dist=distribution)
network = LikelihoodNetwork(X_train, Y_train_new, np.array([40, 30, 20]), n_epochs=120, verbose=1, normalization=True,
                            get_second_derivative=False)
# model = network.model
# X_ood = np.random.normal(loc=np.mean(X_train, axis=0), scale=np.std(X_train, axis=0), size=(100,8))
CI_normal = network.CI(X_test[0:10], X_train, Y_train_new, alpha=0.2, rho=25)
# CI_ood = network.CI(X_ood, X_train, Y_train_new, alpha=0.2, D=network.D)
# CI_normal
# np.mean(CI_normal[:, 1] - CI_normal[:,0])
# network.f(X_test[1:5])
# f(X_test[0:5])


N_simulations = 100
metrics = IntervalMetrics(N_simulations, [0.2], N_test)
for i in range(0, N_simulations):
    print(f'Simulation {i + 1}/{N_simulations}')
    # Simulate new data
    Y_train_new = gen_new_targets(X_train, f, var_true, dist=distribution)
    Y_test = gen_new_targets(X_test, f, var_true, dist=distribution)
    f_test = f(X_test)
    network = LikelihoodNetwork(X_train, Y_train_new, np.array([40, 30, 20]), n_epochs=80, verbose=0,
                                normalization=True,
                                get_second_derivative=True)
    CI = network.CI(X_test, X_train, Y_train_new, alpha=0.2, rho=network.rho)
    metrics.update_bias(network.f(X_test), f_test, i)
    metrics.update_CI(CI, f_test, i, 0)


## Saving and plotting the results
CICF_values = np.mean(metrics.CI_correct, axis=0)[0]
average_widths = np.mean(metrics.CI_width, axis=0)[0]

plt.hist(np.mean(metrics.CI_correct, axis=0)[0])
plt.xlim((0,1))
plt.show()

np.shape(metrics.CI_width)
_PLOT_LOCATION = "/Users/laurens/OneDrive/Onedrivedocs/PhD/Code/2022/Hessian_intervals/Results/" +  data_directory + '/plots/'
_METRICS_LOCATION = "/Users/laurens/OneDrive/Onedrivedocs/PhD/Code/2022/Hessian_intervals/Results/" + data_directory + "/raw_results/hdpertreluc"
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

bias = np.mean(metrics.bias, axis=0)
plt.scatter(bias, average_widths)
plt.xlabel('bias')
plt.ylabel('width')
plt.show()