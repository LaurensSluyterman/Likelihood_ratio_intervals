import numpy as np
from utils import load_data
from neural_networks import LikelihoodNetwork
from sklearn.model_selection import train_test_split
from target_simulation import create_true_function_and_variance, gen_new_targets
from utils import get_second_derivative_constant

X, Y = load_data('bostonHousing')
distribution = 'Gaussian'
f, var_true = create_true_function_and_variance(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
N_test = len(X_test)

Y_train_new = gen_new_targets(X_train, f, var_true, dist=distribution)
network = LikelihoodNetwork(X_train, Y_train_new, np.array([40, 30, 20]), n_epochs=80, verbose=1, normalization=True,
                            get_second_derivative=False)

CI = network.CI(X_test[1:3], X, Y, 0.1, D=4)
model = network.model

get_gradient(X[1], model)

D = get_second_derivative_constant(X, model)

