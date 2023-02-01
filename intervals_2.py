from utils import normalize
import scipy
from copy import deepcopy
import numpy as np
import gc
import keras.backend as K

def CI_NN(MVE_network, X, X_train, Y_train, alpha, n_steps=10,
          n_epochs=40, step=1, fraction=0.1):
    CI = np.zeros((len(X), 2))
    mu_hats = MVE_network.f(X_train)
    sigma_hats = MVE_network.sigma(X_train)
    for j, x in enumerate(X):
        lowerbound, upperbound = CI_NNx(MVE_network=MVE_network,
                                        x=x,
                                        X_train=X_train,
                                        Y_train=Y_train,
                                        mu_hats=mu_hats,
                                        sigma_hats=sigma_hats,
                                        step=step,
                                        fraction=fraction,
                                        n_steps=n_steps,
                                        n_epochs=n_epochs,
                                        alpha=alpha,)
        # todo: fix make this robust for different types of inputs
        CI[j, 0] = lowerbound
        CI[j, 1] = upperbound
    gc.collect()
    # K.clear_session()
    return CI

def CI_NNx(*, MVE_network, x, X_train, Y_train, mu_hats, sigma_hats,
           n_steps, alpha, n_epochs, step, fraction):
    x = np.array([x])
    start_value = MVE_network.f(x)
    MVE_network.train_on(x_new=x, X_train=X_train, Y_train=Y_train,
                                          step=step, n_epochs=n_epochs, fraction=fraction)
    MVE_network.train_on(x_new=x, X_train=X_train, Y_train=Y_train,
                                          step=step, positive=0, n_epochs=n_epochs, fraction=fraction)
    max_value = MVE_network.f_perturbed(x, positive=True)
    min_value = MVE_network.f_perturbed(x, positive=False)
    # perturbed_predictions_positive = np.maximum(alternative_network_positive.f(X_train),
    #                                             MVE_network.f(X_train))
    # perturbed_predictions_negative = np.minimum(alternative_network_negative.f(X_train),
    #                                              MVE_network.f(X_train))
    perturbed_predictions_positive = MVE_network.f_perturbed(X_train, positive=True)
    perturbed_predictions_negative = MVE_network.f_perturbed(X_train, positive=False)
    # perturbed_predictions_positive = threshold_perturbation(perturbed_predictions_positive, mu_hats, 0.001*(max_value - start_value))
    # perturbed_predictions_negative = threshold_perturbation(perturbed_predictions_negative, mu_hats, 0.001*(start_value - min_value))
    # t_fraction = 0.05
    # threshold_1 = t_fraction * (max_value - start_value)
    # threshold_2 = t_fraction * (min_value - start_value)
    # positive_perturbations = get_perturbation(perturbed_predictions_positive, mu_hats, threshold_1)
    # negative_perturbation = get_perturbation(perturbed_predictions_negative, mu_hats, threshold_2)
    Z_values = (Y_train - mu_hats) / sigma_hats

    upperbound = start_value
    lowerbound = start_value
    accepting = True
    l = 1 / n_steps
    while accepting:
        mu_tildes = l * perturbed_predictions_positive + (1-l)*mu_hats
        # mu_tildes = mu_hats + l*positive_perturbations
        # mu_tilde = mu_hats + l * (perturbed_predictions_positive - mu_hats)
        # C_values = (mu_tildes - mu_hats) / sigma_hats
        # accepting = accept(Z_values, C_values, alpha)
        accepting = accept_LR(Y_train, mu_tildes, mu_hats, sigma_hats, alpha)
        if accepting:
            upperbound = start_value * (1-l) + l * max_value
            # upperbound = start_value + l * (1-t_fraction) / t_fraction * threshold_1
            l += 1 / n_steps

    accepting = True
    l = 1 / n_steps
    while accepting:
        mu_tildes = l * perturbed_predictions_negative + (1 - l) * mu_hats
        # mu_tildes = mu_hats + l * negative_perturbation
        # C_values = (mu_tildes - mu_hats) / sigma_hats
        # accepting = accept(Z_values, C_values, alpha)
        accepting = accept_LR(Y_train, mu_tildes, mu_hats, sigma_hats, alpha)
        if accepting:
            lowerbound = start_value * (1-l) + l * min_value
            # lowerbound = start_value + l * (1-t_fraction) / t_fraction * threshold_2
            l += 1 / n_steps
            # if l >= 1:
            #     break
    return lowerbound, upperbound


def accept(Z_values, C_values, alpha):
    T_values = -C_values * Z_values
    C = np.sum(C_values**2)
    T = np.sum(T_values) + C
    z = scipy.stats.norm.ppf(1-alpha/2)
    if C == 0:
        return 1
    if T < z * np.sqrt(C):
        return 1
    else:
        return 0

def accept_LR(Y, mu_0, mu_1, sigma, alpha):
    """Accept if the loglikelihoodratio < log(alpha)"""
    ratio = LR(Y, mu_0, mu_1, sigma)
    if ratio < np.log(alpha):
        return 0
    else:
        return 1

def threshold_perturbation(new_values, old_values, threshold):
    for i in range(len(old_values)):
        if np.abs(new_values[i] - old_values[i]) < threshold:
            new_values[i] = old_values[i]
    return new_values

def get_perturbation(new_values, old_values, threshold):
    sign = np.sign(threshold)
    if sign == 1:
        for i in range(len(old_values)):
            if new_values[i] - old_values[i] < threshold:
                new_values[i] = 0
            else:
                new_values[i] = new_values[i] - old_values[i] - threshold
    if sign == -1:
        for i in range(len(old_values)):
            if new_values[i] - old_values[i] > threshold:
                new_values[i] = 0
            else:
                new_values[i] = new_values[i] - old_values[i] - threshold
    return new_values

def LR(y, mu_0, mu_hat, sigma):
    "Return the log of the likelihood ratio"
    return -(- 0.5 * np.sum(((y - mu_hat) / sigma)**2) + 0.5 * np.sum(((y - mu_0) / sigma)**2))