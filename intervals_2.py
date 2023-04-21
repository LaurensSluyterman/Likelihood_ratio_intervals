import numpy as np
import gc


def CI_NN(MVE_network, X, X_train, Y_train, alpha, n_steps=10,
          n_epochs=40, step=1, fraction=0.1):
    """Calculate a confidence interval for all x values in X.

    This function is basically a wrapper function that calls
    CI_NNx a number of times in order to save some duplicate computations.
    """
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
        CI[j, 0] = lowerbound
        CI[j, 1] = upperbound
    gc.collect()
    return CI


def CI_NNx(*, MVE_network, x, X_train, Y_train, mu_hats, sigma_hats,
           n_steps, alpha, n_epochs, step, fraction):
    # Step 1: Let the network train more to reach different values at x.
    x = np.array([x])
    start_value = MVE_network.f(x)
    MVE_network.train_on(x_new=x, X_train=X_train, Y_train=Y_train,
                         step=step, n_epochs=n_epochs, fraction=fraction)
    MVE_network.train_on(x_new=x, X_train=X_train, Y_train=Y_train,
                         step=step, positive=0, n_epochs=n_epochs, fraction=fraction)
    max_value = MVE_network.f_perturbed(x, positive=True)
    min_value = MVE_network.f_perturbed(x, positive=False)

    # The predictions of the two perturbed networks at the training locations.
    perturbed_predictions_positive = MVE_network.f_perturbed(X_train, positive=True)
    perturbed_predictions_negative = MVE_network.f_perturbed(X_train, positive=False)

    # Step 2: Check which linear combination of original and perturb explain the data well.
    upperbound = start_value
    lowerbound = start_value
    accepting = True
    l = 1 / n_steps
    while accepting:
        if start_value > max_value:
            break
        # Create a new alternative
        mu_tildes = l * perturbed_predictions_positive + (1-l)*mu_hats
        # Check if we can accept this
        accepting = accept(Y_train, mu_tildes, mu_hats, sigma_hats, alpha)
        # Update the upperbound if we accept
        if accepting:
            upperbound = start_value * (1-l) + l * max_value
            l += 1 / n_steps

    accepting = True
    l = 1 / n_steps
    while accepting:
        if start_value < min_value:
            break
        # Create a new alternative
        mu_tildes = l * perturbed_predictions_negative + (1 - l) * mu_hats
        # Check if we can accept this
        accepting = accept(Y_train, mu_tildes, mu_hats, sigma_hats, alpha)
        # Update the upperbound if we accept
        if accepting:
            lowerbound = start_value * (1-l) + l * min_value
            l += 1 / n_steps
    return lowerbound, upperbound


def accept(Y, mu_0, mu_1, sigma, alpha):
    """Accept if the loglikelihoodratio < log(alpha)"""
    ratio = loglikelihoodratio(Y, mu_0, mu_1, sigma)
    if ratio < np.log(alpha):
        return 0
    else:
        return 1


def loglikelihoodratio(y, mu_0, mu_hat, sigma):
    """Return the log of the likelihood ratio"""
    return -(- 0.5 * np.sum(((y - mu_hat) / sigma)**2) + 0.5 * np.sum(((y - mu_0) / sigma)**2))
