import tensorflow as tf
import numpy as np
import scipy
from utils import sigmoid
from scipy import stats


def CI_classificationx(*, model, x, X_train, Y_train, predicted_logits,
                       n_steps, alpha, n_epochs, fraction, weight=1,
                       compile=False, optimizer='Adam', verbose=True,
                       batch_size=32):
    """Create a confidence interval for a binary classification problem.

    Arguments:
        model: The original model trained on (X_train, Y_train).
        x: The new input for which a CI is constructed.
        X_train: The training features.
        Y_train: The training targets.
        predicted_logits: The predictions of the model on the training data.
        n_steps (int): Determines the step size when creating the CI.
            if the predicted logit is 0.8 and 0.99 after retraining, then
            the step size will be 0.19/100.
        alpha (float): Determines the confidence level of the CI. An alpha of
            0.1 corresponds to a 90% CI.
        n_epochs (int): The number of training epochs
        fraction (float): Determines how many new data points are added to train
            the perturbed networks. If this is set to 1/16 and the training data
            set has 2000 data points, then 2000/16 points are added.
        weight (float): The weight that these newly added points get in the
            loss contribution. If set to 1, the sum of the contributions
            of the newly added points adds to 1. If 15 points are added, each
            loss contribution will in this case be divided by 15.
        compile (Boolean): Determines if we recompile the network before
            retraining to find the perturbed networks.
        optimizer: The optimizer of the network
        verbose (Boolean): Determines if the training progress is displayed.
        batch_size (int): The batch size during training.

    Return:
        lowerbound, upperbound: The lower- and upperbound of the CI.
    """
    # Copying the model
    model.save('./models/tempmodel.h5', overwrite=True)
    alternative_network_positive = tf.keras.models.load_model('./models/tempmodel.h5')
    alternative_network_negative = tf.keras.models.load_model('./models/tempmodel.h5')

    # Recompile the model if desired (to reset the optimizer state)
    if compile:
        alternative_network_positive.compile(optimizer=optimizer,
                                             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                             metrics=['accuracy'])
        alternative_network_negative.compile(optimizer=optimizer,
                                             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                             metrics=['accuracy'])

    # Letting the networks train on to get close to a 0 and 1 prediction
    x = np.array([x])
    start_value = model.predict(x, verbose=0)

    # Create the modified data set
    N = len(X_train)
    N_extra = int(N * fraction)
    x = np.reshape(x, np.shape(X_train[0]))
    x_multiple_copies = np.array([x for _ in range(N_extra)])
    positive_labels = np.array([1 for _ in range(N_extra)])
    negative_labels = np.array([0 for _ in range(N_extra)])
    probs = [sigmoid(p)[0] for p in predicted_logits]
    Y_train_positive = np.hstack((probs, positive_labels))
    Y_train_negative = np.hstack((probs, negative_labels))
    sample_weights = np.hstack((np.ones(len(probs)), weight / N_extra * np.ones(N_extra)))
    X_train_new = np.vstack((X_train, x_multiple_copies))

    # Train the two perturbed networks
    alternative_network_positive.fit(X_train_new, Y_train_positive, epochs=n_epochs,
                                     verbose=verbose, batch_size=int(batch_size*(1+fraction)),
                                     sample_weight=sample_weights)
    alternative_network_negative.fit(X_train_new, Y_train_negative, epochs=n_epochs,
                                     verbose=verbose, batch_size=int(batch_size*(1+fraction)),
                                     sample_weight=sample_weights)

    # The maximum and minimum achieved probability predictions
    max_value = alternative_network_positive.predict(np.array([x]), verbose=0)
    min_value = alternative_network_negative.predict(np.array([x]), verbose=0)

    perturbed_predictions_positive = alternative_network_positive.predict(X_train, verbose=0)
    perturbed_predictions_negative = alternative_network_negative.predict(X_train, verbose=0)

    # Checking which alternatives still get accepted
    Y_train_reshaped = np.reshape(Y_train, (len(Y_train), 1))
    upperbound = start_value
    lowerbound = start_value
    accepting = True
    l = 0
    sign = np.sign(max_value - start_value)
    # The positive direction (probabilities closer to 1)
    while accepting:
        l += sign / n_steps
        perturbed_logits = l * perturbed_predictions_positive + (1 - l) * predicted_logits
        accepting = accept_LR_classification(Y_train_reshaped, perturbed_logits, predicted_logits, alpha)
        if accepting:
            # Update the upperbound
            upperbound = start_value * (1-l) + l * max_value
            # A stop criterium when the upperbound reaches 1.
            if np.abs(l) > 1:
                if 0.999 < sigmoid(upperbound):
                    accepting = 0

    # The negative direction
    accepting = True
    l = 0
    sign = np.sign(start_value - min_value)
    while accepting:
        l += sign / n_steps
        perturbed_logits = l * perturbed_predictions_negative + (1 - l) * predicted_logits
        accepting = accept_LR_classification(Y_train_reshaped, perturbed_logits, predicted_logits, alpha)
        if accepting:
            lowerbound = start_value * (1-l) + l * min_value
            # A stop criterium when the upper or lowerbound reaches 0.
            if np.abs(l) > 1:
                if sigmoid(lowerbound) < 0.0001:
                    accepting = 0

    # Transforming the logits to probabilities
    upperbound = sigmoid(upperbound)
    lowerbound = sigmoid(lowerbound)
    return lowerbound[0][0], upperbound[0][0]


def accept_LR_classification(Y_train, perturbed_logits, predicted_logits, alpha):
    """Check if the likelihood ratio exceeds the critical value."""
    log_likelihood_difference = loglikelihood(Y_train, predicted_logits) \
                                - loglikelihood(Y_train, perturbed_logits)
    critical_value = stats.chi2(1).ppf(1-alpha)
    if 2*log_likelihood_difference > critical_value:
        return 0
    else:
        return 1


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                         reduction=tf.keras.losses.Reduction.NONE,
                                         label_smoothing=0.00)


def loglikelihood(y, logits):
    """Calculate the loglikelihood.

    Arguments:
        y: The labels
        logits: The class logits
    Returns:
        The loglikelihood
    """
    loglik = -bce(y, logits)
    return np.sum(loglik)


def CI_ensemble(ensemble, x, alpha=0.05, give_prediction=True):
    """Construct a CI using an ensemble.

    Arguments:
        ensemble: A list containing the ensemble members
        x: The input for which a CI is constructed.
        B (int): The number of forward passes that is used.
        alpha (float): The confidence level of the CI. An alpha of 0.05 corresponds
            to a 95% CI.
        give_prediction (Boolean): Determines if the average of the predictions
            is also returned.
    Return:
        prediction: The average of the predictions of the ensemble members
        lowerbound, upperbound: The lower- and upperbound of the CI.
    """
    predictions = [sigmoid(model.predict(np.array([x]), verbose=0)) for model in ensemble]
    var = np.var(predictions)
    prediction = np.mean(predictions)
    t = scipy.stats.t(df=len(ensemble)-1).ppf(1 - alpha / 2)
    lowerbound = np.max((prediction - t * np.sqrt(var), 0))
    upperbound = np.min((1, prediction + t * np.sqrt(var)))
    if give_prediction:
        return prediction, [lowerbound, upperbound]
    else:
        return [lowerbound, upperbound]


def CI_dropout(dropout_model, x, B=300, alpha=0.05, give_prediction=True):
    """Construct a CI using MC-dropout.

    Arguments:
        dropout_model: A model containing dropout in the dense layers. Dropout
            must remain active after training.
        x: The input for which a CI is constructed.
        B (int): The number of forward passes that is used.
        alpha (float): The confidence level of the CI. An alpha of 0.05 corresponds
            to a 95% CI.
        give_prediction (Boolean): Determines if the average of the predictions
            is also returned.
    Return:
        prediction: The average of the forward passes
        lowerbound, upperbound: The lower- and upperbound of the CI.
    """
    predictions = [sigmoid(dropout_model(np.array([x]), training=1)) for _ in range(B)]
    prediction = np.mean(predictions)
    lowerbound = np.percentile(predictions, 100*(alpha/2))
    upperbound = np.percentile(predictions, 100*(1-alpha/2))
    if give_prediction:
        return prediction, [lowerbound, upperbound]
    else:
        return [lowerbound, upperbound]
