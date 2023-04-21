import tensorflow as tf
import numpy as np
from utils import sigmoid


def CI_classificationx(*, model, x, X_train, Y_train, predicted_logits,
                       n_steps, alpha, n_epochs, fraction, weight=1,
                       compile=False, optimizer='Adam', verbose=True,
                       batch_size=32):
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
    log_likelihood_difference = loglikelihood(Y_train, perturbed_logits) \
                                - loglikelihood(Y_train, predicted_logits)
    if log_likelihood_difference < np.log(alpha):
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
