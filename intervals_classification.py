import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from utils import sigmoid



def CI_classificationx(*, model, x, X_train, Y_train, p_hats,
                       n_steps, alpha, n_epochs, fraction, weight,
                       compile=False, optimizer='Adam', verbose=True, from_sigmoid=True):
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


    N = len(X_train)
    N_extra = int(N * fraction)
    x = np.reshape(x, np.shape(X_train[0]))
    x_multiple_copies = np.array([x for _ in range(N_extra)])
    positive_labels = np.array([1 for _ in range(N_extra)])
    negative_labels = np.array([0 for _ in range(N_extra)])
    probs = tf.math.sigmoid(p_hats)[:, 0].numpy()
    # probs = [sigmoid(p)[0] for p in p_hats]
    # We replace the targets by the predictions of the model to prevent overfitting
    Y_train_positive = np.hstack((probs, positive_labels))
    Y_train_negative = np.hstack((probs, negative_labels))
    sample_weights = np.hstack((np.ones(len(probs)), weight / N_extra * np.ones(N_extra)))
    X_train_new = np.vstack((X_train, x_multiple_copies))
    alternative_network_positive.fit(X_train_new, Y_train_positive, epochs=n_epochs,
                                     verbose=verbose, batch_size=int(32*(1+fraction)),
                                     sample_weight=sample_weights)
    alternative_network_negative.fit(X_train_new, Y_train_negative, epochs=n_epochs,
                                     verbose=verbose, batch_size=int(32*(1+fraction)),
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
    l = 1 / n_steps

    # The positive direction (probabilities closer to 1)
    while accepting:
        if start_value > max_value:
            break
        p_tildes = l * perturbed_predictions_positive + (1-l) * p_hats
        accepting = accept_LR_classification(Y_train_reshaped, p_tildes, p_hats, alpha,
                                             from_sigmoid=from_sigmoid)
        if accepting:
            # Update the upperbound
            upperbound = start_value * (1-l) + l * max_value

            # A stop criterium when the upper or lowerbound reaches 1 or 0.
            if l > 1:
                if 0.999 < sigmoid(upperbound):
                    accepting = 0
            l += 1 / n_steps

    # The negative direction
    accepting = True
    l = 1 / n_steps
    while accepting:
        if start_value < min_value:
            break
        p_tildes = l * perturbed_predictions_negative + (1 - l) * p_hats
        accepting = accept_LR_classification(Y_train_reshaped, p_tildes, p_hats, alpha,
                                             from_sigmoid=from_sigmoid)
        if accepting:
            lowerbound = start_value * (1-l) + l * min_value
            # A stop criterium when the upper or lowerbound reaches 1 or 0.
            if l>1:
                if sigmoid(lowerbound) < 0.0001:
                    accepting = 0
            l += 1 / n_steps

    # Deleting the new networks to alleviate some memory issues
    if from_sigmoid:
        lowerbound = sigmoid(lowerbound)
        upperbound = sigmoid(upperbound)
    return lowerbound[0][0], upperbound[0][0]


def accept_LR_classification(Y_train, p_tildes, p_hat, alpha):
    log_likelihood_difference = loglikelihood(Y_train, p_tildes) - loglikelihood(Y_train, p_hat)
    if log_likelihood_difference < np.log(alpha):
        return 0
    else:
        return 1


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                         reduction=tf.keras.losses.Reduction.NONE,
                                         label_smoothing=0.00)
def loglikelihood(y, p):
    loglik = -bce(y, p)
    return np.sum(loglik)

