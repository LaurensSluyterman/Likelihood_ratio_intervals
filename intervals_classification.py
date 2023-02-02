
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from utils import sigmoid



def CI_classificationx(*, model, x, X_train, Y_train, p_hats,
                       n_steps, alpha, n_epochs, fraction,
                       verbose=True, from_sigmoid=True):
    # Copying the model
    model.save('./models/tempmodel.h5', overwrite=True)
    alternative_network_positive = tf.keras.models.load_model('./models/tempmodel.h5')
    alternative_network_negative = tf.keras.models.load_model('./models/tempmodel.h5')

    # Letting the networks train on to get close to a 0 and 1 prediction
    x = np.array([x])
    start_value = model.predict(x)
    if not from_sigmoid:
        start_value = sigmoid(model.predict(x))
    N = len(X_train)
    x = np.reshape(x, np.shape(X_train[0]))
    x_multiple_copies = np.array([x for _ in range(np.int(N * fraction))])
    positive_labels = np.array([1 for _ in range(np.int(N * fraction))])
    negative_labels = np.array([0 for _ in range(np.int(N * fraction))])
    probs = tf.math.sigmoid(p_hats)[:, 0].numpy()
    # We repeat the targets by the predictions of the model to prevent overfitting
    Y_train_positive = np.hstack((probs, positive_labels))
    Y_train_negative = np.hstack((probs, negative_labels))
    X_train_new = np.vstack((X_train, x_multiple_copies))
    alternative_network_positive.fit(X_train_new, Y_train_positive, epochs=n_epochs,
                                     verbose=verbose, batch_size=None)
    alternative_network_negative.fit(X_train_new, Y_train_negative, epochs=n_epochs,
                                     verbose=verbose, batch_size=None)

    # The maximum and minimum achieved probability predictions
    max_value = alternative_network_positive.predict(np.array([x]))
    min_value = alternative_network_negative.predict(np.array([x]))

    perturbed_predictions_positive = alternative_network_positive.predict(X_train)
    perturbed_predictions_negative = alternative_network_negative.predict(X_train)

    # If interpolating after the sigmoid, we transform everything to probabilities
    if not from_sigmoid:
        perturbed_predictions_positive = tf.math.sigmoid(perturbed_predictions_positive)[:, 0].numpy()
        perturbed_predictions_negative = tf.math.sigmoid(perturbed_predictions_negative)[:, 0].numpy()
        max_value = sigmoid(alternative_network_positive.predict(np.array([x])))
        min_value = sigmoid(alternative_network_negative.predict(np.array([x])))
        p_hats = probs

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
                if 0.999 < sigmoid(upperbound) or sigmoid(upperbound) < 0.0001:
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
                if 0.999 < sigmoid(lowerbound) or sigmoid(lowerbound) < 0.0001:
                    accepting = 0
            l += 1 / n_steps

    # Deleting the new networks to alleviate some memory issues
    del alternative_network_positive, alternative_network_negative
    if from_sigmoid:
        lowerbound = sigmoid(lowerbound)
        upperbound = sigmoid(upperbound)
    return lowerbound[0][0], upperbound[0][0]


def accept_LR_classification(Y_train, p_tildes, p_hat, alpha, from_sigmoid):
    log_likelihood_difference = loglikelihood(Y_train, p_tildes, from_sigmoid) - loglikelihood(Y_train, p_hat, from_sigmoid)
    if log_likelihood_difference < np.log(alpha):
        return 0
    else:
        return 1


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                         reduction=tf.keras.losses.Reduction.NONE,
                                         label_smoothing=0)
def loglikelihood(y, p, from_sigmoid=True):
    loglik = -bce(y, p)
    return np.sum(loglik)




# np.exp(loglikelihood(Y_train_reshaped, p_tildes)
#                                - loglikelihood(Y_train_reshaped, p_hats))
#
# sigmoid(upperbound)
# sigmoid(start_value)
#
# plt.plot(history.history['accuracy'])
# plt.show()
#
#
# [bce(Y_train_reshaped, l/10 * perturbed_predictions_positive + (1-l/10)*p_hats)[20].numpy() for l in range(0, 11)]
# [loglikelihood(Y_train_reshaped, l/10 * perturbed_predictions_negative + (1-l/10)*p_hats) for l in range(0, 11)]
# [loglikelihood(Y_train_reshaped, l/10 * perturbed_predictions_positive + (1-l/10)*p_hats) for l in range(0, 11)]
#
# y_true = [0, 1, 0, 0]
# y_pred = [-18.6, 0.51, 2.94, -12.8]
# bce(y_true, y_pred).numpy()
#
# model.evaluate(X_train, Y_train)
# def loglikelihood(y, p, from_sigmoid=False):
#     ll = 0
#     for i, label in enumerate(y):
#         prob = p[i]
#         if from_sigmoid:
#             prob = sigmoid(prob)
#         if label == 0:
#             ll += np.log(1-p)
#         if label == 1:
#             ll += np.log(p)
#     return ll