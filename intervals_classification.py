
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential



def CI_classificationx(*, model, x, X_train, Y_train, p_hats,
                       n_steps, alpha, n_epochs, fraction,
                       verbose=True):
    model.save('./models/tempmodel.h5', overwrite=True)
    alternative_network_positive = tf.keras.models.load_model('./models/tempmodel.h5')
    alternative_network_negative = tf.keras.models.load_model('./models/tempmodel.h5')
    x = np.array([x])
    # start_value = model.predict(x)
    start_value = sigmoid(model.predict(x))
    N = len(X_train)
    x = np.reshape(x, np.shape(X_train[0]))
    x_multiple_copies = np.array([x for _ in range(np.int(N * fraction))])
    positive_labels = np.array([1 for _ in range(np.int(N * fraction))])
    negative_labels = np.array([0 for _ in range(np.int(N * fraction))])
    probs = tf.math.sigmoid(model.predict(X_train))[:, 0].numpy()
    Y_train_positive = np.hstack((probs, positive_labels))
    Y_train_negative = np.hstack((probs, negative_labels))
    X_train_new = np.vstack((X_train, x_multiple_copies))
    alternative_network_positive.fit(X_train_new, Y_train_positive, epochs=n_epochs,
                                     verbose=verbose, batch_size=None)
    alternative_network_negative.fit(X_train_new, Y_train_negative, epochs=n_epochs,
                                     verbose=verbose, batch_size=None)
    max_value = sigmoid(alternative_network_positive.predict(np.array([x])))
    min_value = sigmoid(alternative_network_negative.predict(np.array([x])))

    # perturbed_predictions_positive = alternative_network_positive.predict(X_train)
    # perturbed_predictions_negative = alternative_network_negative.predict(X_train)

    perturbed_predictions_positive = tf.math.sigmoid(alternative_network_positive.predict(X_train))[:, 0].numpy()
    perturbed_predictions_negative = tf.math.sigmoid(alternative_network_negative.predict(X_train))[:, 0].numpy()
    p_hats = probs

    Y_train_reshaped = np.reshape(Y_train, (len(Y_train), 1))
    upperbound = start_value
    lowerbound = start_value
    accepting = True
    l = 1 / n_steps
    #Todo: transform p_tildes and p_hats to probabilities first and change the bce to not from logits
    while accepting:
        p_tildes = l * perturbed_predictions_positive + (1-l) * p_hats
        accepting = accept_LR_classification(Y_train_reshaped, p_tildes, p_hats, alpha)
        if accepting:
            upperbound = start_value * (1-l) + l * max_value
            # if l > 1:
            #     if 0.999 < sigmoid(upperbound) or sigmoid(upperbound) < 0.0001:
            #         accepting = 0
            l += 1 / n_steps
            if l > 1:
                accepting = 0

    accepting = True
    l = 1 / n_steps
    while accepting:
        p_tildes = l * perturbed_predictions_negative + (1 - l) * p_hats
        accepting = accept_LR_classification(Y_train_reshaped, p_tildes, p_hats, alpha)
        if accepting:
            lowerbound = start_value * (1-l) + l * min_value
            # if l>1:
            #     if 0.999 < sigmoid(lowerbound) or sigmoid(lowerbound) < 0.0001:
            #         accepting = 0
            l += 1 / n_steps
            if l>1:
                accepting = 0

    del alternative_network_positive, alternative_network_negative
    return lowerbound[0][0], upperbound[0][0]


def accept_LR_classification(Y_train, p_tildes, p_hat, alpha):
    if min(p_tildes) < 0:
        return 0
    if max(p_tildes) > 1:
        return 0
    log_likelihood_difference = loglikelihood(Y_train, p_tildes) - loglikelihood(Y_train, p_hat)
    if log_likelihood_difference < np.log(alpha):
        return 0
    else:
        return 1


bce = tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                         reduction=tf.keras.losses.Reduction.NONE,
                                         label_smoothing=0)
# def loglikelihood(y, p):
#     loglik = -bce(y, p)
#     return np.sum(loglik)


def sigmoid(x):
    if x > 0:
        return np.exp(x) / (1 + np.exp(x))
    else:
        return 1 / (1 + np.exp(-x))


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
def loglikelihood(y, p):
    ll = 0
    for i, label in enumerate(y):
        if label == 0:
            ll += np.log(1-(p[i]))
        if label == 1:
            ll += np.log((p[i]))
    return ll