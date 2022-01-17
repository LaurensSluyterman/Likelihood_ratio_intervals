import tensorflow as tf
import numpy as np
import logging

tf.get_logger().setLevel('ERROR')

@tf.function
def get_gradient(x, model):
    a = tf.constant(x, shape=(1, 1), dtype=float)
    loss = model(a)[:, 0]
    gradient = tf.gradients(loss, a)
    return gradient


def get_second_derivative_constant(X, model):
    gradients = []
    for x in X:
        gradient = get_gradient(x, model)
        gradients.append(gradient[0][0].numpy()[0])

    V = np.zeros((len(X), len(X)))
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            if j != i:
                V[i, j] = np.abs(gradients[i] - gradients[j]) / (np.abs(x - y))
    return np.mean(np.max(V, axis=1))


def CI_coverage_probability(intervals, true_values):
    """This function checks if the intervals contain the true values."""
    lower_correct = intervals[:, 0] < true_values
    upper_correct = intervals[:, 1] > true_values
    intervals_correct = lower_correct * upper_correct
    return intervals_correct

