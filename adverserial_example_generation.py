import tensorflow as tf
import numpy as np


def generate_adversarial_example(model, x, y, epsilon):
    """Create adversarial example using FGSM.

    The bulk of this small function was written by chatGPT3.5.
    Arguments:
        model: the model
        x: The input for which we want an adversarial example
        y: The true label of the input
        epsilon: The strength of the perturbation

    Return:
        An adversarial example
    """
    x = tf.keras.backend.variable(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = bce(np.array([y]), y_pred)
    grads = tape.gradient(loss, x)
    signed_grads = tf.sign(grads)
    adversarial_example = x + epsilon * signed_grads
    return adversarial_example.numpy()
