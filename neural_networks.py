import keras.models
import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from utils import normalize, reverse_normalized
from keras.layers import Input, Dense
from keras.models import Model

from intervals import CI_intervals
from second_derivative_utils import get_rho
l2 = keras.regularizers.l2

class LikelihoodNetwork:
    """
    This class represents a trained neural network.

    The networks are trained using the negative loglikelihood as loss function
    and output an estimate for the mean, f, and standard deviation, sigma.

    Attributes:
        model: The trained neural network

    Methods:
        f: An estimate of the mean function, without any normalisation .
        sigma: An estimate of the standard deviation, without any normalisation.
    """

    def __init__(self, X, Y, n_hidden, n_epochs, save_epoch=0,
                 reg=True, batch_size=None, verbose=False, normalization=True,
                 get_rho_constant=True):
        """
        Arguments:
            X: The unnormalized training covariates.
            Y: The unnormalized training targets
            n_hidden (array): An array containing the number of hidden units for
                each hidden layer.
            n_epochs (int): The number of training epochs
            name: The name the network is saved as after save_epoch number of
                training epochs
            save_epoch: The number of training epochs after which the model
                is saved.
            reg: The regularisation constant. If set to False,
                no regularisation is used.
            batch_size (int): The used batch size for training, if set to None
                the standard size of 32 is used.
            verbose (bool): Determines if training progress is printed.
        """
        self._normalization = normalization
        assert save_epoch < n_epochs
        if normalization is True:
            self._X_mean = np.mean(X, axis=0)
            self._X_std = np.std(X, axis=0)
            self._Y_mean = np.mean(Y, axis=0)
            self._Y_std = np.std(Y, axis=0)
            X = normalize(X)
            Y = normalize(Y)
        model = train_network(X, Y, n_hidden, loss=negative_log_likelihood, n_epochs=n_epochs,
                              reg=reg, batch_size=batch_size,
                              verbose=verbose)
        try:
            self.covariance_matrix_inv = np.linalg.inv(np.cov(X, rowvar=False))
        except:
            self.covariance_matrix_inv = [1 / np.var(X)]
        self.model = model
        if get_rho_constant:
            rho = get_rho(X, self.model)
            self.rho = rho


    def f(self, X_test):
        """Return the mean prediction without any regularisation"""
        if self._normalization is True:
            X_test = normalize(X_test, self._X_mean, self._X_std)
            predictions = self.model.predict(X_test)[:, 0]
            return reverse_normalized(predictions, self._Y_mean, self._Y_std)
        else:
            return self.model.predict(X_test)[:, 0]


    def sigma(self, X_test):
        """Return the standard deviation prediction without any regularisation."""
        if self._normalization is True:
            X_test = normalize(X_test, self._X_mean, self._X_std)
            predictions = K.exp(self.model.predict(X_test)[:, 1]) + 1e-3
            return predictions * self._Y_std
        else:
            return K.exp(self.model.predict(X_test)[:, 1]) + 1e-3


    def CI(self, X_test, X_train, Y_train, alpha, rho=None):
        if rho is None:
            rho = self.rho
        if self._normalization:
            X_test = normalize(X_test, self._X_mean, self._X_std)
            X_train = normalize(X_train, self._X_mean, self._X_std)
            Y_train = normalize(Y_train, self._Y_mean, self._Y_std)
        CI = CI_intervals(self.model, X_test, X_train, Y_train, alpha, rho)
        if self._normalization:
            CI[:, 0] = reverse_normalized(CI[:, 0], self._Y_mean, self._Y_std)
            CI[:, 1] = reverse_normalized(CI[:, 1], self._Y_mean, self._Y_std)
        return CI


def train_network(X_train, Y_train, n_hidden, n_epochs, loss, reg=True,
                  batch_size=None, verbose=False):
    """Train a network that outputs the mean and standard deviation.

    This function trains a network that outputs the mean and standard
    deviation. The network is trained using the negative loglikelihood
    of a normal distribution as the loss function.

    Parameters:
            X_train: A matrix containing the inputs of the training data.
            Y_train: A matrix containing the targets of the training data.
            n_hidden (array): An array containing the number of hidden units
                     for each hidden layer. The length of this array
                     specifies the number of hidden layers used for the
                     training of the main model.
            n_epochs: The amount of epochs used in training.
            reg: The regularisation that is used. If it is set to True,
                the standard of 1 / len(X) is used. If it is set to a float,
                then that value is used.
            batch_size: The batch-size used during training
            verbose (boolean): A boolean that determines if the training-
                    information is displayed.

    Returns:
        model: A trained network that outputs a mean and log of standard
            deviation.
    """
    try:
        input_shape = (np.shape(X_train)[1],)
    except IndexError:
        input_shape = (1,)
    if reg is True:
        c = 1 / len(Y_train)
    else:
        c = reg
    inputs = Input(shape=input_shape)
    inter = Dense(n_hidden[0], activation='relu',
                  kernel_regularizer=l2(c))(inputs)
    for i in range(len(n_hidden) - 1):
        inter = Dense(n_hidden[i + 1], activation='relu',
                      kernel_regularizer=keras.regularizers.l2(c))(inter)
    outputs = Dense(2, activation='linear')(inter)
    model = Model(inputs, outputs)
    model.compile(loss=loss, optimizer='adam')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=n_epochs,
              verbose=verbose)
    return model

def negative_log_likelihood(targets, outputs):
    """Calculate the negative loglikelihood."""
    mu = outputs[..., 0:1]
    sigma = K.exp(outputs[..., 1:2]) + 1e-3
    y = targets[..., 0:1]
    loglik = - K.log(sigma) - 0.5 * K.square((y - mu) / sigma)
    return - loglik
