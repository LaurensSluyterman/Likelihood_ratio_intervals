import tensorflow as tf
import numpy as np
import logging
import scipy

tf.get_logger().setLevel('ERROR')

@tf.function
def get_gradient(x, model):
    out = model(x)[:, 0]
    gradient = tf.gradients(out, x)
    return gradient

def get_var(x, model, std=1, B=100):
    try:
        x_tensor = tf.constant(x, shape=(1, np.shape(x)[0]), dtype=tf.float32)
    except IndexError:
        x_tensor = tf.constant(x, shape=(1, 1), dtype=tf.float32)
    gradient = get_gradient(x_tensor, model)[0][0].numpy()
    values = np.zeros((B))
    for i in range(B):
        z = np.random.normal(loc=x_tensor, scale=std)
        values[i] = (model.predict(z)[:, 0] - np.sum(z * gradient)) / std**2
    return np.var(values)


#
# vars = [get_var(x, model, std=0.1) for x in X_n[0:80]]
# np.mean(np.sqrt(2 * vars))

# def get_second_derivative_constant(X, model):
#     gradients = []
#     for x in X:
#         try:
#             x = tf.constant(x, shape=(1, np.shape(x)[0]), dtype=tf.float32)
#         except IndexError:
#             x = tf.constant(x, shape=(1, 1), dtype=tf.float32)
#         gradient = get_gradient(x, model)
#         gradients.append(gradient[0][0].numpy())
#
#
#     V = np.zeros((len(X), len(X)))
#     for i, x in enumerate(X):
#         for j, y in enumerate(X):
#             if j != i:
#                 V[i, j] = root_sum_squared(gradients[i] - gradients[j]) / (root_sum_squared(x - y) + 1e-3)
#     return np.mean(np.max(V, axis=1))


def get_second_derivative_constant(X, model):
    gradients = []
    for x in X:
        try:
            n_dim = np.shape(x)[0]
            x_tensor = tf.constant(x, shape=(1, n_dim), dtype=tf.float32)
        except IndexError:
            x_tensor = tf.constant(x, shape=(1, 1), dtype=tf.float32)
            n_dim = 1
        gradient = get_gradient(x_tensor, model)
        gradients.append(gradient[0][0].numpy())
    Ds = [np.sqrt(get_var(x, model, std=0.1) * 2 / n_dim) for x in X]
    return np.mean(Ds)

def get_rho(X, model):
    gradients = []
    for x in X:
        try:
            n_dim = np.shape(x)[0]
            x_tensor = tf.constant(x, shape=(1, n_dim), dtype=tf.float32)
        except IndexError:
            x_tensor = tf.constant(x, shape=(1, 1), dtype=tf.float32)
        gradient = get_gradient(x_tensor, model)
        gradients.append(gradient[0][0].numpy())
    vars = [get_var(x, model, std=0.1) for x in X]
    return 2 * np.mean(vars)


def CI_coverage_probability(intervals, true_values):
    """This function checks if the intervals contain the true values."""
    lower_correct = intervals[:, 0] < true_values
    upper_correct = intervals[:, 1] > true_values
    intervals_correct = lower_correct * upper_correct
    return intervals_correct


def normalize(x, mean=None, std=None):
    """This function normalizes x using a given mean and standard deviation"""
    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
    return (x - mean) / std


def reverse_normalized(x_normalized, mean, std):
    """This function reverses the normalization done by the function 'normalize' """
    return x_normalized * std + mean

def root_sum_squared(x):
    return np.sqrt(np.sum(x**2))

def root_mean_squared(x):
    return np.sqrt(np.mean(x**2))


def load_data(directory):
    """Load data from given directory"""
    _DATA_FILE = "./UCI_Datasets/" + directory + "/data/data.txt"
    _INDEX_FEATURES_FILE = "./UCI_Datasets/" + directory + "/data/index_features.txt"
    _INDEX_TARGET_FILE = "./UCI_Datasets/" + directory + "/data/index_target.txt"
    index_features = np.loadtxt(_INDEX_FEATURES_FILE)
    index_target = np.loadtxt(_INDEX_TARGET_FILE)
    data = np.loadtxt(_DATA_FILE)
    X = data[:, [int(i) for i in index_features.tolist()]]
    Y = data[:, int(index_target.tolist())]
    return X, Y

def CI_coverage_probability(intervals, true_values):
    """This function checks if the intervals contain the true values."""
    lower_correct = intervals[:, 0] < true_values
    upper_correct = intervals[:, 1] > true_values
    intervals_correct = lower_correct * upper_correct
    return intervals_correct


def get_second_derivative_matrix(model, x, B=50):
    try:
        x_tensor = tf.constant(x, shape=(1, np.shape(x)[0]), dtype=tf.float32)
        d = np.shape(x)[0]
    except IndexError:
        x_tensor = tf.constant(x, shape=(1, 1), dtype=tf.float32)
    gradient = get_gradient(x_tensor, model)[0][0].numpy()
    n_d = np.int(d * (d + 1) / 2)
    B = np.int(B * d)
    covariates_matrix = np.zeros((B, n_d))
    targets_matrix = np.zeros((B, 1))
    for b in range(B):
        m = np.zeros(n_d)
        epsilon = np.random.normal(0, 0.1, size=d)
        count = 0
        for i in range(d):
            for j in range(i, d):
                if i == j:
                    m[count] = epsilon[i] * epsilon[j]
                else:
                    m[count] = 2 * epsilon[i] * epsilon[j]
                count += 1
        z = x_tensor + epsilon
        y = 2 * (model.predict(z.numpy())[:, 0] - model.predict(x_tensor.numpy())[:, 0] - np.sum(epsilon * gradient))
        covariates_matrix[b] = m
        targets_matrix[b] = y

    beta_hat = np.linalg.inv(np.transpose(covariates_matrix) @ covariates_matrix) \
                @ np.transpose(covariates_matrix) \
                @ targets_matrix

    count = 0
    matrix = np.zeros((d,d))
    for i in range(d):
        for j in range(i, d):
            matrix[i, j] = beta_hat[count]
            matrix[j, i] = beta_hat[count]
            count += 1
    return matrix



def get_E(A, norm):
    eigen_values, eigen_vectors = np.linalg.eigh(A)
    c = np.sqrt(norm / np.sum(eigen_values**2))
    D_matrix = np.zeros((len(eigen_values), len(eigen_values)))
    for i, value in enumerate(eigen_values):
        D_matrix[i, i] = c * np.abs(value)
    E = eigen_vectors @ D_matrix @ np.linalg.inv(eigen_vectors)
    return E

