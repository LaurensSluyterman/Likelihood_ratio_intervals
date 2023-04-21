import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
import tensorflow as tf
import gc
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from intervals_classification import CI_classificationx
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from utils import sigmoid
plt.rcParams['text.usetex'] = True
plt.rcParams["font.size"] = 17
plt.rcParams['axes.linewidth'] = 1


#%% Generating the data
x, y = make_moons(n_samples=100, random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
mean = np.mean(x_train, axis=0)
stdev = np.std(x_train, axis=0)
x_train_norm = (x_train - mean) / stdev


#%% Creating and training a model
def get_model(c_reg, n_epochs, verbose=True):
    """Train a simple network with three hidden layers"""
    model = Sequential()
    model.add(Dense(30, activation='elu', input_shape=(2, ),
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dense(1,
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x=x_train_norm, y=y_train, epochs=n_epochs, verbose=verbose)
    return model


model = get_model(c_reg=1e-3, n_epochs=500)

#%% Creating confidence intervals using our approach
x_1 = np.linspace(-2, 3, 30)
x_2 = np.linspace(-1.5, 2, 30)
x_val = [[x1, x2] for x1 in x_1 for x2 in x_2]
CIs = np.zeros((len(x_val), 2))
p_hats = model.predict(x_train_norm, verbose=0)
for i, x in enumerate(x_val):
    print(i+1)
    CIs[i, 0], CIs[i, 1] = CI_classificationx(model=model, x=(x - mean) / stdev,
                                              X_train=x_train_norm,
                                              Y_train=y_train, predicted_logits=p_hats,
                                              n_steps=100, alpha=0.1,
                                              n_epochs=500, fraction=1/16,
                                              verbose=0, weight=1)
    tf.keras.backend.clear_session()
    gc.collect()

#%% Saving the results
with open('./Results/twomoon-1e-3-500epoch-w1.pickle', 'wb') as handle:
    pickle.dump({'confidence_intervals': CIs, 'locations': x_val}, handle)

#%%
with open('./Results/twomoon-1e-3-500epoch-w1.pickle', 'rb') as handle:
    data_dict = pickle.load(handle)
CIs = data_dict['confidence_intervals']
x_val = data_dict['locations']

#%% Visualizing the results
z = CIs[:, 1] - CIs[:, 0]
z = np.reshape(z, (30, 30))
c = plt.pcolormesh(x_1, x_2, np.transpose(z), vmin=0, vmax=1, cmap='viridis')
plt.colorbar(c)
plt.scatter(x_train[np.where(y_train == 1)][:, 0], x_train[np.where(y_train == 1)][:, 1])
plt.scatter(x_train[np.where(y_train == 0)][:, 0], x_train[np.where(y_train == 0)][:, 1])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()

#%% Using an ensemble
ensemble = [get_model(1e-3, 500, verbose=True) for _ in range(10)]

CI_ensemble = np.zeros((len(x_val), 2))
p_hats_ensemble = np.zeros((len(x_val)))
t = scipy.stats.t(df=9).ppf(1-0.1/2)
for i, x in enumerate(x_val):
    print(i)
    predictions = [sigmoid(model.predict(np.array([(x-mean)/stdev]), verbose=0)) for model in ensemble]
    var = np.var(predictions)
    prediction = np.mean(predictions)
    CI_ensemble[i, 0] = prediction - t * np.sqrt(var / len(ensemble))
    CI_ensemble[i, 1] = prediction + t * np.sqrt(var / len(ensemble))
    p_hats_ensemble[i] = prediction

#%% Saving CIs from ensemble approach
with open('./Results/twomoon-ensemble-w1.pickle', 'wb') as handle:
    pickle.dump({'confidence_intervals': CI_ensemble, 'locations': x_val}, handle)

#%% Visualizing the ensemble approach
z = CI_ensemble[:, 1] - CI_ensemble[:, 0]
z = np.reshape(z, (30, 30))
c = plt.pcolormesh(x_1, x_2, np.transpose(z), vmin=0, vmax=np.max(z))
plt.colorbar(c)
plt.scatter(x_train[np.where(y_train == 1)][:, 0], x_train[np.where(y_train == 1)][:, 1])
plt.scatter(x_train[np.where(y_train == 0)][:, 0], x_train[np.where(y_train == 0)][:, 1])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()


#%% Dropout example
def get_model_dropout(c_reg, n_epochs, verbose=True):
    """Train a model with dropout"""
    model = Sequential()
    model.add(Dense(30, activation='elu', input_shape=(2, ),
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dense(1,
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(x=x_train_norm, y=y_train, epochs=n_epochs, verbose=verbose)
    return model


model = get_model_dropout(c_reg=1e-4, n_epochs=500)

B = 1000  # Amount of forward passes
CI_dropout = np.zeros((len(x_val), 2))
p_hats_dropout = np.zeros((len(x_val)))
for i, x in enumerate(x_val):
    print(i)
    predictions = [sigmoid(model(np.array([(x-mean)/stdev]), training=1)) for _ in range(B)]
    var = np.var(predictions)
    prediction = np.mean(predictions)
    CI_dropout[i, 0] = np.percentile(predictions, 5)
    CI_dropout[i, 1] = np.percentile(predictions, 95)
    p_hats_dropout[i] = prediction

#%% Visualize the dropout approach
z = CI_dropout[:, 1] - CI_dropout[:, 0]
z = np.reshape(z, (30, 30))
c = plt.pcolormesh(x_1, x_2, np.transpose(z), vmin=0, vmax=1)
plt.colorbar(c)
plt.scatter(x_train[np.where(y_train == 1)][:, 0], x_train[np.where(y_train == 1)][:, 1])
plt.scatter(x_train[np.where(y_train == 0)][:, 0], x_train[np.where(y_train == 0)][:, 1])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()
