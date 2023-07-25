import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import gc
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from intervals_classification import CI_classificationx, CI_dropout, CI_ensemble
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
plt.rcParams['text.usetex'] = True
plt.rcParams["font.size"] = 25
plt.rcParams['axes.linewidth'] = 1


#%% Generating the data
x, y = make_moons(n_samples=100, random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
mean = np.mean(x_train, axis=0)
stdev = np.std(x_train, axis=0)
x_train_norm = (x_train - mean) / stdev


#%% Creating and training a model
def get_model(c_reg, n_epochs, dropout_rate=0, verbose=True):
    """Train a simple network with three hidden layers"""
    model = Sequential()
    model.add(Dense(30, activation='elu', input_shape=(2, ),
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dropout(dropout_rate))
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
predicted_logits = model.predict(x_train_norm, verbose=0)
for i, x in enumerate(x_val):
    print(i+1)
    CIs[i, 0], CIs[i, 1] = CI_classificationx(model=model, x=(x - mean) / stdev,
                                              X_train=x_train_norm,
                                              Y_train=y_train, predicted_logits=predicted_logits,
                                              n_steps=100, alpha=0.05,
                                              n_epochs=500, fraction=1/16,
                                              verbose=0, weight=1)
    tf.keras.backend.clear_session()
    gc.collect()

#%% Saving the results
with open('./Results/twomoon-1e-3-500epoch-v2.pickle', 'wb') as handle:
    pickle.dump({'confidence_intervals': CIs, 'locations': x_val}, handle)

#%%
with open('./Results/twomoon-1e-3-500epoch-w1.pickle', 'rb') as handle:
    data_dict = pickle.load(handle)
CIs = data_dict['confidence_intervals']
x_val = data_dict['locations']

#%% Visualizing the results
z = CIs[:, 1] - CIs[:, 0]
z = np.reshape(z, (30, 30))
plt.figure(dpi=200)
c = plt.pcolormesh(x_1, x_2, np.transpose(z), vmin=0, vmax=1, cmap='viridis')
plt.colorbar(c)
plt.scatter(x_train[np.where(y_train == 1)][:, 0], x_train[np.where(y_train == 1)][:, 1])
plt.scatter(x_train[np.where(y_train == 0)][:, 0], x_train[np.where(y_train == 0)][:, 1])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.tight_layout()
plt.show()

#%% Using an ensemble
ensemble = [get_model(1e-3, 500, verbose=True) for _ in range(10)]
CI_ens = np.zeros((len(x_val), 2))
p_hats_ensemble = np.zeros((len(x_val)))
for i, x in enumerate(x_val):
    print(i)
    p_hats_ensemble[i], CI_ens[i] = CI_ensemble(ensemble, x=(x-mean)/stdev,
                                             alpha=0.05, give_prediction=True)

#%% Saving CIs from ensemble approach
with open('./Results/twomoon-ensemble-w1.pickle', 'wb') as handle:
    pickle.dump({'confidence_intervals': CI_ensemble, 'locations': x_val}, handle)

#%% Visualizing the ensemble approach
plt.figure(dpi=200)
z = CI_ens[:, 1] - CI_ens[:, 0]
z = np.reshape(z, (30, 30))
c = plt.pcolormesh(x_1, x_2, np.transpose(z), vmin=0, vmax=1)
plt.colorbar(c)
plt.scatter(x_train[np.where(y_train == 1)][:, 0], x_train[np.where(y_train == 1)][:, 1])
plt.scatter(x_train[np.where(y_train == 0)][:, 0], x_train[np.where(y_train == 0)][:, 1])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.tight_layout()
plt.show()


#%% Dropout example
dropout_model = get_model(c_reg=1e-3, n_epochs=1000, dropout_rate=0.2)
CI_d = np.zeros((len(x_val), 2))
p_hats_dropout = np.zeros((len(x_val)))
for i, x in enumerate(x_val):
    print(i)
    p_hats_dropout[i], CI_d[i] = CI_dropout(dropout_model, (x-mean)/stdev,
                                            B=1000, alpha=0.05)

#%% Visualize the dropout approach
z = CI_d[:, 1] - CI_d[:, 0]
z = np.reshape(z, (30, 30))
plt.figure(dpi=200)
c = plt.pcolormesh(x_1, x_2, np.transpose(z), vmin=0, vmax=1)
plt.colorbar(c)
plt.scatter(x_train[np.where(y_train == 1)][:, 0], x_train[np.where(y_train == 1)][:, 1])
plt.scatter(x_train[np.where(y_train == 0)][:, 0], x_train[np.where(y_train == 0)][:, 1])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.tight_layout()
plt.show()
