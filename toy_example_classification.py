import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import scipy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from intervals_classification import CI_classificationx
from scipy import stats
from utils import sigmoid
# matplotlib.use("TkAgg")
plt.rcParams['text.usetex'] = True
plt.rcParams["font.size"] = 17
plt.rcParams['axes.linewidth'] = 0.2


#%% Generate a data set
def p(x):
    return 0.5 + 0.4*np.cos(6*x)


np.random.seed(2)
n = 30
x_train = np.hstack((np.random.uniform(0, 0.2, n), np.random.uniform(0.8, 1, n)))
x_train = np.reshape(x_train, (len(x_train), 1))
y_train = np.random.binomial((1), p=p(x_train))[:, 0]
x_lin = np.linspace(0, 1, 50)
plt.plot(x_train, y_train, 'o', label='y-values')
plt.plot(np.linspace(0, 1, 50), p(np.sort(np.linspace(0, 1, 50))))
plt.xlabel(r'$x$')
plt.ylabel(r'$p(x)$')
plt.tight_layout()
plt.show()


#%% Train a model and visualize the predictions
c_reg = 1e-4
n_epochs = 300


def get_model():
    """Train simple network with three hidden layers"""
    model = Sequential()
    model.add(Dense(30, activation='elu', input_shape=(1, ),
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dense(1,
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0),
                  metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=n_epochs)
    return model


model = get_model()
plt.plot(x_train, y_train, 'o')
plt.plot(np.linspace(0, 1, 50), p(np.sort(np.linspace(0, 1, 50))), label=r'$p(x)$')
plt.plot(x_lin, tf.math.sigmoid(model.predict(x_lin)), label=r'$\hat{p}(x)$')
plt.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=2)
plt.xlabel(r'$x$')
plt.ylabel(r'$p(x)$')
plt.tight_layout()
plt.show()

#%% A single prediction
CI_classificationx(model=model, x=np.array([0]), X_train=x_train,
                   Y_train=y_train, predicted_logits=model.predict(x_train),
                   n_steps=100, alpha=0.1, n_epochs=500, fraction=1/16, weight=1)

#%% Creating CIs for a test set
x_lin = np.linspace(-0.8, 1.8, 50)
CIs = np.zeros((len(x_lin), 2))
p_hats = model.predict(x_train)
for i, x in enumerate(x_lin):
    print(i+1)
    CIs[i, 0], CIs[i, 1] = CI_classificationx(model=model,
                                              x=x_lin[i],
                                              X_train=x_train,
                                              Y_train=y_train,
                                              predicted_logits=p_hats,
                                              n_steps=100,
                                              alpha=0.1,
                                              n_epochs=n_epochs,
                                              fraction=1/16,
                                              verbose=0,
                                              weight=1)

#%% Visualizing the results
p_hat = tf.math.sigmoid(model.predict(x_lin))
plt.plot(x_train, y_train, 'o')
plt.plot(x_lin, p(x_lin), label=r'$p(x)$')
plt.plot(x_lin, p_hat, label=r'$\hat{p}(x)$')
plt.fill_between(x_lin, CIs[:, 0], CIs[:, 1], color='blue', alpha=0.2, linewidth=0.1, label=r'CI')
plt.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=3)
plt.xlabel(r'$x$')
plt.ylabel(r'$p(x)$')
plt.tight_layout()
plt.show()
plt.close()

#%% Creating CIs using an ensemble
ensemble = [get_model() for _ in range(10)]
CI_ensemble = np.zeros((len(x_lin), 2))
p_hats_ensemble = np.zeros((len(x_lin)))
for i, x in enumerate(x_lin):
    print(i+1)
    t = scipy.stats.t(df=9).ppf(1-0.1/2)
    predictions = [sigmoid(model.predict(np.array([x]))) for model in ensemble]
    var = np.var(predictions)
    mean = np.mean(predictions)
    CI_ensemble[i, 0] = mean - t * np.sqrt(var)
    CI_ensemble[i, 1] = mean + t * np.sqrt(var)
    p_hats_ensemble[i] = mean

#%% Visualizing the ensemble results
plt.plot(x_train, y_train, 'o')
plt.plot(x_lin, p(x_lin), label=r'$p(x)$')
plt.plot(x_lin, p_hats_ensemble, label=r'$\hat{p}(x)$')
plt.fill_between(x_lin, CI_ensemble[:, 0], CI_ensemble[:, 1], color='blue', alpha=0.2, linewidth=0.1, label=r'CI')
plt.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=3)
plt.xlabel(r'$x$')
plt.ylabel(r'$p(x)$')
plt.tight_layout()
plt.show()