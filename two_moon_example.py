import sklearn
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from intervals_classification import CI_classificationx
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
x, y = make_moons(n_samples=100, random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)


plt.scatter(x_train[np.where(y_train==1)][:,0], x_train[np.where(y_train==1)][:,1])
plt.scatter(x_train[np.where(y_train==0)][:,0], x_train[np.where(y_train==0)][:,1])
plt.show()

def get_model():
    model = Sequential()
    model.add(Dense(30, activation='elu', input_shape=(2, ),
                    kernel_regularizer=tf.keras.regularizers.l2( l=0.00)))
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2( l=0.00)))
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2( l=0.00)))
    model.add(Dense(1,
                    kernel_regularizer=tf.keras.regularizers.l2( l=0.00)))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0),
                  metrics=['accuracy'])

    model.fit(x=x_train, y=y_train, epochs=500)
    return model

model = get_model()

x_1 = np.linspace(-1.5, 2.5, 20)
x_2 = np.linspace(-1, 1.5, 20)
x_val = [[x1, x2] for x1 in x_1 for x2 in x_2]
CIs = np.zeros((len(x_val), 2))
for i, x in enumerate(x_val):
    print(i+1)
    CIs[i, 0], CIs[i, 1] = CI_classificationx(model=model, x=x, X_train=x_train,
                          Y_train=y_train, p_hats=model.predict(x_train),
                          n_steps=100, alpha=0.1, n_epochs=250, fraction=1,
                                              verbose=0)
z= CIs[:, 1] - CIs[:, 0]
z = np.reshape(z, (20, 20))
c = plt.pcolormesh(x_1, x_2, np.transpose(z), vmin=0, vmax=1)
plt.colorbar(c)
plt.scatter(x_train[np.where(y_train==1)][:,0], x_train[np.where(y_train==1)][:,1])
plt.scatter(x_train[np.where(y_train==0)][:,0], x_train[np.where(y_train==0)][:,1])
plt.show()


ensemble = [get_model() for _ in range(10)]


x_1 = np.linspace(-3, 4, 40)
x_2 = np.linspace(-3, 3, 40)
x_val = [[x1, x2] for x1 in x_1 for x2 in x_2]
CI_ensemble = np.zeros((len(x_val), 2))
p_hats_ensemble = np.zeros((len(x_val)))
t = scipy.stats.t(df=9).ppf(1-0.1/2)
for i, x in enumerate(x_val):
    predictions = [sigmoid(model.predict(np.array([x]))) for model in ensemble]
    var = np.var(predictions)
    mean = np.mean(predictions)
    CI_ensemble[i, 0] = mean - t * np.sqrt(var / len(ensemble))
    CI_ensemble[i, 1] = mean + t * np.sqrt(var / len(ensemble))
    p_hats_ensemble[i] = mean

z = CI_ensemble[:, 1] - CI_ensemble[:, 0]
z = np.reshape(z, (40, 40))
c = plt.pcolormesh(x_1, x_2, np.transpose(z), vmin=np.min(z), vmax=np.max(z))
plt.colorbar(c)
plt.scatter(x_train[np.where(y_train==1)][:,0], x_train[np.where(y_train==1)][:,1])
plt.scatter(x_train[np.where(y_train==0)][:,0], x_train[np.where(y_train==0)][:,1])
plt.show()
