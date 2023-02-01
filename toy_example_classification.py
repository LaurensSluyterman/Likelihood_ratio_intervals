import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from intervals_classification import CI_classificationx
def p(x):
    return 0.5 + 0.4*np.cos(6*x)

n = 50
x_train = np.hstack((np.random.uniform(0, 0.2, n), np.random.uniform(0.8,1, n)))
x_train = np.reshape(x_train, (len(x_train), 1))
y_train = np.random.binomial((1), p=p(x_train))[:, 0]
x_lin = np.linspace(0, 1, 50)
plt.plot(x_train, y_train, 'o')
plt.plot(np.linspace(0, 1, 50), p(np.sort(np.linspace(0, 1, 50))))
plt.show()

def get_model():
    model = Sequential()
    model.add(Dense(30, activation='elu', input_shape=(1, ),
                    kernel_regularizer=tf.keras.regularizers.l2(l=0.00)))
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(l=0.00)))
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(l=0.00)))
    model.add(Dense(1,
                    kernel_regularizer=tf.keras.regularizers.l2(l=0.00)))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0),
                  metrics=['accuracy'])

    model.fit(x=x_train, y=y_train, epochs=300)
    return model

model = get_model()
plt.plot(x_train, y_train, 'o')
plt.plot(np.linspace(0, 1, 50), p(np.sort(np.linspace(0, 1, 50))), label='p(x)')
plt.plot(x_lin, tf.math.sigmoid(model.predict(x_lin)), label='p_hat(x)')
plt.legend()
plt.show()

CI_classificationx(model=model, x=np.array([0.1]), X_train=x_train,
                          Y_train=y_train, p_hats=model.predict(x_train),
                          n_steps=100, alpha=0.05, n_epochs=2, fraction=0.2)


x_lin = np.linspace(-0.8, 1.8, 30)
CIs = np.zeros((len(x_lin), 2))
for i, x in enumerate(x_lin):
    print(i+1)
    CIs[i, 0], CIs[i, 1] = CI_classificationx(model=model, x=x_lin[i], X_train=x_train,
                          Y_train=y_train, p_hats=model.predict(x_train),
                          n_steps=100, alpha=0.2, n_epochs=150, fraction=0.2,
                                              verbose=0)

p_hat = tf.math.sigmoid(model.predict(x_lin))
plt.plot(x_train, y_train, 'o')
plt.plot(x_lin, p(x_lin), label='p(x)')
plt.plot(x_lin, p_hat, label='p_hat(x)')
plt.fill_between(x_lin, CIs[:, 0], CIs[:, 1], color='blue', alpha=0.2, linewidth=0.1, label='CI')
plt.xlabel('x')
plt.legend()
plt.show()

ensemble = [get_model() for _ in range(10)]

CI_ensemble = np.zeros((len(x_lin), 2))
p_hats_ensemble = np.zeros((len(x_lin)))
for i, x in enumerate(x_lin):
    print(i+1)
    t = scipy.stats.t(df=9).ppf(1-0.05/2)
    predictions = [sigmoid(model.predict(np.array([x]))) for model in ensemble]
    var = np.var(predictions)
    mean = np.mean(predictions)
    CI_ensemble[i, 0] = mean - t * np.sqrt(var / len(ensemble))
    CI_ensemble[i, 1] = mean + t * np.sqrt(var / len(ensemble))
    p_hats_ensemble[i] = mean

plt.plot(x_train, y_train, 'o')
plt.plot(x_lin, p(x_lin), label='p(x)')
plt.plot(x_lin, p_hats_ensemble, label='p_hat(x)')
plt.fill_between(x_lin, CI_ensemble[:, 0], CI_ensemble[:, 1], color='blue', alpha=0.2, linewidth=0.1, label='CI')
plt.xlabel('x')
plt.legend()
plt.show()