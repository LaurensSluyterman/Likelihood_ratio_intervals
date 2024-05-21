import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, AveragePooling2D
from keras.datasets import cifar10
from adverserial_example_generation import generate_adversarial_example
from intervals_classification import CI_classificationx, CI_ensemble, CI_dropout

#%% Data import and preprocessing
data = cifar10.load_data()

(x_train, y_train), (x_test, y_test) = data
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')

x_train /= 255.
x_test /= 255.

x_train_0 = x_train[np.where(y_train[:, 0] == 0)]
x_train_1 = x_train[np.where(y_train[:, 0] == 1)]
x_train_4 = x_train[np.where(y_train[:, 0] == 4)]
x_train_7 = x_train[np.where(y_train[:, 0] == 7)]
x_train_8 = x_train[np.where(y_train[:, 0] == 8)]

y_train_0 = np.zeros(len(x_train_0))
y_train_1 = np.ones(len(x_train_1))

x_test_0 = x_test[np.where(y_test[:, 0] == 0)]
x_test_1 = x_test[np.where(y_test[:, 0] == 1)]
y_test_0 = np.zeros(len(x_test_0))
y_test_1 = np.ones(len(x_test_1))

x_train01 = np.vstack((x_train_0, x_train_1))
y_train01 = np.hstack((y_train_0, y_train_1))

x_test01 = np.vstack((x_test_0, x_test_1))
y_test01 = np.hstack((y_test_0, y_test_1))

#%% Visual check
plt.imshow(x_train_7[4])
plt.show()

#%% Training a model
input_shape = (32, 32, 3)
c_reg = 1e-5
batch_size = 32
n_epochs = 15
optimizer = 'sgd'
np.random.seed(2)


def train_model(c_reg, n_epochs, dropout_rate=0.0):
    """Train a simple CNN.

    This function trains a simple CNN consisting of multiple
    convolutional layers followed by max-pooling layers.
    The classification part consists of three hidden layers with
    30 units and elu activation functions. By default, no dropout
    is used.

    Arguments:
        c_reg (float): The (l2) regularization constant.
        n_epochs (int): The number of training epochs.
        dropout_rate (float): The dropout rate.

    Return:
          model: The trained model
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     input_shape=input_shape, activation='elu',
                     kernel_regularizer=tf.keras.regularizers.l2(l=0)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='elu',
                     kernel_regularizer=tf.keras.regularizers.l2(l=0)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1,
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(x=x_train01, y=y_train01, epochs=n_epochs, batch_size=batch_size, validation_split=0)
    return model


trained_model = train_model(c_reg, n_epochs)
#%%
trained_model.save('./models/CNNcifar-1e-5-sgd-n15')
#%% Calculating confidence intervals
trained_model = tf.keras.models.load_model('./models/CNNcifar-1e-5-sgd-n15')

print(CI_classificationx(model=trained_model, x=x_train01[-2], X_train=x_train01,
                         Y_train=y_train01, predicted_logits=trained_model.predict(x_train01),
                         n_steps=200, alpha=0.05, n_epochs=n_epochs, fraction=2/batch_size,
                         verbose=1, optimizer=optimizer, compile=True,
                         weight=1))

print(CI_classificationx(model=trained_model, x=x_test01[-2], X_train=x_train01,
                         Y_train=y_train01, predicted_logits=trained_model.predict(x_train01),
                         n_steps=200, alpha=0.05, n_epochs=n_epochs, fraction=2/batch_size,
                         verbose=1, optimizer=optimizer, compile=True,
                         batch_size=batch_size, weight=1))

print(CI_classificationx(model=trained_model, x=x_train_4[0], X_train=x_train01,
                         Y_train=y_train01, predicted_logits=trained_model.predict(x_train01),
                         n_steps=200, alpha=0.05, n_epochs=n_epochs, fraction=2/batch_size,
                         verbose=1, compile=True, optimizer=optimizer,
                         batch_size=batch_size, weight=1))


#%% Display an image
plt.imshow(x_train01[-1])
plt.show()

#%% Adversarial example
trained_model = tf.keras.models.load_model('./models/CNNcifar-1e-5-sgd-n15')

x_normal = x_test01[0]
label = np.array([y_test01[0]])
x_adversarial = generate_adversarial_example(trained_model, np.array([x_normal]),
                                             label, epsilon=0.03)

print(CI_classificationx(model=trained_model, x=x_normal, X_train=x_train01,
                         Y_train=y_train01, predicted_logits=trained_model.predict(x_train01),
                         optimizer=optimizer, n_steps=200, alpha=0.05,
                         n_epochs=n_epochs, fraction=2/batch_size,
                         verbose=1, weight=1, compile=True))

print(CI_classificationx(model=trained_model, x=x_adversarial[0], X_train=x_train01,
                         Y_train=y_train01, predicted_logits=trained_model.predict(x_train01),
                         n_steps=200, alpha=0.05, n_epochs=n_epochs,
                         optimizer=optimizer, fraction=2/batch_size,
                         verbose=1, weight=1, compile=True))


# Display original and adversarial input. This code snippet was made by chatGPT.
f, ax = plt.subplots(1, 2)
ax[0].imshow(x_normal.squeeze())
ax[0].set_title('Original Input')
ax[1].imshow(x_adversarial.squeeze())
ax[1].set_title('Adversarial Example')
plt.show()


#%% Dropout and ensemble
ensemble = [train_model(c_reg, n_epochs) for _ in range(10)]
model_dropout = train_model(c_reg, n_epochs, dropout_rate=0.2)

x = x_train_4[0]
CI_dropout(model_dropout, x=x, alpha=0.05)
CI_ensemble(ensemble, x, alpha=0.05)
