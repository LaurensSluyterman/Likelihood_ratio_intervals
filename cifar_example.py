import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.datasets import cifar10
from intervals_classification import CI_classificationx

#%% Relaod
from importlib import reload
import intervals_classification
reload(intervals_classification)
from intervals_classification import CI_classificationx

#%% Data import and preprocessing
data = cifar10.load_data()

(x_train, y_train), (x_test, y_test) = data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255.
x_test /= 255.

x_train_0 = x_train[np.where(y_train[:, 0]==0)]
x_train_1 = x_train[np.where(y_train[:, 0]==1)]
y_train_0 = np.zeros(len(x_train_0))
y_train_1 = np.ones(len(x_train_1))

x_train_7 = x_train[np.where(y_train[:, 0]==7)]
x_train_8 = x_train[np.where(y_train[:, 0]==8)]
x_test_0 = x_test[np.where(y_test[:, 0]==0)]
x_test_1 = x_test[np.where(y_test[:, 0]==1)]
y_test_0 = np.zeros(len(x_test_0))
y_test_1 = np.ones(len(x_test_1))

x_train01 = np.vstack((x_train_0, x_train_1))
y_train01 = np.hstack((y_train_0, y_train_1))

x_test01 = np.vstack((x_test_0, x_test_1))
y_test01 = np.hstack((y_test_0, y_test_1))
#%%
plt.imshow(x_train_7[4])
plt.show()

#%% Training a model
input_shape = (32, 32, 3)
c_reg = 0.000000
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape,
                 kernel_regularizer=tf.keras.regularizers.l2( l=c_reg)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu',
                 kernel_regularizer =tf.keras.regularizers.l2( l=c_reg)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu',
                 kernel_regularizer =tf.keras.regularizers.l2( l=c_reg)))
model.add(Flatten())
model.add(Dense(30, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2( l=c_reg)))
model.add(Dense(30, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2( l=c_reg)))
model.add(Dense(30, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2( l=c_reg)))
model.add(Dense(1,
                kernel_regularizer=tf.keras.regularizers.l2( l=c_reg)))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0),
              metrics=['accuracy'])

model.fit(x=x_train01, y=y_train01, epochs=10)

#%% Calculating confidence intervals
model = tf.keras.models.load_model('./models/CNNcifar')
print(CI_classificationx(model=model, x=x_train_7[4], X_train=x_train01,
                          Y_train=y_train01, p_hats=model.predict(x_train01),
                          n_steps=100, alpha=0.05, n_epochs=1, fraction=0.1,
                         verbose=1, from_sigmoid=True))

print(CI_classificationx(model=model, x=x_test01[12], X_train=x_train01,
                          Y_train=y_train01, p_hats=model.predict(x_train01),
                          n_steps=100, alpha=0.05, n_epochs=1, fraction=0.1,
                         verbose=0, from_sigmoid=True))

print(CI_classificationx(model=model, x=x_train01[12], X_train=x_train01,
                          Y_train=y_train01, p_hats=model.predict(x_train01),
                          n_steps=100, alpha=0.05, n_epochs=1, fraction=0.1,
                         verbose=0, from_sigmoid=True))

#%% Adversarial example
model = tf.keras.models.load_model('./models/CNNcifar')

x_normal = x_train01[-6]
x_adversarial = generate_adversarial_example(model, x_train01[5:6], y=y_train01[5:6],
                                   epsilon=0.01)
print(CI_classificationx(model=model, x=x_normal, X_train=x_train01,
                          Y_train=y_train01, p_hats=model.predict(x_train01),
                          n_steps=100, alpha=0.05, n_epochs=1, fraction=0.1,
                         verbose=1, from_sigmoid=True))

print(CI_classificationx(model=model, x=x_adversarial[0], X_train=x_train01,
                          Y_train=y_train01, p_hats=model.predict(x_train01),
                          n_steps=100, alpha=0.05, n_epochs=1, fraction=0.1,
                         verbose=1, from_sigmoid=True))

x_normal = x_test01[-5]
x_adversarial = generate_adversarial_example(model, np.array([x_normal]), y=np.array([1]),
                                   epsilon=0.01)

plt.imshow(x_normal)
plt.show()

plt.imshow(x_adversarial[0])
plt.show()

model.predict(np.array([x_normal]))
model.predict(x_adversarial)