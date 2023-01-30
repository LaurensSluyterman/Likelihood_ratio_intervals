from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample


data = mnist.load_data()
(x_train, y_train), (x_test, y_test) = data


np.shape(x_train[0])
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
x_val /= 255

x_train_0 = x_train[np.where(y_train==0)]
x_train_1 = x_train[np.where(y_train==1)]
y_train_0 = np.zeros(len(x_train_0))
y_train_1 = np.ones(len(x_train_1))

x_train_7= x_train[np.where(y_train==7)]
x_test_0 = x_train[np.where(y_test==0)]
x_test_1 = x_train[np.where(y_test==1)]
y_test_0 = np.zeros(len(x_train_0))
y_test_1 = np.ones(len(x_test_1))

x_train01 = np.vstack((x_train_0, x_train_1))
y_train01 = np.hstack((y_train_0, y_train_1))

x_test01 = np.vstack((x_test_0, x_test_1))
y_test01 = np.hstack((y_test_0, y_test_1))

input_shape = (28, 28, 1)
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(20, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
model.add(Dense(1))
# model.add(Activation(activation=tf.nn.sigmoid))
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x=x_train01, y=y_train01, epochs=5)

model.save('./models/basicCNN')
sigmoid(model.predict(np.array([x_train_7[20]])))

model = tf.keras.models.load_model('./models/basicCNN')
CI_classificationx(model=model, x=x_train_7[20], X_train=x_train01,
                          Y_train=y_train01, p_hats=model.predict(x_train01),
                          n_steps=100, alpha=0.05, n_epochs=1, fraction=0.1)

CI_classificationx(model=model, x=x_test01[2], X_train=x_train01,
                          Y_train=y_train01, p_hats=model.predict(x_train01),
                          n_steps=100, alpha=0.05, n_epochs=1, fraction=0.1)

CI_classificationx(model=model, x=5*x_train01[20], X_train=x_train01,
                          Y_train=y_train01, p_hats=model.predict(x_train01),
                          n_steps=100, alpha=0.05, n_epochs=1, fraction=0.1)

print(sigmoid(b))
print(sigmoid(l))


a = [4, -2]
b = [-11, 1]


logit_path_1 = [sigmoid(t/100 *b[0] + (1-t/100)*a[0]) for t in range(0, 101)]
logit_path_2 = [sigmoid(t/100 *b[1] + (1-t/100)*a[1]) for t in range(0, 101)]
sigmoid_path_1 = [t/100 *sigmoid(b[0]) + (1-t/100)*sigmoid(a[0]) for t in range(0, 101)]
sigmoid_path_2 = [t/100 *sigmoid(b[1]) + (1-t/100)*sigmoid(a[1]) for t in range(0, 101)]

plt.scatter(logit_path_1, logit_path_2, label='logit-interpolation')
plt.scatter(sigmoid_path_1, sigmoid_path_2, label='sigmoid-interpolation')
plt.legend()
plt.xlabel('p0')
plt.ylabel('p1')
plt.show()
