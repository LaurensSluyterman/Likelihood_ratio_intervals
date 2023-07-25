import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from intervals_classification import CI_classificationx, CI_ensemble, CI_dropout
from utils import sigmoid
plt.rcParams['text.usetex'] = True
plt.rcParams["font.size"] = 15
plt.rcParams['axes.linewidth'] = 0.7

#%% Import and preprocess data
data = mnist.load_data()
(x_train, y_train), (x_test, y_test) = data


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Convert to float
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')

# Normalize
x_train /= 255.
x_test /= 255.


# Create a training and test set containing only classes 0 and 1
x_train_0 = x_train[np.where(y_train == 0)]
x_train_1 = x_train[np.where(y_train == 1)]
y_train_0 = np.zeros(len(x_train_0))
y_train_1 = np.ones(len(x_train_1))

x_train_01 = np.vstack((x_train_0, x_train_1))
y_train_01 = np.hstack((y_train_0, y_train_1))

x_test_0 = x_test[np.where(y_test == 0)]
x_test_1 = x_test[np.where(y_test == 1)]
y_test_0 = np.zeros(len(x_train_0))
y_test_1 = np.ones(len(x_test_1))

x_test_01 = np.vstack((x_test_0, x_test_1))
y_test_01 = np.hstack((y_test_0, y_test_1))

# OoD inputs
x_train_7 = x_train[np.where(y_train == 7)]
x_train_4 = x_train[np.where(y_train == 4)]
x_train_2 = x_train[np.where(y_train == 2)]

#%% Visual check
plt.imshow(x_train_7[0])
plt.show()

#%% Create and train a basic CNN
input_shape = (28, 28, 1)
np.random.seed(2)


def train_model(c_reg, n_epochs, dropout_rate=0.0):
    """Train a simple CNN.

    This function trains a simple CNN consisting of multiple
    convolutional layers followed by max-pooling layers.
    The classification part consists of two hidden layers with
    30 units and elu activation functions. By default, no dropout
    is used.

    Arguments:
        c_reg (float): The (l2) regularization constant.
        n_epochs (int): The number of training epochs.
        dropout_rate(float): The dropout rate.

    Return:
          model: The trained model
    """
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(28, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(30, activation='elu',
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1,
                    kernel_regularizer=tf.keras.regularizers.l2(l=c_reg)))
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy', 'binary_crossentropy'])
    model.fit(x=x_train_01, y=y_train_01, epochs=n_epochs)
    return model


model = train_model(1e-4, 10)
#%% Save the model
model.save('./models/CNNMnistmedium-1e-4-sgd')
#%% Creating CIs for different inputs
model = tf.keras.models.load_model('./models/CNNMnistmedium-1e-4-sgd')

# Training point
print(CI_classificationx(model=model, x=x_train_01[1], X_train=x_train_01,
                         Y_train=y_train_01, predicted_logits=model.predict(x_train_01),
                         n_steps=200, alpha=0.05, n_epochs=10, fraction=1/16, compile=True,
                         optimizer='sgd'))

# Test point
print(CI_classificationx(model=model, x=x_test_01[1], X_train=x_train_01,
                         Y_train=y_train_01, predicted_logits=model.predict(x_train_01),
                         n_steps=200, alpha=0.05, n_epochs=10, fraction=1/16, compile=True,
                         optimizer='sgd'))

# OoD point
print(CI_classificationx(model=model, x=x_train_2[0], X_train=x_train_01,
                         Y_train=y_train_01, predicted_logits=model.predict(x_train_01),
                         n_steps=200, alpha=0.05, n_epochs=10, fraction=1/16,
                         verbose=1, compile=True, weight=1,
                         optimizer='sgd'))

#%% Rotation example
datagen = ImageDataGenerator()
rotations = [30*i for i in range(4)]
rotation_results = {}
predictions = []
img = x_test_01[-1]
for rotation in rotations:
    print(rotation)
    rotated_image = datagen.apply_transform(x=img, transform_parameters={'theta': rotation})
    CI = CI_classificationx(model=model, x=rotated_image, X_train=x_train_01,
                            Y_train=y_train_01, predicted_logits=model.predict(x_train_01),
                            n_steps=150, alpha=0.05, n_epochs=10, fraction=1/16,
                            weight=1,
                            compile=True)
    rotation_results[rotation] = CI
    predictions.append(model.predict(np.array([rotated_image])))

lower_bounds = [a[0] for a in rotation_results.values()]
upper_bounds = [a[1] for a in rotation_results.values()]
p_hats = [sigmoid(p[0][0]) for p in predictions]
plt.figure(dpi=200, figsize=(6, 3))
plt.title('DeepLR')
plt.fill_between(rotations, lower_bounds, upper_bounds, interpolate=False, alpha=0.3)
plt.plot(rotations, p_hats, color='red', linestyle='--', marker='o')
plt.xlabel('rotation', fontsize=14)
plt.ylabel(r'$p_{\hat{\theta}}$', fontsize=14)
plt.xticks([0, 30, 60, 90], fontsize=14)
plt.yticks(fontsize=14)
plt.ylim((0, 1.05))
plt.tight_layout()
plt.show()

#%% Plot rotated image
rotated_image = datagen.apply_transform(x=img, transform_parameters={'theta': 0})
plt.imshow(rotated_image)
plt.axis('off')
plt.show()


#%% Train ensemble and dropout
ensemble = [train_model(1e-4, 10) for _ in range(10)]
model_dropout = train_model(1e-4, 10, dropout_rate=0.2)


datagen = ImageDataGenerator()
rotations = [30*i for i in range(4)]
img = x_test_01[-1]
rotation_results_dropout = {}
rotation_results_ensemble = {}
predictions_dropout = []
predictions_ensemble = []
for rotation in rotations:
    print(rotation)
    rotated_image = datagen.apply_transform(x=img, transform_parameters={'theta': rotation})
    prediction_ensemble, CI_e = CI_ensemble(ensemble, rotated_image, alpha=0.05)
    prediction_dropout, CI_d = CI_dropout(model_dropout, x=rotated_image, alpha=0.05)
    rotation_results_ensemble[rotation] = CI_e
    predictions_ensemble.append(prediction_ensemble)
    rotation_results_dropout[rotation] = CI_d
    predictions_dropout.append(prediction_dropout)

#%% Plot dropout rotation
lower_bounds = [a[0] for a in rotation_results_dropout.values()]
upper_bounds = [a[1] for a in rotation_results_dropout.values()]
p_hats = predictions_dropout
plt.figure(dpi=200, figsize=(6, 3))
plt.title('MC-dropout')
plt.fill_between(rotations, lower_bounds, upper_bounds, interpolate=False, alpha=0.3)
plt.plot(rotations, p_hats, color='red', linestyle='--', marker='o')
plt.xlabel('rotation', fontsize=14)
plt.ylabel(r'$p_{\hat{\theta}}$', fontsize=14)
plt.xticks([0, 30, 60, 90], fontsize=14)
plt.yticks(fontsize=14)
plt.ylim((0, 1.05))
plt.tight_layout()
plt.show()

#%% Plot ensemble rotation
lower_bounds = [a[0] for a in rotation_results_ensemble.values()]
upper_bounds = [a[1] for a in rotation_results_ensemble.values()]
p_hats = predictions_ensemble
plt.figure(dpi=200, figsize=(6, 3))
plt.title('Ensemble')
plt.fill_between(rotations, lower_bounds, upper_bounds, interpolate=False, alpha=0.3)
plt.plot(rotations, p_hats, color='red', linestyle='--', marker='o')
plt.xlabel('rotation', fontsize=14)
plt.ylabel(r'$p_{\hat{\theta}}$', fontsize=14)
plt.xticks([0, 30, 60, 90], fontsize=14)
plt.yticks(fontsize=14)
plt.ylim((0, 1.05))
plt.tight_layout()
plt.show()


#%% Dropout and ensemble
x = x_train_4[0]
CI_dropout(model_dropout, x=x, alpha=0.05)
CI_ensemble(ensemble, x, alpha=0.05)
