import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.regularizers import l2 as l2
from utils import sigmoid
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, MaxPooling2D, Flatten, Conv2D, Activation, BatchNormalization, Dropout
from keras.models import Model
from intervals_classification import CI_classificationx, CI_ensemble

#%% Data import and preprocessing
data = np.load('./Brain_tumor_data/processed_data/data_medium.npz')
data_OoD = np.load('./Brain_tumor_data/processed_data/OoD_medium.npz')
X_OoD = data_OoD['X']
Y_OoD = data_OoD['Y']

X_train, X_test, Y_train, Y_test = train_test_split(data['X'], data['Y'], test_size=0.1, random_state=33)
#%% Visual check
i = 2
plt.title(Y_train[i])
plt.imshow(X_train[i])
plt.show()

#%% Training a model
input_shape = (128, 128, 3)
c_reg = 1e-5
batch_size = 32
n_epochs = 30
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
optimizer = 'sgd'
np.random.seed(2)

def train_model(c_reg, n_epochs, dropout_rate=0.0):
    inputs = Input(shape=input_shape)

    # First Convolutional Block
    x = Conv2D(filters=8, kernel_size=(3, 3), strides=2, kernel_regularizer=l2(c_reg), name='backbone1')(inputs)
    x = BatchNormalization(name='backbone2')(x)
    x = Activation('elu')(x)
    x = Dropout(rate=dropout_rate / 4)(x, training=True)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)


    # Second Convolutional Block
    x = Conv2D(filters=8, kernel_size=(3, 3), strides=1, kernel_regularizer=l2(c_reg), name='backbone3')(x)
    x = BatchNormalization(name='backbone4')(x)
    x = Activation('elu')(x)
    x = Dropout(rate=dropout_rate / 4)(x, training=True)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)


    # Third Convolutional Block
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=1, kernel_regularizer=l2(c_reg), name='backbone5')(x)
    x = BatchNormalization(name='backbone6')(x)
    x = Activation('elu')(x)
    x = Dropout(rate=dropout_rate / 4)(x, training=True)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)


    # Fourth Convolutional Block
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=2, kernel_regularizer=l2(c_reg), name='backbone7')(x)
    x = BatchNormalization(name='backbone8')(x)
    x = Activation('elu')(x)
    x = Dropout(rate=dropout_rate / 4)(x, training=True)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)


    # First Dense layer
    x = Flatten()(x)
    x = Dense(20, kernel_regularizer=l2(c_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(rate=dropout_rate)(x, training=True)

    # Second Dense layer
    x = Dense(10, kernel_regularizer=l2(c_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(rate=dropout_rate)(x, training=True)

    # Third Dense layer
    x = Dense(5, kernel_regularizer=l2(c_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(rate=dropout_rate )(x, training=True)

    # Output layer
    outputs = Dense(1, kernel_regularizer=l2(c_reg))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(x=X_train, y=Y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0)

    # Freeze the backbone after training
    for layer in model.layers:
        if layer.name.startswith('backbone'):
            layer.trainable = False
    return model

trained_model = train_model(c_reg, n_epochs)
trained_model.evaluate(X_test, Y_test)
#%% Save the trained model
trained_model.save('./models/CNNbrain2-1e-5-sgd-n30')

#%% Calculating confidence intervals
# trained_model = tf.keras.models.load_model('./models/CNN_brain-1e-5-sgd-n30') # Optional: Load a trained model

print(CI_classificationx(model=trained_model, x=X_train[0], X_train=X_train,
                         Y_train=Y_train, predicted_logits=trained_model.predict(X_train),
                         n_steps=100, alpha=0.05, n_epochs=n_epochs, fraction=2/batch_size,
                         verbose=1, optimizer=optimizer,
                         weight=1))

print(CI_classificationx(model=trained_model, x=X_test[0], X_train=X_train,
                         Y_train=Y_train, predicted_logits=trained_model.predict(X_train),
                         n_steps=100, alpha=0.05, n_epochs=n_epochs, fraction=2/batch_size,
                         verbose=1, optimizer=optimizer,
                         batch_size=batch_size, weight=1))

print(CI_classificationx(model=trained_model, x=X_OoD[0], X_train=X_train,
                         Y_train=Y_train, predicted_logits=trained_model.predict(X_train),
                         n_steps=100, alpha=0.05, n_epochs=n_epochs, fraction=2/batch_size,
                         verbose=1, optimizer=optimizer,
                         batch_size=batch_size, weight=1))


#%% Display an image
plt.imshow(X_train[0])
plt.show()

plt.imshow(X_OoD[0])
plt.show()

#%% Dropout and ensemble
ensemble = [train_model(c_reg, n_epochs) for _ in range(10)]
model_dropout = train_model(c_reg, n_epochs, dropout_rate=0.2)

x = X_OoD[1]

# Ensemble
CI_ensemble(ensemble, x, alpha=0.05)

# Dropout
predictions = [sigmoid(model_dropout.predict(np.array([x]), verbose=0)) for _ in range(300)]
prediction = np.mean(predictions)
lowerbound = np.percentile(predictions, 100*(0.05/2))
upperbound = np.percentile(predictions, 100*(1-0.05/2))
print(f'prediction{prediction} CI = {[lowerbound, upperbound]}')
