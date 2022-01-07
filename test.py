import tensorflow as tf
import numpy as np
from InputDependentWeightLayer import InputDependentWeightLayer

epochs = 2


model = tf.keras.models.Sequential([
    InputDependentWeightLayer(1)  # one layer with one output node
])

model.compile(loss='mse', optimizer='adam')

# generate training and testing data. The task is to multiply two numbers.

x1_train = np.random.uniform(size=10000)
x2_train = np.random.uniform(size=10000)

x_train = np.column_stack((x1_train, x2_train))
y_train = x1_train * x2_train

x1_test = np.random.uniform(size=1000)
x2_test = np.random.uniform(size=1000)

x_test = np.column_stack((x1_test, x2_test))
y_test = x1_test * x2_test


# train model
model.fit(x_train, y_train, epochs=10, batch_size=1, validation_data=(x_test, y_test))
