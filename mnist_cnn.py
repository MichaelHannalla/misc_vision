import numpy as np
import matplotlib.pyplot as plt

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Validating the TensorFlow version
print(tf.__version__)

# Loading MNIST dataset
mnist_fashion = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist_fashion.load_data()

# Feature scaling
training_images = training_images / 255.0
test_images = test_images / 255.0

# Reshaping to stacked image-like matrices (num, width, height, channels)
training_images = training_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Definition of keras model
cnn_model = keras.models.Sequential()
cnn_model.add(keras.layers.Conv2D(50, (3, 3), activation='relu', input_shape=(28, 28, 1) ,name='Convolutional_layer'))
cnn_model.add(keras.layers.MaxPooling2D((2, 2), name='Maxpooling_2D'))
cnn_model.add(keras.layers.Flatten(name='Flatten'))
cnn_model.add(keras.layers.Dense(50, activation='relu', name='Hidden_layer'))
cnn_model.add(keras.layers.Dense(10, activation='softmax', name='Output_layer'))

# Model compilation
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Defining the training callback function
class heartbeatCb(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        print("One epoch succesfully ended")

# Training
out_model = cnn_model.fit(training_images, training_labels, epochs=10, callbacks=[heartbeatCb()])

# Training Evaluation
training_loss, training_accuracy = cnn_model.evaluate(training_images, training_labels)
print('Training Accuracy {}'.format(round(float(training_accuracy), 2)))

# Test Evaluation
test_loss, test_accuracy = cnn_model.evaluate(test_images, test_labels)
print('Test Accuracy {}'.format(round(float(test_accuracy), 2)))

# Data logs and plotting
loss = out_model.history['loss']
accuracy = out_model.history['accuracy']
epochs = range(10)
plt.figure(figsize=(len(epochs), len(epochs)))
plt.plot(accuracy, 'b', label='Training accuracy')
plt.plot(loss, 'r', label='Training loss')
plt.title('Training loss and Training accuracy')
plt.xlabel('Epoch')
plt.ylabel('Training Value')
plt.ylim([0, 1])
plt.legend()
plt.show()