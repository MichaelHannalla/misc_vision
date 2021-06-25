# Haidy Sorial Samy         16P8104
# Michael Samy Hannalla     16P8202
# Training of a tensorflow based neural network on Caltech 101 dataset
# CSE489 - Machine Vision 
# Faculty of Engineering - Ain Shams University 

import numpy as np
import tensorflow as tf  
from tensorflow import keras
import matplotlib
from matplotlib import pyplot as plt 
import os
import random
import cv2
import resource 
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Limit the resources to avoid device crashing
def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))

memory_limit()

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    # Restrict TensorFlow to only allocate the defined amount of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

class Container():
    def __init__(self, image, label):
        self.image = image
        self.label = int(label)
        
#Importing images and creating dataset
dataset = []
label_names = []

caltech101_path = "101_ObjectCategories"
classes = os.listdir(caltech101_path)
for idx, class_ in enumerate(classes): 
    images = os.listdir(caltech101_path+"/"+class_)
    for image in images:
        img = cv2.imread(caltech101_path+"/"+class_+"/"+image)
        if np.any(img > 0):
            dataset.append(Container(img, idx))
    label_names.append(class_)

# shuffling the dataset
np.random.shuffle(dataset)
train_dataset = dataset[0:int(0.8*len(dataset))]
test_dataset   = dataset[int(0.8*len(dataset)):]

train_images = []
train_labels = []
test_images = []
test_labels = []

#Split labels and images for training dataset, testing dataset, and validation dataset
for i in range(len(train_dataset)):
    temp_image = cv2.resize(train_dataset[i].image, (224,224))
    train_images.append(temp_image)
    train_labels.append(train_dataset[i].label)

train_images = np.asarray(train_images).astype(np.float16)

#Create the model 
cnn_model = keras.models.Sequential(
    [   

        # Implementation of VGG-16 like neural network architecture

        keras.layers.Conv2D(input_shape=(224, 224, 3), kernel_size=(3, 3), filters=64/4, padding='same', activation='relu', name='conv1-1'),
        keras.layers.Conv2D(kernel_size=(3, 3), filters=64/4, padding='same', activation='relu', name='conv1-2'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

        keras.layers.Conv2D(kernel_size=(3,3), filters=128/4, padding='same', activation='relu', name='conv2-1'),
        keras.layers.Conv2D(kernel_size=(3,3), filters=128/4, padding='same', activation='relu', name='conv2-2'),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

        keras.layers.Conv2D(kernel_size=(3,3), filters=256/4, padding='same', activation='relu', name='conv3-1'),
        keras.layers.Conv2D(kernel_size=(3,3), filters=256/4, padding='same', activation='relu', name='conv3-2'),
        keras.layers.Conv2D(kernel_size=(3,3), filters=256/4, padding='same', activation='relu', name='conv3-3'),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

        keras.layers.Conv2D(kernel_size=(3,3), filters=512/4, padding='same', activation='relu', name='conv4-1'),
        keras.layers.Conv2D(kernel_size=(3,3), filters=512/4, padding='same', activation='relu', name='conv4-2'),
        keras.layers.Conv2D(kernel_size=(3,3), filters=512/4, padding='same', activation='relu', name='conv4-3'),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
        
        keras.layers.Conv2D(kernel_size=(3,3), filters=512/4, padding='same', activation='relu', name='conv5-1'),
        keras.layers.Conv2D(kernel_size=(3,3), filters=512/4, padding='same', activation='relu', name='conv5-2'),
        keras.layers.Conv2D(kernel_size=(3,3), filters=512/4, padding='same', activation='relu', name='conv5-3'),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
        keras.layers.Dropout(0.01),

        keras.layers.Flatten(name='flatten'),
        keras.layers.Dense(4096/2, activation='relu', name='fc-1'),
        keras.layers.Dense(4096/2, activation='relu', name='fc-2'),
        keras.layers.Dense(101,  activation='softmax', name='output')
    ]
)

cnn_model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint("caltech101_keras_model.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_period=1)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print(cnn_model.summary())

#Defining the generator
def generator(features, labels, batch_size):
    batch_features = np.zeros((batch_size, 224, 224, 3))
    batch_labels = np.zeros((batch_size,101))
    while True:
        for point_idx in range(0, len(features)-batch_size, batch_size):
            for index in range(batch_size):
                batch_features[index] = (features[index + point_idx])
                batch_labels[index] = keras.utils.to_categorical(labels[index + point_idx], num_classes=101)
            yield batch_features, batch_labels

try:
    #Model training using fit generator
    cnn_model.fit(generator(train_images, train_labels, batch_size=4),
                  steps_per_epoch = 2000, 
                  epochs = 20,
                  verbose = 1,
                  callbacks=[tensorboard_callback, checkpoint])

# This is to halt training and immediately start testing and evaluation steps
except KeyboardInterrupt:
    exit()
    
for i in range(len(test_dataset)):
    temp_image = cv2.resize(test_dataset[i].image, (224,224))
    test_images.append(temp_image)
    test_labels.append(keras.utils.to_categorical(test_dataset[i].label, num_classes=101))

test_images  = np.asarray(test_images).astype(np.float16)
test_labels = np.asarray(test_labels)

print("Starting model evaluation on test data")
test_loss, test_accuracy = cnn_model.evaluate(test_images, test_labels)
print('Test Accuracy: {}, Test Loss: {}'.format(test_accuracy, test_loss))
exit()
