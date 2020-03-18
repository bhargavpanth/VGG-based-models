import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from model import CNN_Model

'''
Baseline model is the worst performing model.
It is present to consider only a baseline performance on your dataset
'''

class Baseline_Model(CNN_Model):

    def model(self):
        # custom models are built to be sequential by default
        return Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224,224,3)),
            MaxPool2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(units=2, activation='softmax')
        ],name='baseline_model')

    def build(self):
        return self.model()
