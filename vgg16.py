import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from model import CNN_Model

class VGG_16(CNN_Model):

    # Set trainable to True if you want the model to drop all
    # its pre trained weights and biases
    def __init__(self, trainable: bool = False):
        self.trainable = trainable

    def model(self):
        return tf.keras.applications.vgg16.VGG16()

    def __sequential(self, output_classes: int):
        non_sequential_vgg_16_model = self.model()
        sequential_vgg_16_model = Sequential()

        for layer in non_sequential_vgg_16_model.layers[:-1]:
            layer.trainable = self.trainable
            sequential_vgg_16_model.add(layer)

        # adding output layer with the required number of output classes
        sequential_vgg_16_model.add(Dense(units=output_classes, activation='softmax'))
        return sequential_vgg_16_model

    def build(self, output_classes: int):
        return self.__sequential(output_classes)

# To compile the model

# vgg_16_model = VGG_16().build(2)
# vgg_16_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) 

# To run it on training set
# vgg_16_model.fit()
