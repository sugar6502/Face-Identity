from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import keras
from keras import layers
import numpy as np


class model:
    def __init__(self):
        vgg16_model = VGG16()
        model = keras.Sequential()

        for layer in vgg16_model.layers[:-1]:
            model.add(layer)

        model.layers.pop()

        model.add(layers.Dense(2048, activation='softmax'))
        self.model = model




        

    def extract(self, image):
        
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        return self.model.predict(image)
