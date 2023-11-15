from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import keras
from keras import layers
import numpy as np


class model:
    def __init__(self):
        vgg16_model = VGG16(weights="imagenet",include_top=False)
        model = keras.Sequential()

        # for layer in vgg16_model.layers[:-1]:
        #     layer.trainable = False
        #     model.add(layer)

        # model.layers.pop()
        #model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),  strides=(2, 2), padding='valid'))
        model.add(keras.layers.Flatten())
        #model.add(layers.Dense(2048, activation='softmax'))
        #model.add(layers.Dense(2048, activation='relu'))
        self.model = model




        

    def extract(self, image):
        
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        return self.model.predict(image)
