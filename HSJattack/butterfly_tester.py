# import torch

import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

import os
# os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

import h5py

from PIL import Image

model_path = './EfficientNetB0_butterfly.h5'
img_path = '../data/butterfly/test/EASTERN COMA/5.jpg'

image_size = 224

f = h5py.File('./EfficientNetB0_butterfly.h5', 'r')
print(f.attrs.get('keras_version'))

model = keras.models.load_model(model_path, custom_objects={'F1_score':'F1_score'})

rawimage = Image.open(img_path)

# print(rawimage)

# image = tf.keras.preprocessing.image.img_to_array(rawimage)

# image = tf.cast(image, tf.float32)
# # image = image/255
# image = tf.image.resize(image, (image_size, image_size))
# image = image[None, ...]

image = tf.keras.preprocessing.image.img_to_array(rawimage)

image = tf.cast(image, tf.float32)
# image = image/255
image = tf.image.resize(image, (image_size, image_size))
image = image[None, ...]

print(image)

# while True:
output = model.predict(image, steps=1)
print(type(output))
print(output.shape)
print(output.max(), output.argmax())

# print(model(image))
# print(model.summary())
print(output.min())