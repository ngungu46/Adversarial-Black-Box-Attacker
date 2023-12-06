# import torch

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

from PIL import Image

model_path = './EfficientNetB0_butterfly.h5'
img_path = './1.jpg'

image_size = 224

model = keras.models.load_model(model_path, custom_objects={'F1_score':'F1_score'})

rawimage = Image.open(img_path)

print(rawimage)

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

output = model.predict(image, steps=1)
print(output)
print(output.shape)
print(output.max(), output.argmax())