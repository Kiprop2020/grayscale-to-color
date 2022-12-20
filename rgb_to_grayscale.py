import tensorflow as tf
from PIL import Image

input_source = "C:/Users/negehiza/OneDrive - NACC/Pictures/example.jpg"
output_source = "C:/Users/negehiza/OneDrive - NACC/Pictures/example.jpg"

image = Image.open(input_source)
image = tf.image.rgb_to_grayscale(image)
image = tf.keras.utils.img_to_array(image)
image = tf.keras.utils.array_to_img(image)
image.save(output_source, format=None)
Image.open(output_source)
