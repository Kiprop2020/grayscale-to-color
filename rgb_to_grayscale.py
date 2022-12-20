import streamlit as st
import tensorflow as tf
from PIL import Image
from io import BytesIO
import numpy 
from tensorflow.image import rgb_to_grayscale
from tensorflow.keras.utils import array_to_img, img_to_array

def convert_image(image):
  buf = BytesIO()
  image.save(buf, format='png')
  byte_im = buf.getvalue()
  return byte_im

def convert_image(upload):
  with open(upload, 'rb') as f:
    image_bytes = f.read()
    image = tf.io.decode_png(image_bytes)
    
  col1.write("Original Image :camera")
  col1.image(image)

  converted = rgb_to_grayscale(image)
#   converted = tf.keras.utils.img_to_array(converted)
  converted = array_to_img(converted)
  col2.write("Fixed Image :wrench:")
  col2.image(converted)
  st.sidebar.markdown("\n")
  st.sidebar.download_button("Download converted image", convert_image(converted), "converted.png", "image/png")
  
  
col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
  convert_image(upload=my_upload)
  
else:
  convert_image("./House design.png")
