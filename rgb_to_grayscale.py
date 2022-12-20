import streamlit as st
import tensorflow as tf
from PIL import Image
from io import BytesIO
from tensorflow.image import rgb_to_grayscale
from tensorflow.keras.utils import array_to_img, img_to_array

def save_image(image):
  buf = BytesIO()
  image.save(buf, format='png')
  byte_im = buf.getvalue()
  return byte_im

def convert_image(upload):
  image = Image.open(upload)
  rgb_image = image.convert("RGB")
  col1.write("Original Image :camera")
  col1.image(rgb_image)

  gray_image = array_to_img(rgb_to_grayscale(rgb_image))
  col2.write("Converted Image :wrench:")
  col2.image(gray_image)
  
  st.sidebar.markdown("\n")
  st.sidebar.download_button("Download converted image", save_image(gray_image), "gray_image.png", "image/png")
  
  
col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
  convert_image(upload=my_upload)
  
else:
  convert_image("./House design.png")
