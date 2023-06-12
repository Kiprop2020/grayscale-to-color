import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.utils import array_to_img

# model_path = "C:/Users/user/My Drive/MACHINE LEARNING/GENERATIVE AI/image colorization/grayscale_to_rgb_model.keras"
model_path  = "J:/My Drive/image colorization/grayscale_to_rgb_model.keras"
# "J:/My Drive/image colorization/main.py"
model = tf.keras.models.load_model(model_path)

st.set_page_config(layout='wide', page_title='Image Colorizer')

st.write('## Add color to your black and white image')
st.write('Try uploading a grayscale/black and white image and see it conveted into a color image')
st.sidebar.write('Upload and download :gear:')

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = 512
def normalize_val_images_grayscale(image):
    return tf.cast(tf.image.rgb_to_grayscale(image), tf.float32)/255.

def normalize_val_images_rgb(image):
    return tf.cast(image, tf.float32)/255.

def preprocess_val_dataset(dataset):
    return dataset.batch(1).prefetch(AUTOTUNE)
 
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def color_image(upload):
    image = Image.open(upload)
    test_image_array = np.array(image)
    test_image_shape = test_image_array.shape[:2]

    resized_test_image_array = np.array(tf.image.resize(test_image_array, (IMAGE_SIZE, IMAGE_SIZE)))
    dataset = tf.data.Dataset.from_tensors(resized_test_image_array)

    normalized_grayscale_val_images = dataset.map(normalize_val_images_grayscale, num_parallel_calls=AUTOTUNE)
    for grayscale_image in normalized_grayscale_val_images:
        grayscale_image_array = np.array(tf.image.resize(grayscale_image, test_image_shape))
        grayscale_image = array_to_img(grayscale_image_array)
    col1.write('Original Image :camera:')
    col1.image(grayscale_image)
    preprocessed_grayscale_val_dataset = preprocess_val_dataset(normalized_grayscale_val_images)
   

    predicted_array = model.predict(preprocessed_grayscale_val_dataset)[0]
    resized_predicted_arrays = tf.image.resize(predicted_array, test_image_shape)
    predicted_image = array_to_img(resized_predicted_arrays)

    col2.write('Colored image :wrench:')
    col2.image(predicted_image)
    st.sidebar.markdown('\n')
    st.sidebar.download_button('Download colored image', convert_image(predicted_image), 'colored.png')

    


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader('Upload a grayscale image', accept_multiple_files=False, type=['jpg'])

if my_upload is not None:
    color_image(my_upload)
else:
    color_image('000000094326.jpg')