from subprocess import call
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
import requests
from io import BytesIO
import gdown
import os

# Requirements:
# People can upload or simply enter the URL of image using streamlit
# then we can convert the image to the required format of predict_class function.
# plus to save the image


# downloading google drive
@st.cache(allow_output_mutation=True)
def downloading_model():
  call('gdown --id 1FgnD8ixlLscDvFCHuq99TDYpMV66I-Mb',shell=True)

downloading_model()

# loading model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('bestmodel_3class.hdf5',compile = False)
    return model

with st.spinner('Loading Model Into Memory....'):
    model = load_model()

# target classes
food_list = ['samosa','pizza','omelette']

path = st.text_input('Enter Image URL to Classify...')
img_file_buffer = st.file_uploader("Upload Your Food Image...")

if img_file_buffer:
    image = img_file_buffer
    image_out = Image.open(img_file_buffer)
    st.image(image_out, caption='Your Image', use_column_width=False, width=400)
    image_out.save("input.jpg")
else:
  if path:
      test_image = repr(path)
      image_url_content = requests.get(test_image).content
      image_out = Image.open(BytesIO(image_url_content))
      st.image(image_out, caption='Your Image', use_column_width=False, width=400)
      image_out.save("input.jpg")
  else:
      path=None
      img_file_buffer=None

st.write(f'(os.listdir($(pwd))}')
# requires an img path -> img
def preprocess_input_image(img):
    img = image.load_img(img, target_size=(299, 299))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img = preprocess_input(img)                                      
    return img

def predict_output(model,img):
  pred = model.predict(img)
  index = np.argmax(pred)
  food_list.sort()
  pred_value = food_list[index]
    
if st.button('Submit'):
    if img_file_buffer is None and path is None:
        st.error("Please Upload Your Image")
    else:
      with st.spinner('classifying.....'):
          img = st.upload.read('input.jpg')
          img = preprocess_input_image(img)
          pred_value=predict_output(model,img)
          st.write(pred_value)    
