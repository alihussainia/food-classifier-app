from subprocess import call
call('pip install streamlit',shell=True)

import streamlit as st
@st.cache(allow_output_mutation=True)
def installing_reqs():
  call('pip install numpy',shell=True)
  call('pip install tensorflow-cpu',shell=True)
  call('pip install gdown',shell=True)
  call('pip install pillow',shell=True)
  call('pip install requests',shell=True)
  
installing_reqs()

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
import requests
from io import BytesIO
import gdown

@st.cache(allow_output_mutation=True)
def downloading_model():
  call('gdown --id 1FgnD8ixlLscDvFCHuq99TDYpMV66I-Mb',shell=True)

downloading_model()

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('bestmodel_3class.hdf5',compile = False)
    return model

with st.spinner('Loading Model Into Memory....'):
    model = load_model()

food_list = ['samosa','pizza','omelette'

uploaded_file = st.file_uploader("Upload Food Image to Classify....")

def processing(model, image):
  IMG_SIZE=[229,229]

  def read_image(image,IMG_SIZE):
      raw = tf.io.read_file(image)
      image = tf.image.decode_jpeg(raw, channels=3, dct_method='INTEGER_ACCURATE')
      image = tf.image.resize(image,IMG_SIZE, method='nearest')
      image = tf.cast(image, 'float32')
      return np.array(image)
  
  img = read_image(uploaded_file,IMG_SIZE)
  img = np.expand_dims(img, axis=0) 
  img = preprocess_input(img)
  return img

img_file_buffer = st.file_uploader("Upload Dog Image to Classify....")

if img_file_buffer  is not None:
    image = img_file_buffer
    image_out = Image.open(img_file_buffer)
    image = image.getvalue()
else:
    test_image = 'https://github.com/alihussainia/AI-Makerspace/raw/master/AI-GKE-Autopilot/images/pizza.jpg'
    image = requests.get(test_image).content
    image_out = Image.open(BytesIO(image))

st.write("Predicted Class :")
with st.spinner('classifying.....'):

st.write("Predicted Class :")
with st.spinner('classifying.....'):
    index =np.argmax(model.predict(processing(image)),axis=1)
    food_list.sort()
    pred_value = food_list[index]
    st.write(pred_value)    
st.write("")
st.image(image_out, caption='Classifying Food Image', use_column_width=True)