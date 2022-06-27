import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from subprocess import call

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
def predict_class(model, uploaded_file):
  bytes_data = uploaded_file.getvalue()
  img = image.load_img(bytes_data, target_size=(299, 299))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0) 
  img = preprocess_input(img) 
  pred = model.predict(img)
  index = np.argmax(pred)
  food_list.sort()
  pred_value = food_list[index]
  return pred_value

st.write(predict_class(model,uploaded_file))