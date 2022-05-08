from predictions import predict
import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

sns.set_theme(style="darkgrid")
sns.set()
st.title('Image Caption Generator')

def save_uploaded_file(uploaded_file):

    try:

        with open(os.path.join('uploaded',uploaded_file.name),'wb') as f:

            f.write(uploaded_file.getbuffer())

        return 1    

    except:

        return 0

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    # display the file
    display_image = Image.open(uploaded_file)
    display_image = display_image.resize((500,300))
    st.image(display_image)
    prediction = predict(os.path.join('uploaded',uploaded_file.name))
    print(prediction)
    os.remove('uploaded/'+uploaded_file.name)
    st.text(f'Predictions: {prediction.capitalize()}')
