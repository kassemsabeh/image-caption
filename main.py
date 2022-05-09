from predictions import predict
import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
from gtts import gTTS
import IPython.display as ipd

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
    prediction = predict(display_image)
    pred_button = st.button('Generate Caption')
    audio_button = st.button('Play')
    if pred_button:
        st.text(f'Predictions: {prediction.capitalize()}')
        
    if audio_button:
        tts = gTTS(text=prediction)
        tts.save('audio.mp3')
        audio_file = open('audio.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)

    