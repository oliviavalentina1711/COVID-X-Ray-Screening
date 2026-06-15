import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)

@st.cache_resource
def load_prediction_model():
    model = load_model("best_covid_model.keras")
    return model

@st.cache_data
def load_class_mapping():
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    index_to_class = {int(v): k for k, v in class_indices.items()}
    return index_to_class

def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image, image_array

st.title("COVID-19 X-ray Image Classification")
st.write("Upload an X-ray image and the model will predict the class.")

model = load_prediction_model()
index_to_class = load_class_mapping()

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image, processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)

    predicted_index = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))
    predicted_label = index_to_class[predicted_index]

    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.subheader("Prediction Result")
    st.write(f"Predicted label: {predicted_label}")
    st.write(f"Confidence score: {confidence:.2%}")

    st.write("Class probabilities:")
    for idx, probability in enumerate(prediction[0]):
        st.write(f"{index_to_class[idx]}: {probability:.2%}")