import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="COVID X-Ray Screening", layout="centered")

st.title("COVID X-Ray Screening")
st.write("Upload gambar X-ray paru-paru untuk melakukan prediksi.")

MODEL_PATH = "best_covid_model.keras"

class_labels = {
    0: "COVID",
    1: "Normal",
    2: "Viral Pneumonia"
}

@st.cache_resource
def load_covid_model():
    model = load_model(MODEL_PATH)
    return model

try:
    model = load_covid_model()
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error("Model gagal dimuat. Pastikan file best_covid_model.keras ada di repository.")
    st.error(e)
    st.stop()

uploaded_file = st.file_uploader(
    "Upload gambar X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        prediction = model.predict(img_array)

        predicted_index = int(np.argmax(prediction[0]))
        predicted_label = class_labels[predicted_index]
        confidence_score = float(prediction[0][predicted_index])

        st.subheader("Hasil Prediksi")
        st.write("Predicted Class:", predicted_label)
        st.write("Confidence Score:", round(confidence_score, 4))

        st.subheader("Probabilitas setiap class")
        for index, probability in enumerate(prediction[0]):
            st.write(class_labels[index], ":", round(float(probability), 4))
else:
    st.info("Silakan upload gambar terlebih dahulu.")
