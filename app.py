import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/dr_model.h5")

model = load_model()

CLASS_NAMES = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]

st.title("Diabetic Retinopathy Prediction")

uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg","png","jpeg"])

def preprocess_image(image):
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]

    st.subheader("Prediction Result")
    st.write(f"**Class:** {CLASS_NAMES[class_idx]}")
    st.write(f"**Confidence:** {confidence:.2f}")
