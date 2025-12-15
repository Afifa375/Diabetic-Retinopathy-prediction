import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="Diabetic Retinopathy Classifier", layout="centered")
st.title("Diabetic Retinopathy Detection")

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    # Load the Keras native model
    model = tf.keras.models.load_model("my_model.keras")  
    return model

model = load_model()

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a retina image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image for model
    IMG_SIZE = (64, 64)  # same size as during training
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    
    # Map index to class name (update if your dataset has different labels)
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    st.write(f"Prediction: **{class_names[class_idx]}**")
    st.write(f"Confidence: **{np.max(prediction)*100:.2f}%**")

