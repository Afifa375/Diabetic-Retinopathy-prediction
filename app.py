import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
  return tf.keras.models.load_model("model/dr_model.h5")
model = load_model()
classes = ["No DR (Healthy)", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
def preprocess_retinal_image(image):
  image = image.resize((128, 128))  # Change 128 to your model's actual input size
img_array = np.array(image) / 255.0
img_array = np.clip(img_array * 1.3, 0, 1)  # Boost contrast
img_array = np.clip(img_array - 0.05, 0, 1)  # Reduce brightness
img_array = np.expand_dims(img_array, axis=0)
return img_array

st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="wide")
st.title("🩺 Diabetic Retinopathy Detection")
st.markdown("Upload a retinal image to detect DR severity")

uploaded_file = st.file_uploader(
    "Choose a retinal image", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  col1, col2 = st.columns(2)
  
with col1:
  st.subheader("Uploaded Image")
  st.image(image, caption="Original Image", use_column_width=True)
  
with col2:
  st.subheader("Analysis")

with st.spinner("Processing image..."):
  processed_img = preprocess_retinal_image(image)
  predictions = model.predict(processed_img, verbose=0)
  predicted_class = np.argmax(predictions[0])
  confidence = predictions[0][predicted_class]
            
st.success(f"**Prediction:** {classes[predicted_class]}")
st.info(f"**Confidence:** {confidence:.1%}")
st.progress(float(confidence))
st.subheader("All Class Probabilities:")
for i, (cls, prob) in enumerate(zip(classes, predictions[0])):
  st.write(f"{cls}: {prob:.1%}")
if predicted_class >= 3:
  st.error("⚠️ **Medical Attention Recommended**")
  st.warning("Please consult an ophthalmologist.")
elif predicted_class == 0:
  st.balloons()
  st.success("✅ No signs of diabetic retinopathy detected!")
st.markdown("---")
st.markdown("**Note:** For educational purposes only. Consult a medical professional for diagnosis.")
