import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Load trained model
# ---------------------------
MODEL_PATH = "solar_panel_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------------
# Class labels (EDIT if needed)
# ---------------------------
class_names = [
    "Bird-Drop",
    "Clean",
    "Dusty",
    "Electrical-Damage",
    "Physical-Damage",
    "Snow-Covered"
]

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="SolarGuard", layout="centered")
st.title("☀️ SolarGuard – Solar Panel Defect Detection")

st.write("Upload a solar panel image to predict its condition.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------
# Prediction logic
# ---------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (same as training)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    st.success(f"🧠 Prediction: **{class_names[class_index]}**")
    st.info(f"📊 Confidence: **{confidence:.2f}**")
