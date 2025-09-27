import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

# --- 1. CONFIGURATION ---
# Define the expected image size for the model
IMAGE_SIZE = (224, 224) 

# IMPORTANT: These labels MUST match the alphabetical order of your 
# dataset subdirectories, which is the default order used by 
# Keras's ImageDataGenerator during training.
CLASS_LABELS = [
    'Bird-Drop', 
    'Clean', 
    'Dusty', 
    'Electrical-Damage', 
    'Physical-Damage', 
    'Snow-Covered'
]

# --- 2. MODEL LOADING (FIXED) ---
@st.cache_resource
def load_classification_model(model_path):
    """Loads the trained Keras classification model, providing the necessary custom object."""
    try:
        # Define the custom objects needed to load the model with the Lambda layer (from your previous training)
        custom_objects = {
            'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input
        }
        
        # Load the model using the custom_objects dictionary
        model = load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading model: Could not find or load 'solar_panel_classifier.h5'. Please ensure it is in the same folder as app.py. Error: {e}")
        st.stop()

# --- 3. PREDICTION FUNCTION (NAMEERROR FIX) ---
def predict_image_class(model, image_array):
    """
    Preprocesses the image with MobileNetV2-specific normalization (-1 to 1) 
    and makes a classification prediction.
    """
    # 1. Resize the image
    img = Image.fromarray(image_array).resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32) # Keep original pixel values (0-255)
    
    # 2. Apply MobileNetV2-specific Preprocessing (CRUCIAL: 0-255 to -1 to 1)
    processed_img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # 3. Expand dimensions to create a batch (1, H, W, C)
    processed_img_array = np.expand_dims(processed_img_array, axis=0)

    # Predict
    predictions = model.predict(processed_img_array)
    
    # --- MISSING LOGIC TO DEFINE PREDICTED_CLASS AND CONFIDENCE ---
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100
    
    predicted_class = CLASS_LABELS[predicted_index]
    # ---------------------------------------------------------------
    
    return predicted_class, confidence

# --- 4. STREAMLIT APPLICATION LAYOUT ---

st.set_page_config(
    page_title="SolarGuard: Intelligent Defect Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("☀️ SolarGuard: Intelligent Defect Detection on Solar Panels")
st.markdown("---")
st.header("Upload a Solar Panel Image for Analysis")

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose an image file (JPG, JPEG, PNG)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Reset the file pointer to the beginning 
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    # Display columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Panel Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("AI Analysis Results")
        
        # Load the model once
        model = load_classification_model('solar_panel_classifier.h5')

        # Run Prediction
        with st.spinner('Analyzing image with Deep Learning Model...'):
            # The NameError is now resolved here because the function will return the variables
            predicted_class, confidence = predict_image_class(model, image_np) 

        st.success("✅ Analysis Complete!")
        st.markdown(f"**Predicted Condition:** <span style='font-size: 24px;'>{predicted_class}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.markdown("---")
        
        # Actionable Insights based on prediction
        st.subheader("Actionable Insight")
        
        if predicted_class == 'Clean':
            st.write("Panel is **CLEAN**. Continue regular monitoring to maintain peak efficiency.")
        elif predicted_class in ['Dusty', 'Bird-Drop', 'Snow-Covered']:
            st.warning("Panel requires **IMMEDIATE CLEANING**. Accumulation/coverage significantly reduces power output.")
        elif predicted_class in ['Electrical-Damage', 'Physical-Damage']:
            st.error("Panel shows **STRUCTURAL/ELECTRICAL DAMAGE**. Schedule immediate technical repair and detailed inspection. **HIGH PRIORITY.**")
        else:
            st.info("Prediction is inconclusive. Manual review recommended.")