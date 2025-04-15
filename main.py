import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Prediction function
def model_prediction(test_image):
    try:
        # Load the trained model
        model = tf.keras.models.load_model('trained_plant_disease_model.keras')

        # Preprocess the image (assuming model expects 64x64x3)
        image = Image.open(test_image).convert("RGB").resize((128,128))  # Resize and ensure 3 channels
        input_arr = np.array(image) / 255.0  # Normalize to [0,1]
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        return result_index

    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

# Streamlit App UI
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("🌿 Plant Disease Recognition System")
    st.image("Home_page.jpg", use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍  
    Upload an image of a plant to detect any diseases.

    ### How It Works
    - **Upload Image**: Go to the **Disease Recognition** page.
    - **Analysis**: Our system will analyze the image using AI.
    - **Results**: Get instant results with recommendations.

    👉 Choose **Disease Recognition** from the sidebar to get started!
    """)

# About Page
elif app_mode == "About":
    st.header("📊 About the Project")
    st.markdown("""
    This system uses a Convolutional Neural Network (CNN) model trained on the PlantVillage dataset.

    - **Training Data**: 70,295 images  
    - **Validation Data**: 17,572 images  
    - **Test Data**: 33 images  

    Our goal is to help farmers detect plant diseases efficiently using AI technology.
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("🩺 Disease Recognition")
    test_image = st.file_uploader("Upload an image of the plant:", type=["jpg", "jpeg", "png"])

    if test_image:
        st.image(test_image, use_container_width=True)

        if st.button("Predict Disease"):
            with st.spinner("Analyzing Image..."):
                result_index = model_prediction(test_image)

                if result_index is not None:
                    class_names = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
                        'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
                        'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca', 'Grape___Leaf_blight',
                        'Grape___healthy', 'Orange___Citrus_greening', 'Peach___Bacterial_spot', 'Peach___healthy',
                        'Pepper___Bacterial_spot', 'Pepper___healthy', 'Potato___Early_blight',
                        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
                        'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
                        'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                        'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                        'Tomato___Spider_mites', 'Tomato___Target_Spot',
                        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                    ]

                    st.success(f"✅ Model Prediction: **{class_names[result_index]}**")
