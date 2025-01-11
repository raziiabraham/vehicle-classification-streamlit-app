import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load the model in `.h5` format
model = tf.keras.models.load_model('vehicle_model.h5')  # Path to your model

# Define class labels
class_labels = ['NonVehicle', 'Vehicle']

# Streamlit app
st.title("Vehicle Classification App")
st.write("Upload an image to classify it as a vehicle or non-vehicle.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # Decode image using OpenCV

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Resize and preprocess the image
    test_image = cv2.resize(image, (128, 128))  # Resize to match the input size
    test_image = np.reshape(test_image, [1, 128, 128, 3])  # Add batch dimension

    # Normalize pixel values (optional, if required by your model)
    test_image = test_image / 255.0  # Scale to [0, 1] if needed

    # Make prediction
    prediction = model.predict(test_image)
    st.write(f"Raw predictions: {prediction}")  # Debugging: show raw output

    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display results
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
