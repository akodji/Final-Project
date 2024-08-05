import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to load and preprocess images
def load_image(image_file, max_dim=512):
    img = Image.open(image_file)
    img = img.convert('RGB')
    img = np.array(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (max_dim, max_dim), preserve_aspect_ratio=True)
    img = img[tf.newaxis, :]
    return img

# Load the style transfer model from TensorFlow Hub
@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Function to apply style transfer
def apply_style_transfer(model, content_image, style_image):
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image

# Streamlit app layout
st.title("Style Transfer App")
st.write("Upload a content image and a style image to generate a stylized image.")

# Upload content and style images
content_image_file = st.file_uploader("Choose a content image...", type=["jpg", "png", "jpeg"])
style_image_file = st.file_uploader("Choose a style image...", type=["jpg", "png", "jpeg"])

if content_image_file is not None and style_image_file is not None:
    # Load and preprocess the images
    content_image = load_image(content_image_file)
    style_image = load_image(style_image_file)

    st.image(content_image[0], caption="Content Image", use_column_width=True)
    st.image(style_image[0], caption="Style Image", use_column_width=True)
    
    # Load the model
    model = load_model()

    # Apply style transfer
    with st.spinner("Generating stylized image..."):
        stylized_image = apply_style_transfer(model, content_image, style_image)

    # Display the stylized image
    st.image(stylized_image[0], caption="Stylized Image", use_column_width=True)

    # (Optional) Display evaluation metrics
    content_loss = tf.reduce_mean(tf.square(stylized_image - content_image))
    st.write(f"Content Loss: {content_loss.numpy()}")

st.write("This application uses a pre-trained TensorFlow Hub model to perform style transfer.")
