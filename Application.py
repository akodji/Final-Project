import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load your GAN model
def load_model():
    # Replace with your model loading code
    model = tf.keras.models.load_model('path/to/your/model.h5')
    return model

# Generate a similar image
def generate_image(model, input_image):
    # Preprocess the input image
    input_image = np.array(input_image.resize((256, 256)))  # Resize to the model's input size
    input_image = (input_image / 255.0).astype(np.float32)  # Normalize
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    
    # Generate the image
    generated_image = model.predict(input_image)
    generated_image = (generated_image.squeeze() * 255).astype(np.uint8)  # Rescale to original range
    return Image.fromarray(generated_image)

# Streamlit app
def main():
    st.title("Image Similarity Generator")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Uploaded Image", use_column_width=True)
        
        model = load_model()
        with st.spinner("Generating similar image..."):
            generated_image = generate_image(model, input_image)
        
        st.image(generated_image, caption="Generated Similar Image", use_column_width=True)

if __name__ == "__main__":
    main()
