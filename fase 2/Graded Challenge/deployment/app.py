# Import necessary libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import eda  # Halaman EDA
import prediction  # Halaman Predictor

# Sidebar navigation
navigation = st.sidebar.selectbox('Halaman:', ('Prediction', 'EDA'))

# Streamlit app
def main():
    # Predictor page
    if navigation == 'Prediction':
        st.title("Waste Image Classification - Predictor")
        
        # Load the trained model
        model = load_model('model_2.keras')
        target_size = (128, 128)

        # Function to import and predict
        def import_and_predict(image_data, model):
            image = load_img(image_data, target_size=target_size)
            img_array = img_to_array(image)
            img_array = tf.expand_dims(img_array, 0)

            # Normalize the image
            img_array = img_array / 255.0

            # Make prediction
            predictions = model.predict(img_array)

            # Get the class with the highest probability
            predicted_class = np.argmax(predictions)

            # Define class labels
            class_labels = ['Cardboard', 'Paper', 'Plastic']

            # Get the predicted label
            predicted_label = class_labels[predicted_class]

            result = f"Prediction: {predicted_label}"

            return result

        # File uploader for prediction
        file = st.file_uploader("Upload an image", type=["jpg", "png"])

        if file is None:
            st.text("Please upload an image file")
        else:
            result = import_and_predict(file, model)
            st.image(file)
            st.write(result)

    # EDA page
    elif navigation == 'EDA':
        st.title("Waste Image Classification - EDA")
        eda.run()  

# Entry point
if __name__ == "__main__":
    main()
