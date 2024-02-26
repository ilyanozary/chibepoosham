import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(
    'body_shape_detection_model2.h5')

# Define class labels based on your dataset
class_labels = ['0', '1', '2', '3', '4', '5', '11', '21', '31', '41', '51']  # class labels


# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))  # Resize image to match model's input size
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image


# Function to predict class of an image
def predict_image_class(image_path):
    preprocessed_image = preprocess_image(image_path)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label


# Path to the image you want to predict
image_path = 'imgpsh_fullsize_anim.jpeg'

# Predict the class of the image
predicted_class = predict_image_class(image_path)
print('Predicted class:', predicted_class)
