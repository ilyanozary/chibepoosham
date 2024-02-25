
import cv2
import numpy as np
import tensorflow as tf

# Load the model from the saved file
model = tf.keras.models.load_model("my_model.h5")

# Define the image size and the class labels
IMG_SIZE = 50 # You can change this according to your model input size
CLASS_LABELS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"] # You can change this according to your model output classes

# Define a function to test the model using webcam
def test_model_webcam(model, webcam):
# Loop until the user presses q key
while True:
# Read the webcam frame
ret, frame = webcam.read()
# If the frame is not read correctly, break the loop
if not ret:
break
# Convert the frame to RGB color space
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# Resize the frame to the model input size
frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
# Convert the frame to a numpy array and normalize it
frame = np.array(frame) / 255.0
# Reshape the frame to match the model input shape
frame = frame.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
# Predict the class label using the model
prediction = model.predict(frame)
# Get the index of the highest probability
prediction = np.argmax(prediction)
# Get the corresponding class label
prediction = CLASS_LABELS[prediction]
# Display the prediction on the frame
cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# Convert the frame back to BGR color space
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
# Show the frame
cv2.imshow("Webcam Test", frame)
# Wait for 1 millisecond
key = cv2.waitKey(1)
# If the user presses q key, break the loop
if key == ord("q"):
    break
# Release the webcam and close the window
webcam.release()
cv2.destroyAllWindows()

# Load the webcam handler
webcam = cv2.VideoCapture(0)

# Test the model using the webcam
test_model_webcam(model, webcam)

