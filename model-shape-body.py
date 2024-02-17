# Import the required libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Define the hyperparameters
num_classes = 5  # The number of body shape classes
img_height = 150  # The height of the input images
img_width = 150  # The width of the input images
batch_size = 32  # The size of the mini-batches for training
epochs = 100  # The number of epochs for training
learning_rate = 0.001  # The learning rate for the optimizer

# Create an image data generator for data augmentation
datagen = ImageDataGenerator(
    rescale=1. / 255,  # Rescale the pixel values to [0,1] range
    rotation_range=20,  # Rotate the images randomly by 20 degrees
    width_shift_range=0.1,  # Shift the images horizontally by 10% of the width
    height_shift_range=0.1,  # Shift the images vertically by 10% of the height
    horizontal_flip=True,  # Flip the images horizontally
    zoom_range=0.1  # Zoom the images by 10%
)

# Create a training data generator from the train directory
train_generator = datagen.flow_from_directory(
    "evaluation_dataset",  # The path to the train directory
    target_size=(img_height, img_width),  # The target size of the images
    batch_size=batch_size,  # The batch size
    class_mode="categorical"  # The class mode (one-hot encoded vectors)
)


# Define the model architecture
model = keras.Sequential([
    # The first convolutional layer with 32 filters and 3x3 kernel size
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, 3)),
    # The first max pooling layer with 2x2 pool size
    layers.MaxPooling2D((2, 2)),
    # The second convolutional layer with 64 filters and 3x3 kernel size
    layers.Conv2D(64, (3, 3), activation="relu"),
    # The second max pooling layer with 2x2 pool size
    layers.MaxPooling2D((2, 2)),
    # The third convolutional layer with 128 filters and 3x3 kernel size
    layers.Conv2D(128, (3, 3), activation="relu"),
    # The third max pooling layer with 2x2 pool size
    layers.MaxPooling2D((2, 2)),
    # The flatten layer to convert the 3D feature maps to 1D feature vectors
    layers.Flatten(),
    # The first dense layer with 256 units and ReLU activation
    layers.Dense(256, activation="relu"),
    # The dropout layer with 0.5 dropout rate
    layers.Dropout(0.5),
    # The output layer with num_classes units and softmax activation
    layers.Dense(num_classes, activation="softmax")
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),  # The optimizer
    loss=keras.losses.CategoricalCrossentropy(),  # The loss function
    metrics=["accuracy"]  # The metric to monitor
)

# Print the model summary
model.summary()

# Train the model
model.fit(
    train_generator,  # The training data generator
    epochs=epochs,  # The number of epochs
    validation_data=train_generator  # The validation data generator
)

#saved_model

model.save("model.keras")

