# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model_path = './saved_model/my_model.h5'  # Adjust path if needed
model = load_model(model_path)

# Define the test image paths
test_images = {
    "Crack": "./test/crack/test_crack.jpg",
    "Missing Head": "./test/missing-head/test_missinghead.jpg",
    "Paint Off": "./test/paint-off/test_paintoff.jpg"
}

# Function to preprocess and predict the class of an image
def process_and_predict(image_path, model):
    # Load the image with the target size matching the model input
    img = image.load_img(image_path, target_size=(500, 500))
    
    # Convert the image to an array and normalize
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)  # Find the class with highest probability
    confidence = np.max(predictions) * 100  # Confidence score

    return predicted_class[0], confidence, img

# Class labels (ensure they match your training data)
class_labels = ["Crack", "Missing Head", "Paint Off"]

# Iterate over the test images and make predictions
for label, img_path in test_images.items():
    pred_class_idx, confidence, img_display = process_and_predict(img_path, model)
    pred_label = class_labels[pred_class_idx]

    # Display the image and prediction
    plt.figure()
    plt.imshow(img_display)
    plt.axis('off')
    plt.title(f"True Label: {label}\nPredicted: {pred_label} ({confidence:.2f}%)")
    plt.show()
