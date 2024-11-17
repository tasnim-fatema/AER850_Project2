# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# # Load the trained model
# model_path = './saved_model/my_model.h5'  # Adjust path if needed
# model = load_model(model_path)

"Load the Model"
Model = load_model("Project 2.keras")


# Define the test image paths as a list of tuples
test_images = [
    (r"C:\Users\16477\Documents\GitHub\AER850_Project2\Data\test\crack\test_crack.jpg", "Crack"), 
    (r"C:\Users\16477\Documents\GitHub\AER850_Project2\Data\test\missing-head\test_missinghead.jpg", "Missing Head"),
    (r"C:\Users\16477\Documents\GitHub\AER850_Project2\Data\test\paint-off\test_paintoff.jpg", "Paint Off")
]

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

# Function to display the image and predictions
def display(image_path, predictions, true_label, class_labels):
    predicted_label = class_labels[np.argmax(predictions)]
    fig, ax = plt.subplots(figsize=(6,6))
    img = plt.imread(image_path)
    plt.imshow(img)
    ax.axis('off')
    plt.title(f"True Label: {true_label}\n"
              f"Predicted Label: {predicted_label}\n")
    
    sorted_labels = sorted(zip(class_labels, predictions[0]), key=lambda x: x[1], reverse=True)
    for index, (label, percentage) in enumerate(sorted_labels):
        ax.text(
            10, 
            25 + index * 30,
            f"{label}: {percentage * 100:.2f}%",
            bbox=dict(facecolor="blue"),
            fontsize=10,
            color='white',
        )
    plt.tight_layout()
    plt.show()

# Iterate over the test images (each is a tuple of path and label)
for img_path, label in test_images:
    pred_class_idx, confidence, img_display = process_and_predict(img_path, Model)
    pred_label = class_labels[pred_class_idx]

    # # Display the image and prediction
    # plt.figure()
    # plt.imshow(img_display)
    # plt.axis('off')
    # plt.title(f"True Label: {label}\nPredicted: {pred_label} ({confidence:.2f}%)")
    # plt.show()
    
def plot_prediction(image, true_label, predicted_label, confidence):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(f"True Label: {true_label}\nPredicted: {predicted_label} ({confidence:.2f}%)")
    plt.show()
