
    
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')
#     plt.title(f"True Label: {true_label}\nPredicted: {predicted_label} ({confidence:.2f}%)")
#     plt.show()
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model("Project 2.keras")

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
    confidence = np.max(predictions) * 100  # Confidence score
    predicted_class = np.argmax(predictions, axis=1)[0]  # Find the class with highest probability

    return predicted_class, confidence, img

# Class labels (ensure they match your training data)
class_labels = ["Crack", "Missing Head", "Paint Off"]

# Function to display the image and predictions
def display(image_path, true_label, predictions, class_labels):
    # Get predicted label and sort confidence scores
    predicted_label = class_labels[np.argmax(predictions)]
    is_correct = (predicted_label == true_label)
    sorted_predictions = sorted(zip(class_labels, predictions.flatten()), key=lambda x: x[1], reverse=True)

    # Load and display the image
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.axis('off')

    # Display title with true and predicted labels in green if correct, red if not
    title_color = 'green' if is_correct else 'red'
    plt.title(
        f"True Label: {true_label}\nPredicted: {predicted_label} ({np.max(predictions) * 100:.2f}%)",
        color=title_color,
        fontsize=14,
    )

    # Show prediction confidence for all classes
    confidence_text = "\n".join(
        [f"{label}: {score * 100:.2f}%" for label, score in sorted_predictions]
    )
    plt.gcf().text(
        0.02, 0.02, confidence_text, fontsize=10, color='white',
        bbox=dict(facecolor='black', alpha=0.8, edgecolor='none')
    )
    plt.show()

# Iterate over the test images (each is a tuple of path and label)
for img_path, label in test_images:
    pred_class_idx, confidence, img_display = process_and_predict(img_path, model)
    pred_label = class_labels[pred_class_idx]

    # Prepare predictions array for display function
    pred_array = np.zeros(len(class_labels))
    pred_array[pred_class_idx] = confidence / 100.0  # Convert to fraction for visualization

    # Display the image and prediction
    display(img_path, label, pred_array, class_labels)
