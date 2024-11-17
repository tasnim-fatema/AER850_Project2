# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:05:45 2024

@author: 16477
"""

"STEP 5: MODEL Testing"

"Import Librarires"

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

"Load the Model"
Model = load_model("Project 2.keras")

"Define Input Image Shape"
Img_width = 500
Img_height = 500

"Define Class Labels"
class_labels = ['Crack', 'Missing-head', 'Paint-off']

"Data Preprocessing"
def preprocess (image_path, target_size = (Img_width, Img_height)):
    image = load_img (image_path, target_size = target_size)
    image_array = img_to_array (image)
    image_array = image_array/255
    image_array = np.expand_dims(image_array, 0)
    return image_array

"Data Prediction"
def predict (image_array, model):
    predictions = model.predict (image_array)
    return predictions

"Data Display"
def display (image_path, predictions, true_label, class_labels):
    predicted_label = class_labels [np.argmax(predictions)]
    fig, ax = plt.subplots(figsize = (6,6))
    img = plt.imread(image_path)
    plt.imshow(img)
    ax.axis('off')
    plt.title (f"True Crack Classification Label: {true_label}\n"
               f"Predicted Crack Classification Label: {predicted_label}\n")
    sorted_labels = sorted(zip(class_labels, predictions[0]))
    for index, (label,percentage) in enumerate(sorted_labels):
        ax.text(
            10, 
            25 + index * 30,
            f"{label}: {percentage * 100:.2f}%",
            bbox=dict(facecolor="blue"),
            fontsize=10,
            color='white',
            )
    plt.tight_layout()
    plt.show ()

"Test Specfic Images"
test_images = [
    (r"AER850_Project 2\Data\test\crack\test_crack.jpg", "Crack"),
    # (r"Project 2 Data\Data\test\missing-head\test_missinghead.jpg", "Missing Head"),
    # (r"Project 2 Data\Data\test\paint-off\test_paintoff.jpg", "Paint-Off")
    ]

for image_path, true_label in test_images:
    img_preprocess = preprocess(image_path)
    predictions = predict(img_preprocess, Model)
    display(image_path, predictions, true_label, class_labels)