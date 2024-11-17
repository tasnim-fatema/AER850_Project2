import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

## STEP 1
# Define the relative paths to data directories
train_dir = './train'
validation_dir = './valid'

# Define the input image size and batch size
target_size = (500, 500)  
batch_size = 32           

# Data augmentation for the training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
)

# Only rescale for validation and test data (no augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Print summary of data generator
print("Train generator:", train_generator.samples, "samples")
print("Validation generator:", validation_generator.samples, "samples")

#STEP 2
# Build the model

#DCNN Model 1
model = Sequential()

# First Convolutional Block with 16 filters
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(500, 500, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Second Convolutional Block with 32 filters
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3)) 

# Third Convolutional Block with 64 filters
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# Fourth Convolutional Block with 128 filters
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

# Fifth Convolutional Block with 256 filters
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))    

# Sixth Convolutional Block with 512 filters
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5)) 

# Flatten the output for fully connected layers
model.add(Flatten())

# Fully Connected Layer
# Dense layer with 64 units and ReLU activation
# Dropout (0.5) to further reduce overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output Layer
# Dense layer with 3 units (for 3 classes) and softmax activation
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

#DCNN Model 2

# # Deeper DCNN Model
# model_deep = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)),
#     MaxPooling2D((2, 2)),
    
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
    
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
    
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(3, activation='softmax')  # 3 classes
# ])

# model_deep.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model_deep.summary()


# #STEP 3

# # Define the CNN model with tuned hyperparameters
# model = Sequential([
#     # Convolutional Layer 1 with LeakyReLU for non-linearity
#     Conv2D(32, (3, 3), input_shape=(500, 500, 3)),
#     # LeakyReLU(alpha=0.1),
#     LeakyReLU(alpha=0.1),
#     MaxPooling2D((2, 2)),

#     # Convolutional Layer 2 with ReLU activation
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),

#     # Convolutional Layer 3 with LeakyReLU
#     Conv2D(128, (3, 3)),
#     LeakyReLU(alpha=0.1),
#     MaxPooling2D((2, 2)),

#     # Flatten and add Dense Layers with varied neurons and ELU activation
#     Flatten(),
#     Dense(128, activation='elu'),  # More neurons in dense layer
#     Dropout(0.5),
#     Dense(128, activation='relu'),  # ReLU for second dense layer
#     Dropout(0.5),

#     # Output layer with softmax for multi-class classification
#     Dense(3, activation='softmax')
#])

# STEP 3

# # Define the CNN model with tuned hyperparameters
# model = Sequential([
#     # Convolutional Layer 1 with ReLU for non-linearity
#     Conv2D(32, (3, 3), input_shape=(500, 500, 3)),
#     keras.layers.ReLU(),
#     MaxPooling2D((2, 2)),

#     # Convolutional Layer 2 with ReLU activation
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),

#     # Convolutional Layer 3 with ReLU
#     Conv2D(128, (3, 3)),
#     keras.layers.ReLU(),
#     MaxPooling2D((2, 2)),

#     # Flatten and add Dense Layers with varied neurons and ELU activation
#     Flatten(),
#     Dense(128, activation='elu'),  # ELU for first dense layer
#     Dropout(0.5),
#     Dense(128, activation='relu'),  # ReLU for second dense layer
#     Dropout(0.5),

#     # Output layer with softmax for multi-class classification
#     Dense(3, activation='softmax')
# ])

# Compile the model with categorical crossentropy loss and Adam optimizer
model.compile(optimizer=Adam(learning_rate=0.0001),  # Adjust learning rate as needed
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display model summary
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
history = model.fit(
    train_generator,
    epochs=25,  
    validation_data=validation_generator,
    # validation_steps=validation_generator,
    callbacks=[early_stopping]

)

import matplotlib.pyplot as plt

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color ='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

mode.save("Project 2.keras")






























