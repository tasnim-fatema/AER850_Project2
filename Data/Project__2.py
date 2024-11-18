import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

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

# Step 2 and Step 3: Neural Network Architecture Design and Hyperparameter Analysis

# Define the model architecture
model = Sequential([
    # First Convolutional Block: 16 Filters with ReLU
    Conv2D(16, (3, 3), activation='relu', input_shape=(500, 500, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Second Convolutional Block: 32 Filters with ReLU
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Third Convolutional Block: 64 Filters with Leaky ReLU
    Conv2D(64, (3, 3)),
    LeakyReLU(alpha=0.1),  # Only this block uses Leaky ReLU
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Fourth Convolutional Block: 128 Filters with ReLU
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    # Fifth Convolutional Block: 256 Filters with ReLU
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    # Flatten the Convolutional Output
    Flatten(),

    # Fully Connected Layer with ReLU
    Dense(64, activation='relu'),
    Dropout(0.5)
])



# # Fully Connected Layer
# # Dense layer with 64 units and ReLU activation
# # Dropout (0.5) to further reduce overfitting
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))

# Output Layer
# Dense layer with 3 units (for 3 classes) and softmax activation
model.add(Dense(3, activation='softmax'))


# Compile the model with categorical crossentropy loss and Adam optimizer
model.compile(optimizer=Adam(learning_rate=0.001),  # Adjust learning rate as needed
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display model summary
model.summary()


#STEP 4- Model Evaluation
# early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
history = model.fit(
    train_generator,
    epochs=50,  
    validation_data=validation_generator,

)

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

model.save("Project 2.keras")
print("Model saved successfully.")

# test_loss, test_accuracy = model.evaluate(test_generator)
# print(f"Test Accuracy: {test_accuracy:.2f}")





























