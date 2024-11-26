import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Define the CNN model
def create_model():
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))  # Define input shape
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(GlobalAveragePooling2D())  # Use GlobalAveragePooling2D instead of Flatten
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the CNN model
model = create_model()

# Print model summary to verify dimensions
model.summary()

# Data Preparation
# Update the path to your dataset directory
data_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, 
                              height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, 
                              horizontal_flip=True, fill_mode='nearest', validation_split=0.2)

train_data = data_gen.flow_from_directory('C:/Users/daved/Downloads/dataset/', target_size=(224, 224), batch_size=32, class_mode='binary', subset='training')
validation_data = data_gen.flow_from_directory('C:/Users/daved/Downloads/dataset/', target_size=(224, 224), batch_size=32, class_mode='binary', subset='validation')

# Train the model, can change the epochs value according to system GPU support
history = model.fit(train_data, validation_data=validation_data, epochs=10)

# Save the model
model.save("finding_waldo_cnn_model.h5")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
