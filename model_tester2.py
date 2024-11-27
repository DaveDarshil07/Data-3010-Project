import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
background_image_path = "C:/Users/darshil/Desktop/Bground.jpg"  # Path to your background image
output_folder = "random_patches"  # Folder to save generated patches
results_file = "predicted_coordinates.csv"  # File to save coordinates
num_images = 10  # Number of random images to generate
patch_size = (128, 128)  # Size of each patch (128x128)

# --- Helper Functions ---

# Function to generate random patches from the background image
def generate_random_patches(background_image_path, output_folder, num_images, patch_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the background image
    bg_image = Image.open(background_image_path)
    bg_width, bg_height = bg_image.size

    for i in range(num_images):
        # Randomly choose the top-left corner of the patch
        x_start = random.randint(0, bg_width - patch_size[0])
        y_start = random.randint(0, bg_height - patch_size[1])

        # Crop the patch
        patch = bg_image.crop((x_start, y_start, x_start + patch_size[0], y_start + patch_size[1]))

        # Save the patch
        patch_path = os.path.join(output_folder, f"random_patch_{i + 1}.jpg")
        patch.save(patch_path)
        print(f"Saved patch: {patch_path}")

# Function to save coordinates to a file
def save_coordinates_to_file(file_name, coordinates):
    # Convert coordinates to a pandas DataFrame
    df = pd.DataFrame(coordinates, columns=["Image", "X", "Y"])
    df.to_csv(file_name, index=False)
    print(f"Saved coordinates to: {file_name}")

# --- Main Code ---
# Generate random patches
generate_random_patches(background_image_path, output_folder, num_images, patch_size)

# Load the model
model = load_model("C:/Users/darshil/Desktop/waldo_model.h5", compile=False)
model.compile(optimizer="adam", loss="mse", metrics=["mean_squared_error"])

# Prepare to save results
coordinates = []

# Process each patch
for image_file in os.listdir(output_folder):
    if image_file.endswith(".jpg"):
        # Load and preprocess the image
        image_path = os.path.join(output_folder, image_file)
        img = Image.open(image_path).resize((500 ,350))  # Resize to match model input
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict coordinates
        predictions = model.predict(img_array)
        print(f"Predictions for {image_file}: {predictions}")
        
        # Check if predictions are valid
        if isinstance(predictions, np.ndarray) and predictions.shape[1] == 2:  # Assuming [x, y] format
            x, y = predictions[0]
            coordinates.append([image_file, x, y])

        # Visualization
        plt.imshow(img)
        plt.axis("off")
        # Assuming predictions are coordinates [x, y]
        if isinstance(predictions, np.ndarray) and len(predictions[0]) == 2:
            x, y = predictions[0]
            coordinates.append([image_file, x, y])

            # Mark the prediction on the image
            plt.scatter(x, y, color="red", s=100, label="Waldo")
            plt.legend()
            plt.title(f"Predicted Coordinates: ({x:.2f}, {y:.2f})")
        else:
            print(f"No valid coordinates predicted for {image_file}")
            plt.title("No valid prediction")
            
        plt.show()

# Save coordinates to a file
save_coordinates_to_file(results_file, coordinates)
