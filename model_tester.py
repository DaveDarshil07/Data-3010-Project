from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Option 1: Load without compilation
model = load_model("C:/Users/darshil/Desktop/waldo_model.h5", compile=False)

# Option 2: Recompile the model
model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

# Check the model's expected input shape
input_shape = model.input_shape  # Example: (None, 350, 500, 3)
print(f"Model expects input shape: {input_shape}")

# Load and preprocess the test image
image_path = r"C:/Users/darshil/Desktop/wheres_wally.jpg"  # Replace with your image path
img = load_img(image_path, target_size=(input_shape[1], input_shape[2]))  # Resize to match input shape
img_array = img_to_array(img) / 255.0  # Normalize the image
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Ensure the input image matches the model's input shape
if img_array.shape[1:] != input_shape[1:]:
    print(f"Resizing image to match model input shape {input_shape[1:]}")
    img_array = np.resize(img_array, (1, input_shape[1], input_shape[2], input_shape[3]))

# Run predictions
predictions = model.predict(img_array)

# Assuming predictions are coordinates (e.g., [[x, y]])
print(f"Predictions: {predictions}")

# Visualization
plt.imshow(img)
plt.axis('off')

# Overlay predictions on the image
if isinstance(predictions, np.ndarray) and len(predictions[0]) == 2:
    x, y = predictions[0]  # Assuming the model outputs [x, y] for the location
    plt.scatter(x, y, color='red', s=100, label="Waldo")  # Mark the predicted location
    plt.legend()
    plt.text(x + 5, y - 5, "Waldo?", color="red", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.title("Prediction Results")
plt.show()
