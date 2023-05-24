import os
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import numpy as np
from keras.utils import to_categorical

# Step 1: Load the trained model
model_path = 'model.h5'
model = load_model(model_path)

# Step 2: Define a function to split an image into a 3x3 grid
def split_image(image):
    height, width, _ = image.shape
    grid_size = min(height, width) // 3
    images = []
    for i in range(3):
        for j in range(3):
            start_x = j * grid_size
            end_x = start_x + grid_size
            start_y = i * grid_size
            end_y = start_y + grid_size
            grid_image = image[start_y:end_y, start_x:end_x]
            images.append(grid_image)
    return images

# Step 3: Preprocess and predict on a single image
image_path = 'Images/Image00003.png'  # Provide the path to your image
image = cv2.imread(image_path)

# Split the image into a 3x3 grid
grid_images = split_image(image)

# Define class labels (modify according to your model's classes)
class_labels = ['A', 'Circle', 'Hexagon', 'L', 'P', 'R', 'Square', 'Triangle']
# Preprocess and predict on each grid image
predictions = []
for i, grid_image in enumerate(grid_images):
    processed_grid_image = cv2.resize(grid_image, (64, 64))  # Resize image to match model input shape
    processed_grid_image = processed_grid_image / 255.0  # Normalize pixel values to the range [0, 1]
    processed_grid_image = np.expand_dims(processed_grid_image, axis=0)
    prediction = model.predict(processed_grid_image)
    predicted_class_index = np.argmax(prediction)
    predictions.append(predicted_class_index)

    # Print the prediction for the current box
    print(f"Box {i+1} prediction: {class_labels[predicted_class_index]}")

# Visualize the original image with highlighted recognized parts and labels
highlighted_image = image.copy()
grid_size = min(image.shape[:2]) // 3
for i, prediction in enumerate(predictions):
    row = i // 3
    col = i % 3
    start_x = col * grid_size
    end_x = start_x + grid_size
    start_y = row * grid_size
    end_y = start_y + grid_size
    cv2.rectangle(highlighted_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    label = class_labels[prediction]
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(highlighted_image, label, (start_x, start_y + label_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the original image and highlighted regions
cv2.imshow("Highlighted Image", highlighted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


