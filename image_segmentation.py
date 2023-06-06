import cv2
import numpy as np
from keras.models import load_model

# Step 1: Define functions for image splitting and prediction

def split_image(image):
    # Split the image into a 3x3 grid
    height, width, _ = image.shape
    split_height = height // 3
    split_width = width // 3

    grid = []
    for i in range(3):
        for j in range(3):
            # Extract the sub-image from the grid cell
            sub_image = image[i * split_height : (i+1) * split_height, j * split_width : (j+1) * split_width]
            grid.append(sub_image)

    return grid

def preprocess_image(image):
    # Preprocess the image (resize, normalize, etc.)
    resized_image = cv2.resize(image, (64, 64))  # Resize image to a fixed size
    normalized_image = resized_image / 255.0  # Normalize pixel values to the range [0, 1]
    return normalized_image

# Step 2: Load the pretrained model

model = load_model('model.h5')
print('Loaded pretrained model')

# Step 3: Predict on new image

new_image_path = 'Images/Image00004.png'

# Load and preprocess the new image
new_image = cv2.imread(new_image_path)
grid = split_image(new_image)

predictions = []
for sub_image in grid:
    processed_sub_image = preprocess_image(sub_image)

    # Expand dimensions to match the input shape of the model
    processed_sub_image = np.expand_dims(processed_sub_image, axis=0)

    # Predict the label for the sub-image
    prediction = model.predict(processed_sub_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)

    # Map the predicted class index to the class name
    predicted_class = [k for k, v in label_to_index.items() if v == predicted_class_index][0]

    predictions.append(predicted_class)

# Print the predicted labels for each sub-image in the grid
for i, prediction in enumerate(predictions):
    print(f"Grid cell {i+1} is classified as {prediction}")
