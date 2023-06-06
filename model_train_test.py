import os
import cv2
import numpy as np
import pytesseract
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical


# Step 1: Define functions for dataset loading and preprocessing

def load_dataset(label_to_index):
    images = []
    labels = []

    # Specify the directory containing the images
    directory = 'Kerasimages/Validate'

    # Loop through subdirectories
    index = 0

    for label in os.listdir(directory):
        # Full path to the subdirectory by joining the directory with the label
        subdirectory = os.path.join(directory, label)

        if os.path.isdir(subdirectory):
            # Get all image files in the subdirectory
            image_files = os.listdir(subdirectory)

            # Process each image file
            for image_file in image_files:
                # Create the full path to the image file
                image_path = os.path.join(subdirectory, image_file)

                # Load image
                image = cv2.imread(image_path)
                # If the image is None (failed to load), skip to the next image file
                if image is None:
                    continue

                processed_image = preprocess_image(image)  # preprocess image

                # Append processed image and label to the lists
                images.append(processed_image)

                # Check if the label is already present in the label_to_index dictionary.
                # If not, assign a new index to it and add the mapping to the dictionary.
                if label not in label_to_index:
                    label_to_index[label] = index
                    index += 1
                labels.append(label_to_index[label])

    # Convert lists to numpy arrays to be used with Keras
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def preprocess_image(image):
    # Preprocess the image (resize, normalize, etc.)
    resized_image = cv2.resize(image, (64, 64))  # Resize image to a fixed size
    normalized_image = resized_image / 255.0  # Normalize pixel values to the range [0, 1]
    return normalized_image

# Step 2: Load and preprocess the dataset

label_to_index = {}  # Dictionary to map label names to numerical indices

# Check if the saved model exists
if os.path.exists('model.h5'):
    # Load the model from disk
    model = load_model('model.h5')
    print('Loaded model from disk')

    # Load and preprocess the dataset
    images, labels = load_dataset(label_to_index)

    # Split the dataset into training and testing sets
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2,
                                                                            random_state=42)
else:
    # Load and preprocess the dataset
    images, labels = load_dataset(label_to_index)

    # Split the dataset into training and testing sets
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2,
                                                                            random_state=42)

    # Get the number of unique classes
    num_classes = len(np.unique(labels))

    # Convert labels to categorical format
    train_labels = to_categorical(train_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

    # Step 3: Build the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Step 4: Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10, batch_size=32)

    # Save the model to disk
    model.save('model.h5')
    print('Saved model to disk')

# Step 5: Predict on new images

new_image_paths = ['Kerasimages/Validate/R/Image00001_R2big.png']

# Preprocess and predict on new images
predictions = []
for new_image_path in new_image_paths:
    # Load and preprocess the new image
    new_image = cv2.imread(new_image_path)
    processed_new_image = preprocess_image(new_image)

    # Expand dimensions to match the input shape of the model
    processed_new_image = np.expand_dims(processed_new_image, axis=0)

    # Predict the label for the new image
    prediction = model.predict(processed_new_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)

    # Map the predicted class index to the class name
    predicted_class = [label for label, index in label_to_index.items() if index == predicted_class_index][0]

    predictions.append(predicted_class)

# Print the predicted labels with filenames for each image
for i in range(len(new_image_paths)):
    filename = os.path.basename(new_image_paths[i])
    print(f"Image {filename} is classified as {predictions[i]}")