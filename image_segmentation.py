import cv2
import pytesseract


def classify_shape(contour):
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    num_sides = len(approx)

    if num_sides == 3:
        return "Triangle"
    elif num_sides == 4:
        return "Square"
    elif num_sides == 6:
        return "Hexagon"
    elif num_sides > 6:
        return "Circle"
    else:
        return "Unknown"


# Load the new image
new_image_path = 'Images/Image00004.png'
new_image = cv2.imread(new_image_path)

# Define the grid size
grid_size = 3

# Get the image dimensions
height, width, _ = new_image.shape

# Calculate the grid dimensions
grid_height = height // grid_size
grid_width = width // grid_size

# Initialize the output image with the new image
output_image = new_image.copy()

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to check if the given grid block is a center or corner block
def is_center_or_corner_block(i, j, grid_size):
    if (i == grid_size // 2 and j == grid_size // 2) or \
            (i == 0 and j == 0) or \
            (i == 0 and j == grid_size - 1) or \
            (i == grid_size - 1 and j == 0) or \
            (i == grid_size - 1 and j == grid_size - 1):
        return True
    return False

# Process each grid square
for i in range(grid_size):
    for j in range(grid_size):
        if is_center_or_corner_block(i, j, grid_size):
            continue
        # Define the coordinates of the grid square
        x = j * grid_width
        y = i * grid_height
        x_end = (j + 1) * grid_width
        y_end = (i + 1) * grid_height

        # Extract the grid square from the image
        grid_square = new_image[y:y_end, x:x_end, :]

        # # Show the subgrid square for debugging
        # cv2.imshow(f"Grid {i + 1},{j + 1}", grid_square)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Split the grid square into a 3x3 grid
        subgrid_size = 3
        subgrid_width = grid_width // subgrid_size
        subgrid_height = grid_height // subgrid_size

        # Process each subgrid
        for si in range(subgrid_size):
            for sj in range(subgrid_size):
                # Get the subgrid square coordinates relative to the main grid square
                sx = x + sj * subgrid_width
                sy = y + si * subgrid_height
                sx_end = x + (sj + 1) * subgrid_width
                sy_end = y + (si + 1) * subgrid_height

                # Extract the subgrid square from the grid square
                subgrid_square = new_image[sy:sy_end, sx:sx_end, :]

                # Convert the subgrid square to grayscale
                # gray_subgrid = cv2.cvtColor(subgrid_square, cv2.COLOR_BGR2GRAY)

                # Apply thresholding to extract the shape
                _, thresh = cv2.threshold(subgrid_square, 128, 255, cv2.THRESH_BINARY)

                # Perform OCR on the subgrid square
                letter = pytesseract.image_to_string(thresh, config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ')

                # Write the recognized letter on the subgrid square
                cv2.putText(subgrid_square, letter, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1,
                            cv2.LINE_AA)

                # Show the preprocessed subgrid square
                # cv2.imshow("Preprocessed Subgrid", subgrid_square)
                # cv2.waitKey(0)

                # Check if it is the center subgrid
                if si == subgrid_size // 2 and sj == subgrid_size // 2:
                    # Convert the subgrid square to grayscale
                    gray_subgrid = cv2.cvtColor(subgrid_square, cv2.COLOR_BGR2GRAY)

                    # Apply thresholding to extract the shape
                    _, thresh = cv2.threshold(gray_subgrid, 128, 255, cv2.THRESH_BINARY)

                    # Find contours in the thresholded image
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Check if any contour is found
                    if len(contours) > 0:
                        # Get the largest contour by area
                        contour = max(contours, key=cv2.contourArea)

                        # Classify the shape based on the number of sides
                        shape = classify_shape(contour)

                        # Draw the shape label on the subgrid square
                        cv2.putText(subgrid_square, shape, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1,
                                    cv2.LINE_AA)

                # Update the coordinates to match the grid square position
                sx_output = x + sj * subgrid_width
                sy_output = y + si * subgrid_height
                sx_end_output = x + (sj + 1) * subgrid_width
                sy_end_output = y + (si + 1) * subgrid_height

                # Add the subgrid square with shape label and letter to the output image
                output_image[sy_output:sy_end_output, sx_output:sx_end_output, :] = subgrid_square

                # Draw a rectangle around the subgrid square
                cv2.rectangle(output_image, (sx_output, sy_output), (sx_end_output, sy_end_output), (0, 255, 0), 1)

# Show the output image
cv2.imshow("Output Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
