import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

new_image_path = 'Images/Image00004.png'
new_image = cv2.imread(new_image_path)

height, width, _ = new_image.shape
row_height = height // 3

for i in range(3):
    # Calculate the y-coordinate range for the current row
    start_y = i * row_height
    end_y = (i + 1) * row_height

    # Extract the row from the image
    row = new_image[start_y:end_y, :]

    # Convert the row to grayscale
    # gray_row = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)

    # Apply image preprocessing if necessary (e.g., thresholding, denoising)

    # Perform OCR using pytesseract
    text = pytesseract.image_to_string(row)

    # Print the detected text for the current row
    print(f"Row {i+1} Text: {text}")

