import sys
import cv2
import pytesseract
import time
import numpy as np
from PIL import Image

# Function to perform OCR on the image
def recognize_text(image_path):
    try:
        # Open the image using PIL (Pillow)
        with Image.open(image_path) as img_pil:
            # Convert PIL image to OpenCV format
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Convert image to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Use Tesseract to do OCR on the image
        start_time = time.time()
        text = pytesseract.image_to_string(gray)
        end_time = time.time()

        # Calculate execution time
        execution_time = end_time - start_time

        return text, execution_time

    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Example usage
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python handwriting_recognition.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    text, execution_time = recognize_text(image_path)

    if text is not None:
        print(f'Recognized Text:\n{text}')
        print(f'Execution Time: {execution_time:.2f} seconds')
    else:
        print("Text recognition failed.")
