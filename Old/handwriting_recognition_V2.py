import sys
import cv2
import pytesseract
import time
import numpy as np
from PIL import Image
from threading import Thread
from multiprocessing import Process, Queue

# Function to perform OCR on the image
def recognize_text(image_path, processing_type):
    try:
        # Open the image using PIL (Pillow)
        with Image.open(image_path) as img_pil:
            # Convert PIL image to OpenCV format
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Convert image to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        if processing_type == 'heavy':
            return heavy_processing(gray)
        elif processing_type == 'fast':
            return fast_processing(gray)
        else:
            raise ValueError("Invalid processing type. Choose 'heavy' or 'fast'.")

    except Exception as e:
        print(f"Error: {e}")
        return None, None

def heavy_processing(gray):
    # Simulate heavy processing
    start_time = time.time()
    
    # Denoising
    gray = cv2.fastNlMeansDenoising(gray, h=30)

    # Binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Skew correction
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Perform OCR on the processed image
    text = pytesseract.image_to_string(binary)

    end_time = time.time()
    execution_time = end_time - start_time
    return text, execution_time

def fast_processing(gray):
    queue = Queue()

    def worker(queue, img_part):
        text = pytesseract.image_to_string(img_part)
        queue.put(text)

    start_time = time.time()

    # Split the image into parts
    height, width = gray.shape
    parts = [gray[i * height // 4: (i + 1) * height // 4, :] for i in range(4)]

    # Create multiple worker threads or processes
    threads = [Thread(target=worker, args=(queue, part)) for part in parts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    end_time = time.time()
    execution_time = end_time - start_time

    # Collect results from the queue
    text = ""
    while not queue.empty():
        text += queue.get()

    return text, execution_time

# Example usage
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python handwriting_recognition.py <image_path> <processing_type>")
        sys.exit(1)

    image_path = sys.argv[1]
    processing_type = sys.argv[2]
    text, execution_time = recognize_text(image_path, processing_type)

    if text is not None:
        print(f'Recognized Text:\n{text}')
        print(f'Execution Time: {execution_time:.2f} seconds')
    else:
        print("Text recognition failed.")
