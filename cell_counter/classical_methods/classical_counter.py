import cv2
import numpy as np

def count_cells_classical(pil_image):
    image = pil_image.convert("L")  # grayscale
    np_image = np.array(image)  # PIL → NumPy
    _, binary = cv2.threshold(np_image, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)