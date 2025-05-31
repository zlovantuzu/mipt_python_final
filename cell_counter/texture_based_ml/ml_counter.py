from sklearn.cluster import KMeans
import cv2
import numpy as np

def count_cells_ml(pil_image):
    image = pil_image.convert("L")  # grayscale
    np_image = np.array(image)  # PIL → NumPy
    features = np_image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2).fit(features)
    segmented = kmeans.labels_.reshape(np_image.shape)
    binary = (segmented == segmented[0, 0]).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)