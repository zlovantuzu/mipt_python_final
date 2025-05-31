import numpy as np
import cv2
import os
from datetime import datetime

def generate_synthetic_image(output_path, params):
    image = np.zeros((512, 512), dtype=np.uint8)
    num_cells = params.get("num_cells", 20)
    for _ in range(num_cells):
        x, y = np.random.randint(0, 512, 2)
        radius = np.random.randint(10, 20)
        cv2.circle(image, (x, y), radius, (255,), -1)
    filename = f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    full_path = os.path.join(output_path, filename)
    cv2.imwrite(full_path, image)
    return full_path, params