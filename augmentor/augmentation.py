import math
import os
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2


class ImageLoader:
    def __init__(self, directory):
        self.directory = directory
        self.images = []

    def load_images(self):
        self.images = []
        for file_name in os.listdir(self.directory):
            file_path = os.path.join(self.directory, file_name)
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                image = Image.open(file_path)
                self.images.append(image)
        return self.images


class AugmentationPipeline:
    def __init__(self):
        self.operations = []

    def apply(self, images):
        "Применяет последовательность операций к списку изображений."
        augmented_images = []
        for img in images:
            aug_img = img.copy()
            for operation in self.operations:
                aug_img = operation(aug_img)
            augmented_images.append(aug_img)
        return augmented_images

    def add_rotation(self, angle):
        "Добавляет операцию поворота."
        self.operations.append(lambda img: img.rotate(angle))

    def add_flip(self, horizontal=False, vertical=False):
        "Добавляет операцию горизонтального отражения."
        if horizontal:
            self.operations.append(lambda img: ImageOps.mirror(img))
        if vertical:
            self.operations.append(lambda img: ImageOps.flip(img))

    def add_noise(self, noise_level=10):
        "Добавляет аддитивный шум."

        def add_noise_to_image(img):
            np_img = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, noise_level, np_img.shape)
            noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_img)

        self.operations.append(add_noise_to_image)

    def remove_noise_mean(self):
        "Уменьшает шум с помощью усреднения."
        self.operations.append(lambda img: img.filter(ImageFilter.BLUR))

    def remove_noise_gaussian(self, radius=2):
        "Уменьшает шум с помощью фильтра Гаусса."
        self.operations.append(lambda img: img.filter(ImageFilter.GaussianBlur(radius)))

    def histogram_equalization(self):
        "Добавляет эквализацию гистограммы."

        def equalize(img):
            np_img = np.array(img)
            if len(np_img.shape) == 2:
                eq_img = cv2.equalizeHist(np_img)
            else:
                ycrcb = cv2.cvtColor(np_img, cv2.COLOR_RGB2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                eq_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            return Image.fromarray(eq_img)

        self.operations.append(equalize)

    def statistical_color_correction(self, strength=1.0):
        "Добавляет статистическую цветокоррекцию."

        def color_correct(img):
            enhancer = ImageEnhance.Color(img)
            return enhancer.enhance(strength)

        self.operations.append(color_correct)

    def add_translation(self, x_offset=50, y_offset=0):
        "Добавляет перенос"
        def translate(img):
            np_img = np.array(img)
            translation_matrix = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
            translated = cv2.warpAffine(np_img, translation_matrix, (np_img.shape[1], np_img.shape[0]))
            return Image.fromarray(translated)
        self.operations.append(translate)

    def add_rotation_wave1(self):
        "Добавляет эффект вона1"
        def rotation_wave(img):
            np_img = np.array(img)
            rows, cols, _ = np_img.shape
            distorted = np_img.copy()
            for i in range(rows):
                for j in range(cols):
                    x, y = int(min(i + 20 * math.sin(20 * math.pi * j / 60), rows - 1)), j
                    distorted[i][j] = np_img[x][y]
            return Image.fromarray(distorted)

        self.operations.append(rotation_wave)

    def add_rotation_wave2(self):
        "Добавляет эффект вона2"
        def rotation_wave2(img):
            np_img = np.array(img)
            rows, cols, _ = np_img.shape
            distorted = np_img.copy()
            for i in range(rows):
                for j in range(cols):
                    x, y = int(min(i + 20 * math.sin(20 * math.pi * i / 30), rows - 1)), j
                    distorted[i][j] = np_img[x][y]
            return Image.fromarray(distorted)

        self.operations.append(rotation_wave2)

    def add_glass_effect(self):
        "Добавляет эффект стекла."
        def glass_effect(img):
            np_img = np.array(img)
            rows, cols, _ = np_img.shape
            distorted = np_img.copy()
            for i in range(rows):
                for j in range(cols):
                    dx, dy = (np.random.randint(2) - 0.5) * 10, (np.random.randint(2) - 0.5) * 10
                    x, y = int(min(max(i + dy, 0), rows - 1)), int(min(max(j + dx, 0), cols - 1))
                    distorted[i][j] = np_img[x][y]
            return Image.fromarray(distorted)
        self.operations.append(glass_effect)

    def add_motion_blur(self, kernel_size=15):
        "Добавляет motion blur."
        def motion_blur(img):
            np_img = np.array(img)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel /= kernel_size
            blurred = cv2.filter2D(np_img, -1, kernel)
            return Image.fromarray(blurred)
        self.operations.append(motion_blur)


class ImageSaver:
    def __init__(self, directory):
        self.directory = directory

    def save_images(self, images):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        for i, img in enumerate(images):
            save_path = os.path.join(self.directory, f"augmented_{i + 1}.png")
            img.save(save_path)
