from tkinter import Tk, Label, Button, filedialog, Frame, Scale, IntVar, Checkbutton, Canvas
from augmentor.augmentation import ImageLoader, AugmentationPipeline, ImageSaver
from PIL import ImageTk


class ImageAugmentorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Итоговая работа")

        # UI Frames
        self.left_frame = Frame(root, width=200, bg="lightgray")
        self.left_frame.pack(side="left", fill="y")

        self.right_frame = Frame(root, width=400)
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.images = []
        self.augmented_images = []
        self.pipeline = AugmentationPipeline()

        self.rotation_var = IntVar()
        self.flip_var = IntVar()
        self.flip_v_var = IntVar()

        # UI: левая панель
        Label(self.left_frame, text="Инструмены", bg="lightgray", font=("Arial", 12, "bold")).pack(pady=10)
        Button(self.left_frame, text="Поворот", command=self.show_rotation_controls).pack(fill="x", pady=5)
        Button(self.left_frame, text="Отражение", command=self.show_flip_controls).pack(fill="x", pady=5)
        Button(self.left_frame, text="Шум", command=self.show_noise_controls).pack(fill="x", pady=5)
        Button(self.left_frame, text="Удаление шума", command=self.show_denoise_controls).pack(fill="x", pady=5)
        Button(self.left_frame, text="Гисторгаммы", command=self.show_histogram_controls).pack(fill="x", pady=5)
        Button(self.left_frame, text="Геометрические", command=self.show_geometric_controls).pack(fill="x", pady=5)
        Frame(self.left_frame, height=4, bd=1, relief="sunken", bg="gray").pack(fill="x", pady=5)
        Button(self.left_frame, text="Загрузить", command=self.load_dataset).pack(fill="x", pady=5)
        Button(self.left_frame, text="Сохранить", command=self.save_augmented_images).pack(fill="x", pady=5)

        # Правая панель
        self.control_frame = Frame(self.right_frame)
        self.control_frame.pack(pady=10)
        self.canvas = Canvas(self.right_frame, width=400, height=300)
        self.canvas.pack()

# region  controls

    def clear_controls(self):
        for widget in self.control_frame.winfo_children():
            widget.destroy()

    def show_rotation_controls(self):
        self.clear_controls()
        Label(self.control_frame, text="Настройка поворота").pack()
        Scale(self.control_frame, from_=0, to=90, orient="horizontal",
              label="Rotation (degrees)", variable=self.rotation_var).pack()
        Button(self.control_frame, text="Применить", command=self.apply_augmentation).pack()
        Button(self.control_frame, text="Сбросить", command=self.reset_image).pack()

    def show_flip_controls(self):
        self.clear_controls()
        Label(self.control_frame, text="Настройки отражения").pack()
        Checkbutton(self.control_frame, text="Горизонтальное отражение", variable=self.flip_var).pack()
        Checkbutton(self.control_frame, text="Вертикальное отражение", variable=self.flip_v_var).pack()
        Button(self.control_frame, text="Применить", command=self.apply_augmentation).pack()
        Button(self.control_frame, text="Сбросить", command=self.reset_image).pack()

    def show_noise_controls(self):
        "Показывает настройки для добавления шума."
        self.clear_controls()
        Label(self.control_frame, text="Настройки шума").pack()
        Scale(self.control_frame, from_=0, to=50, orient="horizontal", label="уровень шума", length=200, variable=self.rotation_var).pack()
        Button(self.control_frame, text="Применить", command=self.apply_noise).pack()
        Button(self.control_frame, text="Сбросить", command=self.reset_image).pack()

    def show_denoise_controls(self):
        "Показывает настройки для удаления шума."
        self.clear_controls()
        Label(self.control_frame, text="Удаление шума").pack()
        Button(self.control_frame, text="Усреднение", command=self.apply_mean_filter).pack()
        Button(self.control_frame, text="Фильтр Гауса", command=self.apply_gaussian_filter).pack()
        Button(self.control_frame, text="Сбросить", command=self.reset_image).pack()

    def show_histogram_controls(self):
        "Показывает настройки для преобразований гистограммы."
        self.clear_controls()
        Label(self.control_frame, text="Гистограммы").pack()
        Button(self.control_frame, text="Эквализация", command=self.apply_histogram_equalization).pack()
        Scale(self.control_frame, from_=0.5, to=2.0, resolution=0.05, orient="horizontal", length=200, label="Статическая цветокорекция", variable=self.rotation_var).pack()
        Button(self.control_frame, text="Применить", command=self.apply_color_correction).pack()
        Button(self.control_frame, text="Сбросить", command=self.reset_image).pack()

    def show_geometric_controls(self):
        "Показывает настройки для геометрических фильтров."
        self.clear_controls()
        Label(self.control_frame, text="Геометрические фильтры").pack()
        Button(self.control_frame, text="Перенос", command=self.apply_translation).pack()
        Button(self.control_frame, text="Поворот волна 1", command=self.apply_wave1).pack()
        Button(self.control_frame, text="Поворот волна 2", command=self.apply_wave2).pack()
        Button(self.control_frame, text="Эфект стекла", command=self.apply_glass_effect).pack()
        Scale(self.control_frame, from_=1, to=30, orient="horizontal", label="Сила размытия в движении", length=200, variable=self.rotation_var).pack()
        Button(self.control_frame, text="Применить", command=self.apply_motion_blur).pack()
        Button(self.control_frame, text="Сбросить", command=self.reset_image).pack()
# endregion

# region  funcs
    def apply_noise(self):
        "Применяет шум."
        self.pipeline.add_noise(self.rotation_var.get())
        self.get_preview_image()

    def apply_mean_filter(self):
        "Применяет усреднение для удаления шума."
        self.pipeline.add_noise(self.rotation_var.get())
        self.pipeline.remove_noise_mean()
        self.get_preview_image()

    def apply_gaussian_filter(self):
        "Применяет фильтр Гаусса для удаления шума."
        self.pipeline.remove_noise_gaussian()
        self.get_preview_image()

    def apply_histogram_equalization(self):
        "Применяет эквализацию гистограммы."
        self.pipeline.histogram_equalization()
        self.get_preview_image()

    def apply_color_correction(self):
        "Применяет статистическую цветокоррекцию."
        self.pipeline.statistical_color_correction(strength=self.rotation_var.get())
        self.get_preview_image()

    def apply_glass_effect(self):
        "Применяет эффект стекла."
        self.pipeline.add_glass_effect()
        self.get_preview_image()

    def apply_translation(self):
        "Применяет перенос."
        self.pipeline.add_translation()
        self.get_preview_image()

    def apply_wave1(self):
        "Применяет поворот волна 1."
        self.pipeline.add_rotation_wave1()
        self.get_preview_image()

    def apply_wave2(self):
        "Применяет поворот волна 2."
        self.pipeline.add_rotation_wave2()
        self.get_preview_image()

    def apply_motion_blur(self):
        "Применяет motion blur."
        self.pipeline.add_motion_blur(kernel_size=self.rotation_var.get())
        self.get_preview_image()
# endregion

    def load_dataset(self):
        directory = filedialog.askdirectory(title="Select Dataset Directory")
        if directory:
            loader = ImageLoader(directory)
            self.images = loader.load_images()
            self.show_temporary_message(f"Loaded {len(self.images)} images")

    def apply_augmentation(self):
        self.pipeline = AugmentationPipeline()
        if self.rotation_var.get():
            self.pipeline.add_rotation(self.rotation_var.get())
        if self.flip_var.get() or self.flip_v_var.get():
            self.pipeline.add_flip(self.flip_var.get(), self.flip_v_var.get())
        self.get_preview_image()

    def reset_image(self):
        self.pipeline = AugmentationPipeline()
        self.get_preview_image()

    def save_augmented_images(self):
        directory = filedialog.askdirectory(title="Select Save Directory")
        if directory and self.augmented_images:
            saver = ImageSaver(directory)
            self.augmented_images = self.pipeline.apply(self.images)
            saver.save_images(self.augmented_images)
            self.show_temporary_message(f"Saved {len(self.augmented_images)} images!")

    def show_image(self, image):
        image.thumbnail((400, 300))
        photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(200, 150, image=photo)
        self.root.image = photo

    def get_preview_image(self):
        self.augmented_images = self.pipeline.apply([self.images[0]])
        self.show_image(self.augmented_images[0])

    def show_temporary_message(self, text):
        message_label = Label(self.right_frame, text=text, fg="green")
        message_label.pack()
        self.root.after(5000, message_label.destroy)


if __name__ == "__main__":
    root = Tk()
    app = ImageAugmentorApp(root)
    root.mainloop()
