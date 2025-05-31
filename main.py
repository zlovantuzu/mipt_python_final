from tkinter import Tk, Label, Button, filedialog, Frame, Scale, IntVar, Checkbutton, Canvas
from augmentor.augmentation import ImageLoader, AugmentationPipeline, ImageSaver
from PIL import ImageTk
import sqlite3
from datetime import datetime
from tkinter.ttk import Treeview
from cell_counter.classical_methods.classical_counter import count_cells_classical
from cell_counter.texture_based_ml.ml_counter import count_cells_ml
from cell_counter.cnn_deep_learning.cnn_counter import count_cells_cnn


class ImageAugmentorApp:
    def __init__(self, root):

        self.db_conn = sqlite3.connect("experiments.db")
        self.create_experiments_table()

        self.root = root
        self.root.title("Итоговая работа")

        self.left_frame = Frame(root, width=200, bg="lightgray")
        self.left_frame.pack(side="left", fill="y")
        self.right_frame = Frame(root, width=400)
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.images = []
        self.augmented_images = []
        self.pipeline = AugmentationPipeline()
        self.current_index = 0
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
        Button(self.left_frame, text="Подсчёт клеток", command=self.show_count_controls).pack(fill="x", pady=5)
        Frame(self.left_frame, height=4, bd=1, relief="sunken", bg="gray").pack(fill="x", pady=5)
        Button(self.left_frame, text="Загрузить", command=self.load_dataset).pack(fill="x", pady=5)
        Button(self.left_frame, text="Сохранить", command=self.save_augmented_images).pack(fill="x", pady=5)

        # Правая панель
        self.control_frame = Frame(self.right_frame)
        self.control_frame.pack(pady=10)
        self.canvas = Canvas(self.right_frame, width=400, height=300)
        self.canvas.pack()

        self.nav_frame = Frame(self.right_frame)
        self.nav_frame.pack(side="bottom", fill="x", pady=10)

        Button(self.nav_frame, text="Прошлая", command=self.show_previous_image).pack(side="left", padx=10)
        Button(self.nav_frame, text="Следующая", command=self.show_next_image).pack(side="right", padx=10)

    # region  controls

    def clear_controls(self):
        for widget in self.control_frame.winfo_children():
            widget.destroy()

    def show_rotation_controls(self):
        self.clear_controls()
        Label(self.control_frame, text="Настройка поворота").pack()
        Scale(self.control_frame, from_=0, to=90, orient="horizontal", label="Поворот (градус)", variable=self.rotation_var).pack()
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

    def show_count_controls(self):
        self.clear_controls()
        Label(self.control_frame, text="Подсчёт клеток").pack()
        Button(self.control_frame, text="Метод 1 (Classic)", command=self.apply_classical_count).pack()
        Button(self.control_frame, text="Метод 2 (ML)", command=self.apply_ml_count).pack()
        Button(self.control_frame, text="Метод 3 (CNN)", command=self.apply_cnn_count).pack()
        Button(self.control_frame, text="Все методы", command=self.apply_all_count).pack()
        Button(self.control_frame, text="Показать таблицу", command=self.show_results_table).pack()

    # endregion

# region  funcs
    def apply_noise(self):
        "Применяет шум."
        self.pipeline.add_noise(self.rotation_var.get())
        self.show_preview_image()

    def apply_mean_filter(self):
        "Применяет усреднение для удаления шума."
        self.pipeline.add_noise(self.rotation_var.get())
        self.pipeline.remove_noise_mean()
        self.show_preview_image()

    def apply_gaussian_filter(self):
        "Применяет фильтр Гаусса для удаления шума."
        self.pipeline.remove_noise_gaussian()
        self.show_preview_image()

    def apply_histogram_equalization(self):
        "Применяет эквализацию гистограммы."
        self.pipeline.histogram_equalization()
        self.show_preview_image()

    def apply_color_correction(self):
        "Применяет статистическую цветокоррекцию."
        self.pipeline.statistical_color_correction(strength=self.rotation_var.get())
        self.show_preview_image()

    def apply_glass_effect(self):
        "Применяет эффект стекла."
        self.pipeline.add_glass_effect()
        self.show_preview_image()

    def apply_translation(self):
        "Применяет перенос."
        self.pipeline.add_translation()
        self.show_preview_image()

    def apply_wave1(self):
        "Применяет поворот волна 1."
        self.pipeline.add_rotation_wave1()
        self.show_preview_image()

    def apply_wave2(self):
        "Применяет поворот волна 2."
        self.pipeline.add_rotation_wave2()
        self.show_preview_image()

    def apply_motion_blur(self):
        "Применяет motion blur."
        self.pipeline.add_motion_blur(kernel_size=self.rotation_var.get())
        self.show_preview_image()

    def apply_classical_count(self):
        "Применяет классический метод подсчёта клеток."
        augmented_images = self.pipeline.apply([self.images[self.current_index]])
        result = count_cells_classical(augmented_images[0])
        self.save_result(method1=result)
        self.show_temporary_message(f"Классический метод: найдено {result} клеток")

    def apply_ml_count(self):
        "Применяет ML-метод подсчёта клеток."
        augmented_images = self.pipeline.apply([self.images[self.current_index]])
        result = count_cells_ml(augmented_images[0])
        self.save_result(method2=result)
        self.show_temporary_message(f"ML метод: найдено {result} клеток")

    def apply_cnn_count(self):
        "Применяет CNN-метод подсчёта клеток."
        augmented_images = self.pipeline.apply([self.images[self.current_index]])
        result = count_cells_cnn(augmented_images[0])
        self.save_result(method3=result)
        self.show_temporary_message(f"CNN метод: найдено {result} клеток")

    def apply_all_count(self):
        "Применяет CNN-метод подсчёта клеток."
        augmented_images = self.pipeline.apply([self.images[self.current_index]])
        result1 = count_cells_classical(augmented_images[0])
        result2 = count_cells_ml(augmented_images[0])
        result3 = count_cells_cnn(augmented_images[0])
        self.save_result(method1=result1, method2=result2, method3=result3)
        self.show_temporary_message(f"Классический метод: найдено {result1} клеток, ML метод: найдено {result2} клеток, CNN метод: найдено {result3} клеток")

    # endregion

    def create_experiments_table(self):
        cursor = self.db_conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            file_path TEXT,
            gen_params TEXT,
            method1_result INTEGER,
            method2_result INTEGER,
            method3_result INTEGER
        )''')
        self.db_conn.commit()

    def save_result(self, method1=None, method2=None, method3=None):
        cursor = self.db_conn.cursor()
        file_path = getattr(self.images[self.current_index], "filename", "unknown")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
                       INSERT INTO experiments (date, file_path, gen_params, method1_result, method2_result, method3_result)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ''', (now, file_path, '', method1, method2, method3))
        self.db_conn.commit()
        self.show_temporary_message("Результат сохранён.")

    def show_results_table(self):
        top = Tk()
        top.title("Результаты экспериментов")

        tree = Treeview(top, columns=("ID", "Date", "File", "M1", "M2", "M3"), show="headings")
        tree.heading("ID", text="ID")
        tree.heading("Date", text="Дата")
        tree.heading("File", text="Файл")
        tree.heading("M1", text="Метод 1")
        tree.heading("M2", text="Метод 2")
        tree.heading("M3", text="Метод 3")
        tree.pack(fill="both", expand=True)

        cursor = self.db_conn.cursor()
        cursor.execute("SELECT id, date, file_path, method1_result, method2_result, method3_result FROM experiments")
        for row in cursor.fetchall():
            tree.insert("", "end", values=row)

    def load_dataset(self):
        directory = filedialog.askdirectory(title="Выбрать директорию загрузки")
        if directory:
            loader = ImageLoader(directory)
            self.images = loader.load_images()
            self.show_temporary_message(f"Загружено {len(self.images)} картинок")

    def apply_augmentation(self):
        self.pipeline = AugmentationPipeline()
        if self.rotation_var.get():
            self.pipeline.add_rotation(self.rotation_var.get())
        if self.flip_var.get() or self.flip_v_var.get():
            self.pipeline.add_flip(self.flip_var.get(), self.flip_v_var.get())
        self.show_preview_image()

    def reset_image(self):
        self.pipeline = AugmentationPipeline()
        self.show_preview_image()

    def save_augmented_images(self):
        directory = filedialog.askdirectory(title="Выбрать директорию сохранения")
        if directory and self.augmented_images:
            saver = ImageSaver(directory)
            self.augmented_images = self.pipeline.apply(self.images)
            saver.save_images(self.augmented_images)
            self.show_temporary_message(f"Сохранено {len(self.augmented_images)} картинок")

    def show_image(self, image):
        image.thumbnail((400, 300))
        photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(200, 150, image=photo)
        self.root.image = photo

    def show_preview_image(self):
        self.augmented_images = self.pipeline.apply([self.images[self.current_index]])
        self.show_image(self.augmented_images[0])

    def show_temporary_message(self, text):
        message_label = Label(self.right_frame, text=text, fg="green")
        message_label.pack()
        self.root.after(5000, message_label.destroy)

    def show_next_image(self):
        "Отображает следующее изображение."
        if self.images and self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.show_image(self.images[self.current_index])
            self.show_preview_image()

    def show_previous_image(self):
        "Отображает предыдущее изображение."
        if self.images and self.current_index > 0:
            self.current_index -= 1
            self.show_image(self.images[self.current_index])
            self.show_preview_image()


if __name__ == "__main__":
    root = Tk()
    app = ImageAugmentorApp(root)
    root.mainloop()
