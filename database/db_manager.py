import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect("experiments.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY,
                    date TEXT,
                    image_path TEXT,
                    generation_params TEXT,
                    result_classical INTEGER,
                    result_ml INTEGER,
                    result_cnn INTEGER
                )''')
    conn.commit()
    conn.close()

def save_experiment(image_path, params, res_classical, res_ml, res_cnn):
    conn = sqlite3.connect("experiments.db")
    c = conn.cursor()
    c.execute("INSERT INTO experiments (date, image_path, generation_params, result_classical, result_ml, result_cnn) VALUES (?, ?, ?, ?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path, str(params), res_classical, res_ml, res_cnn))
    conn.commit()
    conn.close()

def fetch_all_experiments():
    conn = sqlite3.connect("experiments.db")
    c = conn.cursor()
    c.execute("SELECT * FROM experiments")
    rows = c.fetchall()
    conn.close()
    return rows