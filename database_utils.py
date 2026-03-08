import sqlite3
from datetime import datetime
import hashlib

DB_NAME = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            disease TEXT,
            status TEXT,
            confidence REAL,
            timestamp TEXT,
            hash TEXT,
            block_index INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def generate_patient_id():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()
    return f"P{count+1:03d}"

def get_next_block_index():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(block_index) FROM predictions")
    last_block = cursor.fetchone()[0]
    conn.close()
    if last_block is None:
        return 1
    return last_block + 1

def save_prediction(disease, confidence):
    patient_id = generate_patient_id()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "Detected" if confidence >= 0.5 else "Not Detected"
    block_index = get_next_block_index()

    record_string = f"{patient_id}-{disease}-{status}-{confidence}-{timestamp}-{block_index}"
    result_hash = hashlib.sha256(record_string.encode()).hexdigest()

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO predictions 
        (patient_id, disease, status, confidence, timestamp, hash, block_index)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (patient_id, disease, status, confidence, timestamp, result_hash, block_index))

    conn.commit()
    conn.close()

    return result_hash
