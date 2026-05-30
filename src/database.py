import sqlite3
import cv2
import os
from datetime import datetime
import threading
import time
from src.notifications import send_whatsapp_alert

class DatabaseLogger:
    def __init__(self, db_path="data/loomvision.db", output_dir="output/defects"):
        self.db_path = db_path
        self.output_dir = output_dir
        
        # Ensure output directory exists before saving any images
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self._initialize_db()
        
        # Rate limiting state
        self.last_alert_time = 0.0
        self.alert_cooldown = 10.0  # seconds between alerts

    def _initialize_db(self):
        """Creates the SQLite database and the defects table if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table with new schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS defects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                defect_type TEXT NOT NULL,
                image_path TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                anomaly_score REAL DEFAULT 0.0,
                engine_used TEXT DEFAULT 'unknown',
                session_id TEXT DEFAULT 'default'
            )
        ''')
        
        # Try to add new columns if upgrading from old schema
        try:
            cursor.execute("ALTER TABLE defects ADD COLUMN confidence REAL DEFAULT 0.0")
            cursor.execute("ALTER TABLE defects ADD COLUMN anomaly_score REAL DEFAULT 0.0")
            cursor.execute("ALTER TABLE defects ADD COLUMN engine_used TEXT DEFAULT 'unknown'")
            cursor.execute("ALTER TABLE defects ADD COLUMN session_id TEXT DEFAULT 'default'")
        except sqlite3.OperationalError:
            pass # Columns already exist
            
        conn.commit()
        conn.close()

    def log_defect(self, defect_type, annotated_frame, confidence=0.0, anomaly_score=0.0, engine_used="unknown", session_id="default"):
        """
        1. Saves the defect image to the disk.
        2. Logs the defect event (with path and metrics) into the database.
        """
        # Generate unique filename using current time
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename_safe_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"defect_{filename_safe_time}.jpg"
        
        # Determine full path to save the image to
        # Since we might run from the root LoomVisionAI dir, relative path works
        image_path = os.path.join(self.output_dir, filename)
        
        cv2.imwrite(image_path, annotated_frame)
        print(f"[Alert] {defect_type} logged at {timestamp_str}. Image saved to {image_path}")
        
        # Trigger WhatsApp alert asynchronously with rate limiting
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            self.last_alert_time = current_time
            threading.Thread(target=send_whatsapp_alert, args=(defect_type, timestamp_str), daemon=True).start()

        
        # Insert record into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO defects (timestamp, defect_type, image_path, confidence, anomaly_score, engine_used, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp_str, defect_type, image_path, confidence, anomaly_score, engine_used, session_id))
        
        conn.commit()
        conn.close()

    def get_recent_defects(self, limit=10):
        """Fetches the most recent defects from the database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM defects 
            ORDER BY id DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

    def clear_all_defects(self):
        """Clears all defect logs from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM defects')
        conn.commit()
        conn.close()
