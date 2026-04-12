import sqlite3
import cv2
import os
from datetime import datetime

class DatabaseLogger:
    def __init__(self, db_path="data/loomvision.db", output_dir="output/defects"):
        self.db_path = db_path
        self.output_dir = output_dir
        
        # Ensure output directory exists before saving any images
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self._initialize_db()

    def _initialize_db(self):
        """Creates the SQLite database and the defects table if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS defects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                defect_type TEXT NOT NULL,
                image_path TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def log_defect(self, defect_type, annotated_frame):
        """
        1. Saves the defect image to the disk.
        2. Logs the defect event (with path) into the database.
        """
        # Generate unique filename using current time
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename_safe_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"defect_{filename_safe_time}.jpg"
        
        # Determine full path to save the image to
        # Since we might run from the root LoomVisionAI dir, relative path works
        image_path = os.path.join(self.output_dir, filename)
        
        # Save image to file system
        cv2.imwrite(image_path, annotated_frame)
        print(f"[Alert] {defect_type} logged at {timestamp_str}. Image saved to {image_path}")
        
        # Insert record into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO defects (timestamp, defect_type, image_path)
            VALUES (?, ?, ?)
        ''', (timestamp_str, defect_type, image_path))
        
        conn.commit()
        conn.close()
