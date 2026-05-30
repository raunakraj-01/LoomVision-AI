import unittest
import json
import sqlite3
import os

from flask_server import app, init_system

class TestFlaskAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the Flask test client and ensure the database exists."""
        # Initialize the global variables in flask_server
        init_system()
        
        # Enable testing mode in Flask
        app.config['TESTING'] = True
        cls.client = app.test_client()

        # Setup test database to ensure the API doesn't fail
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect("data/loomvision.db")
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS defects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                defect_type TEXT NOT NULL,
                image_path TEXT NOT NULL
            )
        ''')
        
        # Insert a dummy record if the table is empty
        cursor.execute("SELECT COUNT(*) FROM defects")
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
                INSERT INTO defects (timestamp, defect_type, image_path)
                VALUES (?, ?, ?)
            ''', ("2024-01-01 12:00:00", "Normal", "test.jpg"))
            conn.commit()
        conn.close()

    def test_metrics_endpoint(self):
        """Test the /api/v1/metrics endpoint."""
        response = self.client.get('/api/v1/metrics')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn("total_scans", data)
        self.assertIn("defects_found", data)
        self.assertIn("uptime", data)
        self.assertIn("accuracy_rate", data)
        self.assertIn("inspection_active", data)

    def test_defects_endpoint(self):
        """Test the /api/v1/defects endpoint."""
        response = self.client.get('/api/v1/defects')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        # Should be a list
        self.assertIsInstance(data, list)
        
        # Since we inserted a dummy record, it should have at least 1 item
        self.assertGreaterEqual(len(data), 1)
        
        # Check structure of the first item
        first_item = data[0]
        self.assertIn("id", first_item)
        self.assertIn("timestamp", first_item)
        self.assertIn("defect_type", first_item)
        self.assertIn("image_path", first_item)

if __name__ == "__main__":
    unittest.main()
