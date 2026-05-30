import unittest
import threading
import time
import urllib.request
import json
import sqlite3
import os
from http.server import ThreadingHTTPServer

# Import the APIHandler from the api_server script
from api_server import APIHandler

class TestAPIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Start the API server in a background thread for testing."""
        cls.port = 8085
        cls.server = ThreadingHTTPServer(('localhost', cls.port), APIHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        
        # Give it a moment to start
        time.sleep(1)

        # Setup test database
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
        # Insert a dummy record if table is empty
        cursor.execute("SELECT COUNT(*) FROM defects")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO defects (timestamp, defect_type, image_path) VALUES (?, ?, ?)", 
                           ("2026-01-01 12:00:00", "Test Defect", "output/defects/test.jpg"))
            conn.commit()
        conn.close()

    @classmethod
    def tearDownClass(cls):
        """Shut down the background API server."""
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join()

    def test_metrics_endpoint(self):
        """Test GET /api/v1/metrics returns proper JSON."""
        req = urllib.request.Request(f"http://localhost:{self.port}/api/v1/metrics")
        with urllib.request.urlopen(req) as response:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read().decode())
            self.assertIn("total_scans", data)
            self.assertIn("defects_found", data)
            self.assertIn("accuracy_rate", data)
            self.assertIn("inspection_active", data)

    def test_defects_endpoint(self):
        """Test GET /api/v1/defects returns a list."""
        req = urllib.request.Request(f"http://localhost:{self.port}/api/v1/defects")
        with urllib.request.urlopen(req) as response:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read().decode())
            self.assertIsInstance(data, list)
            if len(data) > 0:
                self.assertIn("id", data[0])
                self.assertIn("defect_type", data[0])

    def test_toggle_inspection_endpoint(self):
        """Test POST /api/v1/toggle_inspection toggles state."""
        req = urllib.request.Request(f"http://localhost:{self.port}/api/v1/toggle_inspection", method="POST")
        with urllib.request.urlopen(req) as response:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read().decode())
            self.assertIn("inspection_active", data)

if __name__ == '__main__':
    unittest.main()
