from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import unittest

# --- Mock imports before app load to handle missing heavy dependencies in CI/Test env ---
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["librosa"] = MagicMock()
sys.modules["soundfile"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# Setup mocks for app code logic
with patch.dict('os.environ', {'API_KEY': 'test-key', 'MODEL_ID': 'test-model'}):
    from app.main import app
    from app.inference import classifier

client = TestClient(app)

class TestVoiceDetectionAPI(unittest.TestCase):
    def setUp(self):
        # Override dependency or mock internal logic
        # We mock the predict function to avoid loading the model
        self.original_predict = classifier.predict
        classifier.predict = MagicMock(return_value=("HUMAN", 0.98, "Test explanation"))

    def tearDown(self):
        classifier.predict = self.original_predict

    def test_health_check(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
        self.assertEqual(response.json()["status"], "ok")

    def test_detect_voice_success(self):
        # Mock valid MP3 base64 (content doesn't matter as we mock predict)
        payload = {
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//OEAAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAAEAAABIADAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwM=="
        }
        headers = {"x-api-key": "test-key"}
        
        response = client.post("/api/voice-detection", json=payload, headers=headers)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["classification"], "HUMAN")
        self.assertEqual(data["language"], "English")
        self.assertIsInstance(data["confidenceScore"], float)

    def test_detect_voice_unauthorized(self):
        payload = {
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": "dummy"
        }
        # No header
        response = client.post("/api/voice-detection", json=payload)
        self.assertEqual(response.status_code, 403) # FastAPI Depends(get_api_key) raises 403 if invalid/missing usually, or 401. Let's act based on security.py which likely uses HTTPException(status_code=403)

    def test_detect_voice_invalid_payload(self):
        # Missing audioBase64
        payload = {
            "language": "English",
            "audioFormat": "mp3"
        }
        headers = {"x-api-key": "test-key"}
        response = client.post("/api/voice-detection", json=payload, headers=headers)
        self.assertEqual(response.status_code, 422) # Validation error

if __name__ == "__main__":
    unittest.main()
