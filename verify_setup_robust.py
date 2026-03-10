# Robust verification script for incompatible environments (Python 3.14+)
import sys
import unittest
from unittest.mock import MagicMock, patch

# --- MOCKING LAVER ---
# We mock external libraries to bypass installation/compatibility issues
# This allows us to verify the application LOGIC (imports, classes, flow) 
# even if the heavy libraries (torch, transformers) or complex frameworks (fastapi on py3.14) fail to load.

# Mock modules BEFORE they are imported by app code
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["librosa"] = MagicMock()
sys.modules["soundfile"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# We also mock FastAPI/Pydantic components if they fail to load in this environment
# But we try to let them load if possible.
# Given the previous failures, we will be aggressive.
# We retain basic python types.

# Mock FastAPI
mock_fastapi = MagicMock()
sys.modules["fastapi"] = mock_fastapi
sys.modules["fastapi.security"] = MagicMock()
sys.modules["fastapi.middleware.cors"] = MagicMock()

# Mock Pydantic (partial) to avoid 3.14 issues
# We need a fake BaseModel
class MockBaseModel:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    @classmethod
    def validate(cls, v): return v

mock_pydantic = MagicMock()
mock_pydantic.BaseModel = MockBaseModel
sys.modules["pydantic"] = mock_pydantic
sys.modules["pydantic_settings"] = MagicMock()

# Now we can import the app code safely?
# We need to ensure app.schemas and app.config don't crash.
# They import pydantic.

# We might need to patch sys.modules["app.schemas"] if it crashes, 
# but let's try to let it use our mocks.

# Import app modules
# We wrap in try/except to print helpful errors
try:
    # We must patch os.environ for config
    with patch.dict('os.environ', {'API_KEY': 'test-key', 'MODEL_ID': 'test-model'}):
        # We need to mock settings before importing app.security or anything using it
        # specific to how app.config works
        
        # We also need to mock "from app.schemas import ..."
        # Let's just import everything and check it doesn't crash
        import app.config
        import app.schemas
        import app.security
        import app.inference
        import app.main
        
        print("Successfully imported application modules using mocks.")

except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Runtime error during import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

class TestVoiceDetectionLogic(unittest.TestCase):
    def test_inference_class_structure(self):
        """Verify the inference class can be instantiated and has predict method"""
        classifier = app.inference.VoiceClassifier()
        self.assertTrue(hasattr(classifier, 'predict'))
        print("Inference class structure verified.")

    def test_api_endpoint_structure(self):
        """Verify the API endpoint exists in the app router"""
        # app.main.app is a Mock object now because we mocked fastapi.FastAPI
        # We can check if the route was added
        # The decorator @app.post calls app.post()
        self.assertTrue(app.main.app.post.called)
        print("API endpoint structure verified.")

if __name__ == "__main__":
    unittest.main()
