import requests
import base64
import json
import time

def test_live_api():
    print("Test Live API Script")
    print("====================")
    
    # 1. Configuration
    API_URL = "http://127.0.0.1:8000"
    API_KEY = "your-secure-api-key-here" # Default key from config.py

    # 2. Check Health
    try:
        print(f"Checking health at {API_URL}/health...")
        r = requests.get(f"{API_URL}/health")
        if r.status_code == 200:
            print("✅ API is healthy.")
            print(f"Response: {r.json()}")
        else:
            print(f"❌ API might be down. Status: {r.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is it running? (Run 'uvicorn app.main:app --reload')")
        return

    # 3. Prepare Payload
    # Minimal valid MP3 header/frame (silent) base64 encoded
    # This is a very short sequence that mimics an MP3 file start
    mp3_base64 = "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//OEAAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAAEAAABIADAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwM=="
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": mp3_base64
    }

    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    # 4. Send Request
    print("\nSending Voice Detection Request...")
    start_time = time.time()
    try:
        response = requests.post(f"{API_URL}/api/voice-detection", json=payload, headers=headers)
        duration = time.time() - start_time
        
        print(f"Request took {duration:.2f} seconds.")
        
        if response.status_code == 200:
            print("✅ Success!")
            data = response.json()
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    test_live_api()
