# AI Voice Detection System

A production-ready REST API for classifying MP3 audio as AI_GENERATED or HUMAN.

## Features
- **Language Support**: Tamil, English, Hindi, Malayalam, Telugu.
- **Security**: API Key authentication.
- **Architecture**: Modular FastAPI backend with Hugging Face Transformers integration.
- **Containerization**: Full Docker support.

## Setup

### 1. Environment Variables
Create a `.env` file in the root directory:
```bash
API_KEY=your-secret-key-123
MODEL_ID=facebook/wav2vec2-base-960h  # Or your fine-tuned deepfake detection model path
```

### 2. Local Installation
```bash
# Install dependencies (requires ffmpeg installed on system)
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload
```

### 3. Docker Deployment
```bash
# Build image
docker build -t voice-detection-api .

# Run container
docker run -p 8000:8000 --env-file .env voice-detection-api
```

## API Usage

**Endpoint**: `POST /api/voice-detection`
**Headers**: `x-api-key: your-secret-key-123`

**Request Body**:
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64_encoded_mp3_string>"
}
```

**Success Response**:
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.98,
  "explanation": "Detected high probability (0.98) of synthetic features."
}
```

## Technical Notes
- **Model**: The system is designed to load a `transformers` model. By default, it attempts to load a generic model. For true accumulation, ensure `MODEL_ID` points to a model fine-tuned on the ASVspoof dataset.
- **Audio Processing**: All audio is downsampled to 16kHz for consistency with standard speech models.
