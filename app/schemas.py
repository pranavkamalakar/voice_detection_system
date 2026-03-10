from enum import Enum
from pydantic import BaseModel, Field, validator
import base64

class Language(str, Enum):
    TAMIL = "Tamil"
    ENGLISH = "English"
    HINDI = "Hindi"
    MALAYALAM = "Malayalam"
    TELUGU = "Telugu"

class ClassificationResult(str, Enum):
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"

class VoiceDetectionRequest(BaseModel):
    language: Language
    audioFormat: str = Field(..., pattern="^mp3$")
    audioBase64: str

    @validator('audioBase64')
    def validate_base64(cls, v):
        try:
            # Check if it's strictly base64
            # We loosely validate here, deep validation happens during decode
            if not v:
                raise ValueError("Empty audio data")
            return v
        except Exception:
            raise ValueError("Invalid Base64 string")

class VoiceDetectionSuccess(BaseModel):
    status: str = "success"
    language: str
    classification: ClassificationResult
    confidenceScore: float
    explanation: str

class VoiceDetectionError(BaseModel):
    status: str = "error"
    message: str
