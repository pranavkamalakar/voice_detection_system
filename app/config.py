import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_KEY: str = "your-secure-api-key-here"
    MODEL_ID: str = "facebook/wav2vec2-base-960h"  # Example default, in prod use a finetuned deepfake detector
    
    # Model configuration
    # For a real deepfake detection task, you would point this to a specific finetuned model
    # e.g., "MelodyMachine/Deepfake-audio-detection" or local path
    
    class Config:
        env_file = ".env"

settings = Settings()
