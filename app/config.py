
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    MODEL_NAME = os.getenv("MODEL_NAME", "mo-thecreator/Deepfake-audio-detection")
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
    ALLOWED_EXTENSIONS = {"mp3", "wav", "ogg", "flac"}
    
    # API Security - Load from environment variable
    API_KEY: Optional[str] = os.getenv("API_KEY", None)
    
    # If API_KEY is not set, generate a warning
    @classmethod
    def validate(cls):
        if not cls.API_KEY:
            print("⚠️  WARNING: API_KEY not set! API is running without authentication.")
            print("   Set API_KEY environment variable for production use.")
        else:
            print("✅ API Key authentication enabled")
    
settings = Config()
settings.validate()
