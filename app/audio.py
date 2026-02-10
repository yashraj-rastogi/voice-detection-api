
import io
import base64
import librosa
import numpy as np
import soundfile as sf
from fastapi import HTTPException
from app.config import settings
import logging

logger = logging.getLogger(__name__)

def decode_base64_audio(base64_string: str) -> io.BytesIO:
    """
    Decodes a Base64 string into a BytesIO object.
    """
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
            
        audio_data = base64.b64decode(base64_string)
        return io.BytesIO(audio_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 audio: {str(e)}")

def preprocess_audio(audio_file: io.BytesIO):
    """
    Clean and standardized preprocessing for AI detection.
    Focuses on natural signal preservation to avoid false AI classifications.
    """
    import tempfile
    import os
    
    try:
        # Save to temporary file for librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Load audio at 16kHz (Standard for Wav2Vec2)
            y, sr = librosa.load(tmp_path, sr=settings.SAMPLE_RATE)
            
            # Ensure mono
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            
            # 1. Basic Silence Trimming (Safer threshold)
            y_trimmed, _ = librosa.effects.trim(y, top_db=40)
            if len(y_trimmed) > sr * 0.1: # Only use if not too much was cut
                y = y_trimmed

            # 2. Gentle Normalization
            # Instead of target RMS, we use standard peak normalization
            # This preserves the natural dynamics which models use for detection
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            # 3. Time Clamping
            max_duration = 30
            if len(y) > sr * max_duration:
                y = y[:sr * max_duration]
            
            logger.info(f"Natural preprocessing complete: {len(y)/sr:.2f}s")
            return y
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio file: {str(e)}")
