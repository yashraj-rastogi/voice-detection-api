
import io
import base64
import librosa
import numpy as np
import soundfile as sf
from fastapi import HTTPException
from app.config import settings

def decode_base64_audio(base64_string: str) -> io.BytesIO:
    """
    Decodes a Base64 string into a BytesIO object.
    """
    try:
        # Check if header exists (e.g., "data:audio/mp3;base64,...")
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
            
        audio_data = base64.b64decode(base64_string)
        return io.BytesIO(audio_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 audio: {str(e)}")

def preprocess_audio(audio_file: io.BytesIO):
    """
    Loads audio from bytes, resamples to 16kHz, and prepares it for the model.
    Returns the raw waveform as a numpy array.
    """
    import tempfile
    import os
    
    try:
        # librosa needs a file path, so we save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Load audio with librosa (handles various formats)
            # target_sr=16000 is crucial for Wav2Vec2
            y, sr = librosa.load(tmp_path, sr=settings.SAMPLE_RATE)
            
            # Ensure mono
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
                
            # Normalize audio for consistent predictions
            y = librosa.util.normalize(y)
            
            return y
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio file: {str(e)}")
