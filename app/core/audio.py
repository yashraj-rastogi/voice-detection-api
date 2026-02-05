
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
        # Check if header exists (e.g., "data:audio/mp3;base64,...")
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
            
        audio_data = base64.b64decode(base64_string)
        return io.BytesIO(audio_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 audio: {str(e)}")

def reduce_noise(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply noise reduction using spectral gating.
    This helps remove background noise while preserving voice quality.
    """
    try:
        # Compute Short-Time Fourier Transform
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        
        # Estimate noise floor from quieter parts
        noise_threshold = np.percentile(magnitude, 10)
        
        # Create mask to suppress noise
        mask = magnitude > (noise_threshold * 1.5)
        
        # Apply mask
        stft_cleaned = stft * mask
        
        # Reconstruct audio
        y_cleaned = librosa.istft(stft_cleaned)
        
        return y_cleaned
    except Exception as e:
        logger.warning(f"Noise reduction failed, using original audio: {e}")
        return y

def trim_silence(y: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    """
    Remove silence from beginning and end of audio.
    Helps focus on actual voice content.
    """
    try:
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        
        # Ensure we don't trim too much (keep at least 0.5 seconds)
        if len(y_trimmed) < sr * 0.5:
            return y
            
        return y_trimmed
    except Exception as e:
        logger.warning(f"Silence trimming failed, using original audio: {e}")
        return y

def enhance_audio_quality(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply audio enhancements to improve signal quality.
    """
    try:
        # Apply pre-emphasis filter to enhance high frequencies
        # This helps emphasize voice characteristics
        pre_emphasis = 0.97
        y_emphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        
        return y_emphasized
    except Exception as e:
        logger.warning(f"Audio enhancement failed, using original audio: {e}")
        return y

def preprocess_audio(audio_file: io.BytesIO):
    """
    Enhanced audio preprocessing with noise reduction and quality improvements.
    Returns the raw waveform as a numpy array optimized for deepfake detection.
    """
    import tempfile
    import os
    
    try:
        # Save to temporary file for librosa
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
            
            # Check if audio is valid (not empty or too short)
            if len(y) < sr * 0.1:  # Less than 0.1 seconds
                raise ValueError("Audio too short for analysis")
            
            # Phase 1 Enhancements:
            
            # 1. Trim silence from edges (focus on voice content)
            y = trim_silence(y, sr, top_db=30)
            
            # 2. Apply noise reduction
            y = reduce_noise(y, sr)
            
            # 3. Enhance audio quality
            y = enhance_audio_quality(y, sr)
            
            # 4. Advanced normalization
            # RMS normalization for consistent loudness
            rms = np.sqrt(np.mean(y**2))
            if rms > 0:
                target_rms = 0.1
                y = y * (target_rms / rms)
            
            # Peak normalization to prevent clipping
            y = librosa.util.normalize(y)
            
            # Ensure audio length is reasonable (clamp very long audio)
            max_duration = 30  # seconds
            if len(y) > sr * max_duration:
                logger.info(f"Audio too long ({len(y)/sr:.1f}s), trimming to {max_duration}s")
                y = y[:sr * max_duration]
            
            logger.info(f"Preprocessed audio: duration={len(y)/sr:.2f}s, shape={y.shape}")
            
            return y
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio file: {str(e)}")
