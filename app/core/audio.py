
import io
import base64
import librosa
import numpy as np
import soundfile as sf
from fastapi import HTTPException
from app.config import settings
import logging
import tempfile
import os

logger = logging.getLogger(__name__)


def decode_base64_audio(base64_string: str) -> io.BytesIO:
    """Decodes a Base64 string into a BytesIO object."""
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        audio_data = base64.b64decode(base64_string)
        return io.BytesIO(audio_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 audio: {str(e)}")


def compute_audio_profile(y: np.ndarray, sr: int) -> dict:
    """
    Compute a technical profile of the audio sample.
    Returns metadata useful for quality assessment and forensic analysis.
    """
    duration = len(y) / sr

    # RMS energy
    rms = float(np.sqrt(np.mean(y ** 2)))

    # SNR estimate (signal RMS vs noise floor estimate)
    snr_db = 0.0
    if rms > 1e-6:
        # Estimate noise floor from the quietest 10% of frames
        frame_len = int(0.025 * sr)
        hop_len = int(0.010 * sr)
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        noise_floor = float(np.percentile(rms_frames, 10))
        snr_db = round(20 * np.log10(rms / max(noise_floor, 1e-10)), 1)

    # Clipping detection — samples at or near ±1.0
    clip_threshold = 0.999
    clipping_ratio = float(np.mean(np.abs(y) > clip_threshold))
    clipping_detected = clipping_ratio > 0.001

    # Silence ratio
    silence_threshold = rms * 0.1
    silence_ratio = float(np.mean(np.abs(y) < silence_threshold))

    return {
        "duration_sec": round(duration, 2),
        "snr_db": round(snr_db, 1),
        "clipping_detected": clipping_detected,
        "silence_ratio": round(silence_ratio, 3),
        "rms_energy": round(rms, 4),
        "sample_rate": sr,
    }


def segment_audio(y: np.ndarray, sr: int, segment_sec: float = 5.0,
                  overlap_sec: float = 1.0) -> list:
    """
    Split audio into overlapping segments for per-segment analysis.
    Short audio (< segment_sec) is returned as a single segment.
    """
    segment_len = int(segment_sec * sr)
    hop_len = int((segment_sec - overlap_sec) * sr)

    if len(y) <= segment_len:
        return [y]

    segments = []
    start = 0
    while start < len(y):
        end = min(start + segment_len, len(y))
        seg = y[start:end]
        # Only include if at least 1 second long
        if len(seg) >= sr:
            segments.append(seg)
        start += hop_len

    return segments if segments else [y]


def preprocess_audio(audio_file: io.BytesIO):
    """
    Clean and standardized preprocessing for AI detection.
    Focuses on natural signal preservation to avoid false AI classifications.
    Returns: (audio_array, audio_profile_dict)
    """
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

            # Reject extremely short audio
            if len(y) < sr * 0.3:
                raise HTTPException(
                    status_code=400,
                    detail="Audio too short. Minimum 0.3 seconds required."
                )

            # 1. Basic Silence Trimming (Safer threshold)
            y_trimmed, _ = librosa.effects.trim(y, top_db=40)
            if len(y_trimmed) > sr * 0.1:
                y = y_trimmed

            # 2. Gentle Peak Normalization
            # Preserves natural dynamics which models use for detection
            peak = np.max(np.abs(y))
            if peak > 0:
                y = y / peak

            # 3. Time Clamping — max 30 seconds
            max_duration = 30
            if len(y) > sr * max_duration:
                y = y[:sr * max_duration]

            # 4. Compute audio profile
            profile = compute_audio_profile(y, sr)

            logger.info(
                f"Preprocessing complete: {profile['duration_sec']}s, "
                f"SNR={profile['snr_db']}dB, "
                f"clipping={'YES' if profile['clipping_detected'] else 'NO'}"
            )

            return y, profile

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio file: {str(e)}")
