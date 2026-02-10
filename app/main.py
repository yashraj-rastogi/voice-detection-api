

from fastapi import FastAPI, HTTPException, Header, Depends, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any, List
from app.core.audio import decode_base64_audio, preprocess_audio
from app.core.model import voice_detector
from app.core.explanation import generate_explanation
from app.config import settings
import uvicorn
import logging
import os

# Setup Logger
logger = logging.getLogger("api")

app = FastAPI(
    title="VoiceGuard — AI Voice Detection API",
    description=(
        "World-class deepfake audio detection API. "
        "Combines neural model inference with multi-analyzer forensic analysis "
        "to detect AI-generated speech with maximum accuracy."
    ),
    version="3.0.0"
)

# CORS Security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files (for the Web UI)
static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Custom Error Handlers (Hackathon Specification)
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )


# ===============================================================
#  Request / Response Models
# ===============================================================

class AudioRequest(BaseModel):
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(
        ..., description="Language of the audio"
    )
    audioFormat: Literal["mp3", "wav"] = Field(
        ..., description="Audio format (mp3 or wav)"
    )
    audioBase64: str = Field(
        ..., description="Base64-encoded audio file"
    )


class ForensicDetail(BaseModel):
    score: float = Field(..., description="Analyzer score (0=human, 1=AI)")
    verdict: str = Field(..., description="Analyzer verdict")
    artifacts_found: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class NeuralModelDetail(BaseModel):
    score: float = Field(..., description="Neural model AI probability")
    verdict: str = Field(..., description="Neural model verdict")
    segments_analyzed: int = Field(..., description="Number of audio segments analyzed")
    per_segment_scores: List[float] = Field(default_factory=list)


class AudioProfileResponse(BaseModel):
    duration_sec: float
    snr_db: float
    clipping_detected: bool
    silence_ratio: float
    rms_energy: float
    sample_rate: int


class DetectionResponse(BaseModel):
    status: Literal["success", "error"] = Field(default="success")
    language: str = Field(..., description="Language of the audio")
    classification: Literal["AI_GENERATED", "HUMAN"] = Field(
        ..., description="Voice classification result"
    )
    confidenceScore: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score (0.0–1.0)"
    )
    explanation: str = Field(
        ..., description="Evidence-based explanation for the decision"
    )
    # Extended fields
    inferenceTimeMs: Optional[float] = Field(
        None, description="Total inference time in milliseconds"
    )
    analyzersAgree: Optional[bool] = Field(
        None, description="Whether neural model and forensic analyzers agree"
    )
    forensics: Optional[Dict[str, Any]] = Field(
        None, description="Detailed forensic analysis (when detailed=true)"
    )
    audioProfile: Optional[AudioProfileResponse] = Field(
        None, description="Audio quality profile (when detailed=true)"
    )
    artifactsSummary: Optional[List[str]] = Field(
        None, description="List of all detected artifacts (when detailed=true)"
    )


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str


# ===============================================================
#  Authentication
# ===============================================================

def verify_api_key(
    x_api_key_lower: str = Header(None, alias="x-api-key"),
    x_api_key_upper: str = Header(None, alias="X-API-Key")
):
    x_api_key = x_api_key_lower or x_api_key_upper
    if not settings.API_KEY:
        return True
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required. Please provide x-api-key header."
        )
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


# ===============================================================
#  Routes
# ===============================================================

@app.get("/")
def read_root():
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "status": "healthy",
        "message": "VoiceGuard AI Voice Detection API v3.0",
        "authentication": "enabled" if settings.API_KEY else "disabled"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "engine": "multi-analyzer forensic detection",
        "model": settings.MODEL_NAME,
        "analyzers": ["neural_model", "spectral_analysis", "temporal_analysis",
                       "formant_analysis", "artifact_detection"],
        "authentication": "enabled" if settings.API_KEY else "disabled"
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_voice(
    request: AudioRequest,
    authenticated: bool = Depends(verify_api_key),
    detailed: bool = Query(False, description="Return full forensic breakdown")
):
    """
    🎙️ World-Class AI Voice Detection

    Analyzes audio using a multi-stage pipeline:
    1. **Neural Model** — Wav2Vec2 deepfake classifier (multi-segment inference)
    2. **Spectral Analyzer** — Detects unnatural spectral patterns
    3. **Temporal Analyzer** — Catches robotic timing and uniform pauses
    4. **Formant Analyzer** — Identifies synthetic formant transitions
    5. **Artifact Detector** — Finds phase discontinuities and click artifacts

    Results are fused for maximum accuracy with evidence-based explanations.
    """
    try:
        # 1. Decode Base64
        audio_file = decode_base64_audio(request.audioBase64)

        # 2. Preprocess Audio (now returns profile too)
        audio_array, audio_profile = preprocess_audio(audio_file)

        # 3. Full Detection Pipeline (neural + forensics + fusion)
        result = voice_detector.predict(
            audio_array,
            audio_profile=audio_profile,
            detailed=detailed,
        )

        # 4. Generate Evidence-Based Explanation
        explanation = generate_explanation(result)

        # 5. Build Response
        response = DetectionResponse(
            status="success",
            language=request.language,
            classification=result["classification"],
            confidenceScore=round(result["confidence"], 2),
            explanation=explanation,
            inferenceTimeMs=result.get("inference_time_ms"),
            analyzersAgree=result.get("analyzers_agree"),
        )

        # Include detailed forensics if requested
        if detailed:
            response.forensics = result.get("forensics")
            response.artifactsSummary = result.get("artifacts_summary")
            if audio_profile:
                response.audioProfile = AudioProfileResponse(**audio_profile)

        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing the audio"
        )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
