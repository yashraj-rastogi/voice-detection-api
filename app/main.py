

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
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
    title="AI Voice Detection API",
    description="API to detect whether a voice sample is AI-generated or Human.",
    version="2.0.0"
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
    """Return error responses in hackathon-compliant format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error"
        }
    )

# Request Model (Hackathon Specification)
class AudioRequest(BaseModel):
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(
        ..., 
        description="Language of the audio (Tamil/English/Hindi/Malayalam/Telugu)"
    )
    audioFormat: Literal["mp3"] = Field(
        ..., 
        description="Audio format (must be mp3)"
    )
    audioBase64: str = Field(
        ..., 
        description="Base64-encoded MP3 audio"
    )

# Response Model (Hackathon Specification)
class DetectionResponse(BaseModel):
    status: Literal["success", "error"] = Field(
        default="success",
        description="Response status"
    )
    language: str = Field(..., description="Language of the audio")
    classification: Literal["AI_GENERATED", "HUMAN"] = Field(
        ..., 
        description="Voice classification result"
    )
    confidenceScore: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    explanation: str = Field(
        ..., 
        description="Short reason for the decision"
    )

# Error Response Model
class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str

def verify_api_key(
    x_api_key_lower: str = Header(None, alias="x-api-key"),
    x_api_key_upper: str = Header(None, alias="X-API-Key")
):
    """
    Verify the API key from the request header.
    Accepts both 'x-api-key' (hackathon format) and 'X-API-Key' for compatibility.
    If API_KEY is not configured, this check is skipped (development mode).
    """
    # Get the API key from either header (hackathon uses lowercase)
    x_api_key = x_api_key_lower or x_api_key_upper
    
    # If no API key is configured, allow all requests (development mode)
    if not settings.API_KEY:
        return True
    
    # If API key is configured, verify it
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required. Please provide x-api-key header."
        )
    
    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return True

@app.get("/")
def read_root():
    """
    Serve the Web UI.
    """
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    auth_status = "enabled" if settings.API_KEY else "disabled (development mode)"
    return {
        "status": "healthy",
        "message": "AI Voice Detection API is running. (UI not found)",
        "authentication": auth_status
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring.
    Returns API status, version, and model info.
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "model": settings.MODEL_NAME,
        "authentication": "enabled" if settings.API_KEY else "disabled"
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_voice(
    request: AudioRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Analyzes Base64 MP3 audio and determines if it is AI-generated or Human.
    Supports: Tamil, English, Hindi, Malayalam, Telugu
    """
    try:
        # 1. Decode Base64
        audio_file = decode_base64_audio(request.audioBase64)
        
        # 2. Preprocess Audio
        audio_tensor = preprocess_audio(audio_file)
        
        # 3. Predict
        label, confidence = voice_detector.predict(audio_tensor)
        
        # 4. Generate explanation
        explanation = generate_explanation(label, confidence)
        
        return DetectionResponse(
            status="success",
            language=request.language,
            classification=label,
            confidenceScore=round(confidence, 2),
            explanation=explanation
        )
        
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
