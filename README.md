---
title: AI Voice Detection API
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# 🎙️ VoiceGuard - AI Voice Detection API (v2.0)

**Detect AI-Generated Voices with High Accuracy.**

## 🌟 Features
- **Web UI:** Record or upload audio directly in the browser to test.
- **API Endpoint:** `/detect` endpoint for developers (Hackathon Compliant).
- **Advanced Model:** Uses `mo-thecreator/Deepfake-audio-detection`.
- **Supports:** WAV, MP3, FLAC.

## 🚀 How to Use

### Option 1: Web Interface (Easy)
Just open the App tab above! You can record your voice or upload a file.

### Option 2: API Usage
Send a POST request to `/detect`:

```json
POST /detect
Content-Type: application/json
X-API-Key: YOUR_API_KEY (if enabled)

{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "UklGR..."
}
```

## 🛠️ Local Development

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```
