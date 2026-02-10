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

# 🎙️ VoiceGuard — AI Voice Detection API (v2.0)

**Detect AI-Generated Voices with World-Class Accuracy.**

> A production-grade deepfake audio detection system with a custom training pipeline,
> built for the real world — noisy environments, lossy codecs, and multilingual audio.

---

## 🌟 Features

- **Web UI:** Record or upload audio directly in the browser.
- **REST API:** `/detect` endpoint for developers (Hackathon Compliant).
- **Advanced Model:** Wav2Vec2-based deepfake audio classification.
- **Multilingual:** English, Tamil, Hindi, Telugu, Malayalam.
- **Formats:** WAV, MP3, FLAC.
- **Secure:** API Key authentication.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    VoiceGuard API                         │
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  FastAPI  │───▶│ Audio Engine │───▶│  Wav2Vec2     │  │
│  │  Server   │    │ (Librosa)    │    │  Classifier   │  │
│  └──────────┘    └──────────────┘    └───────────────┘  │
│       │                                      │           │
│       ▼                                      ▼           │
│  ┌──────────┐                        ┌───────────────┐  │
│  │  Web UI  │                        │  JSON Result   │  │
│  │ (Record/ │                        │ HUMAN /        │  │
│  │  Upload) │                        │ AI_GENERATED   │  │
│  └──────────┘                        └───────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Web Interface
Open the app at [https://Pandaisop-voice-detection-api.hf.space/](https://Pandaisop-voice-detection-api.hf.space/) and record or upload audio.

### Option 2: API Usage

```json
POST /detect
Content-Type: application/json
X-API-Key: YOUR_API_KEY

{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "UklGR..."
}
```

**Response:**
```json
{
  "result": "AI_GENERATED",
  "confidence": 0.9743,
  "language": "English"
}
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI + Uvicorn | High-performance async API server |
| **AI Model** | Wav2Vec2 (Transformers) | Self-supervised speech representation |
| **Audio DSP** | Librosa + SoundFile | Audio loading, resampling, normalization |
| **Deployment** | Docker + Hugging Face Spaces | Containerized cloud deployment |
| **Auth** | API Key (X-API-Key header) | Secure endpoint access |

---

## 🧠 Custom Training Pipeline

VoiceGuard includes a **research-grade training pipeline** in `trainer/` for building custom detection models optimized for your specific use case.

### Architecture

```
Raw Audio → Data Engine → Augmentation → SSL Backbone → Classifier → HUMAN / AI
                                            │
                                   ┌────────┼────────┐
                                   │        │        │
                              Wav2Vec2   HuBERT   WavLM
                                   │        │        │
                                   └────────┼────────┘
                                            │
                                      Ensemble Fusion
```

### Training Features

| Feature | Details |
|---------|---------|
| **3 SSL Backbones** | Wav2Vec2-XLSR-53 (multilingual), HuBERT-Large, WavLM-Large |
| **Ensemble Modes** | Late Fusion, Learned Fusion, Confidence-Weighted |
| **Pooling** | Attentive Statistics Pooling (ECAPA-TDNN style) |
| **Loss Function** | Focal Loss (handles class imbalance) |
| **Optimizer** | AdamW + Cosine Annealing with Warm Restarts |
| **Regularization** | EMA, Dropout, BatchNorm, Gradient Clipping |
| **Mixed Precision** | FP16 training for 2× speed on GPU |
| **Data Augmentation** | 7 types (see below) |

### 7-Type Augmentation Pipeline

| Augmentation | Purpose |
|---|---|
| **Additive Noise** (Gaussian, SNR-controlled) | Robustness to noisy recordings |
| **Speed Perturbation** (0.9×–1.1×) | Handle varied speech rates |
| **Pitch Shift** (±2 semitones) | Speaker variability |
| **SpecAugment** (time/freq masking) | Regularization (proven in ASR research) |
| **Codec Simulation** (MP3/OGG encode-decode) | Handle lossy compression artifacts |
| **Volume Perturbation** (±6 dB) | Microphone gain variability |
| **RIR Convolution** (room impulse response) | Simulate different room acoustics |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **EER** (Equal Error Rate) | Standard for spoofing detection (ASVspoof) |
| **AUC-ROC** | Threshold-independent classification quality |
| **ECE** (Expected Calibration Error) | Is 80% confidence really 80% accurate? |
| **Per-Language Breakdown** | No language left behind |
| **Latency Benchmark** | Inference speed per sample |

### Training Workflow

```bash
cd trainer/

# 1. Add samples to data/human/ and data/ai/
# 2. Validate, analyze, and split
python prepare_data.py

# 3. Train (GPU recommended)
python train.py --config config.yaml

# 4. Evaluate
python evaluate_model.py

# 5. Export & auto-deploy
python export_model.py --integrate
```

### Export Formats
- **HuggingFace Hub** — Push directly to your model repository
- **ONNX** — Optimized CPU inference with optional quantization
- **TorchScript** — Portable PyTorch format
- **Auto-Integration** — One command to update the running API

---

## 📁 Project Structure

```
voice-detection-api/
├── app/
│   ├── main.py              # FastAPI server + /detect endpoint
│   ├── config.py             # Model & API configuration
│   ├── core/
│   │   ├── audio.py          # Audio preprocessing pipeline
│   │   └── model.py          # VoiceDetector (inference engine)
│   └── static/
│       └── index.html        # Web UI
├── trainer/                  # 🧠 Custom Model Training Pipeline
│   ├── config.yaml           # All hyperparameters (single source of truth)
│   ├── prepare_data.py       # Data validation, SNR analysis, stratified splits
│   ├── augment.py            # 7-type augmentation pipeline
│   ├── train.py              # Training engine (Focal Loss, EMA, FP16)
│   ├── evaluate_model.py     # EER, AUC-ROC, calibration, per-language
│   ├── export_model.py       # Export to HF Hub / ONNX / TorchScript
│   ├── models/
│   │   ├── backbone.py       # SSL model loaders (Wav2Vec2, HuBERT, WavLM)
│   │   ├── classifier.py     # Attentive Stats Pooling + MLP head
│   │   └── ensemble.py       # 3 ensemble fusion strategies
│   └── data/
│       ├── human/            # Real voice samples
│       └── ai/               # AI-generated samples
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🛠️ Local Development

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

---

## 👥 Team

Built for hackathon by **Vineet Shukla** & **Yashraj Rastogi**.

## 📄 License

MIT License
