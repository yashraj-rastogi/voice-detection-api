# Quick Deployment Guide - Enhanced Code

## What You're Deploying

### Enhanced Files:
1. **`app/core/audio.py`** - Advanced preprocessing with noise reduction
2. **`app/core/model.py`** - Confidence calibration

### Also Deploying (from previous work):
3. **`app/main.py`** - Dual header support (x-api-key + X-API-Key)

---

## Method 1: Git Push (Try This First)

```bash
git push hf main --force
```

**If it asks for credentials:**
- Username: `Pandaisop`
- Password: [Your HF Token from https://huggingface.co/settings/tokens]

---

## Method 2: Web Upload (If Git Fails)

### Step-by-Step:

1. **Go to your Space:**
   https://huggingface.co/spaces/Pandaisop/voice-detection-api

2. **Upload `audio.py`:**
   - Click "Files" tab
   - Navigate to: `app/core/`
   - Click "Add file" → "Upload files"
   - Upload: `x:\voice-detection-api\app\core\audio.py`
   - Commit message: "Enhanced audio preprocessing"
   - Click "Commit"

3. **Upload `model.py`:**
   - Same folder: `app/core/`
   - Upload: `x:\voice-detection-api\app\core\model.py`
   - Commit message: "Added confidence calibration"
   - Click "Commit"

4. **Upload `main.py`** (if not already uploaded):
   - Navigate to: `app/`
   - Upload: `x:\voice-detection-api\app\main.py`
   - Commit message: "Dual header support"
   - Click "Commit"

5. **Wait for rebuild:** Watch for "Building" → "Running" (2-3 min)

---

## After Deployment - Test It!

```bash
python test_deployed_api_quick.py
```

Expected output:
```
✅ Health Check
✅ Lowercase Header (x-api-key) - HACKATHON FORMAT
✅ Uppercase Header (X-API-Key)

🎉 ALL TESTS PASSED!
```

---

## Verify Enhancements Active

Look for these in your Space logs after deployment:
```
INFO: Preprocessed audio: duration=X.XXs, shape=...
INFO: Prediction: ... (confidence: X.XXXX, margin: X.XXXX)
```

This confirms the enhancements are working!
