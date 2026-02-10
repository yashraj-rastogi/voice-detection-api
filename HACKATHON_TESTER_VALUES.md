# Hackathon Tester - Fill-in Guide

## 🎯 Exact Values for the Hackathon API Endpoint Tester

Copy and paste these values directly into the tester interface:

---

### 1️⃣ Headers Section
**Field Name**: `x-api-key`  
**Value** (copy exactly):
```
uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY
```

---

### 2️⃣ Endpoint URL
**Value** (copy exactly):
```
https://Pandaisop-voice-detection-api.hf.space/detect
```

---

### 3️⃣ Request Body

#### Language
**Value** (choose one):
```
English
```
Or: `Tamil`, `Hindi`, `Malayalam`, `Telugu`

#### Audio Format
**Value** (copy exactly):
```
mp3
```

#### Audio Base64 Format
**Test Sample** (minimal MP3 - copy exactly):
```
SUQzAwAAAAAAAAAA//uQAAAAAAAA
```

**OR** if you have a real MP3 audio file:
1. Save your MP3 file
2. Run this command:
   ```bash
   python create_test_audio.py path/to/your/audio.mp3
   ```
3. Copy the base64 output

---

## 🚨 CRITICAL: Before Testing

### ⚠️ Have you deployed the updated code?

**Check your Space status**: https://huggingface.co/spaces/Pandaisop/voice-detection-api

**If the Space shows OLD code** (doesn't have the dual header support):
1. Go to: https://huggingface.co/spaces/Pandaisop/voice-detection-api
2. Files → app → main.py → Edit
3. Update `verify_api_key` function (see DEPLOYMENT_STATUS.md)
4. Commit: "Fix: Accept both x-api-key and X-API-Key headers"
5. **Wait 2-3 minutes** for rebuild
6. **Then** test with the hackathon tester

---

## ✅ Expected Result

If deployment is successful, you should see:
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED" or "HUMAN",
  "confidenceScore": 0.XX,
  "explanation": "..."
}
```

---

## 🧪 Quick Pre-Test

Before using the hackathon tester, verify your API works:
```bash
python test_deployed_api_quick.py
```

This will confirm both lowercase and uppercase headers work!

---

## 📸 Screenshot Reference

Your hackathon tester should look like this when filled:
- **x-api-key**: `uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY`
- **Endpoint URL**: `https://Pandaisop-voice-detection-api.hf.space/detect`
- **Language**: `English`
- **Audio Format**: `mp3`
- **Audio Base64**: `SUQzAwAAAAAAAAAA//uQAAAAAAAA`

Click **"Test Endpoint"** button!
