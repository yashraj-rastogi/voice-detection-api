# ✅ Files Upload Checklist for Hugging Face

## 📦 What You Need to Upload

Here's EXACTLY what to upload to your Space via the web interface:

### Required Files/Folders:

1. **`app/` folder** - Your entire application code
   - Contains: `main.py`, `config.py`, `core/` folder, `static/` folder
   - This is THE MOST IMPORTANT folder

2. **`requirements.txt`** - Python dependencies
   - Location: `x:\voice-detection-api\requirements.txt`

3. **`Dockerfile`** - Docker configuration for HF Spaces
   - Location: `x:\voice-detection-api\Dockerfile`

4. **`README.md`** - Space description and metadata
   - Location: `x:\voice-detection-api\README.md`

### DO NOT Upload:
- `.env` file (contains your secret API key)
- `venv/` folder
- `.git/` folder  
- Test files (`test_*.py`)
- Documentation files (`*.md` except README.md)

---

## 🎯 STEP-BY-STEP Upload Process

### Option A: Upload via Hugging Face Web (EASIEST - No git needed!)

1. **Open your Space**:
   https://huggingface.co/spaces/Pandaisop/voice-detection-api

2. **Click "Files" tab** (at the top)

3. **Click "Add file" → "Upload files"** (button on the right)

4. **Drag and drop OR select these items**:
   - The entire `app` folder from: `x:\voice-detection-api\app`
   - `requirements.txt` from: `x:\voice-detection-api\requirements.txt`
   - `Dockerfile` from: `x:\voice-detection-api\Dockerfile`
   - `README.md` from: `x:\voice-detection-api\README.md`

5. **Commit message**: Type this:
   ```
   Add API code with dual header support for hackathon
   ```

6. **Click "Commit changes to main"**

7. **IMPORTANT: Add your API Key as a Secret**:
   - Go to "Settings" tab
   - Scroll to "Repository secrets"
   - Click "New secret"
   - Name: `API_KEY`
   - Value: `uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY`
   - Click "Add secret"

8. **Wait for build** (2-3 minutes):
   - Watch the status at the top change from "Building" to "Running"
   - Status indicator should turn GREEN

9. **Verify it's working**:
   - Visit: https://Pandaisop-voice-detection-api.hf.space/
   - You should see your Web UI!

---

## 🧪 After Upload - Test It!

Run this command from your local machine:

```bash
python test_deployed_api_quick.py
```

Expected output:
```
✅ Health Check
✅ Lowercase Header (x-api-key) - HACKATHON FORMAT  
✅ Uppercase Header (X-API-Key)

🎉 ALL TESTS PASSED! Your API is ready for hackathon submission!
```

---

## 🎯 Then Use the Hackathon Tester

Now that your API is deployed, fill in the hackathon tester with:

- **x-api-key**: `uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY`
- **Endpoint URL**: `https://Pandaisop-voice-detection-api.hf.space/detect`
- **Language**: `English`
- **Audio Format**: `mp3`
- **Audio Base64 Format**: `SUQzAwAAAAAAAAAA//uQAAAAAAAA`

Click **"Test Endpoint"** button!

---

## ❓ Troubleshooting

### If upload fails or times out:
- Try uploading files one at a time instead of all at once
- Start with `app` folder first, then add others

### If build fails:
- Check the "Logs" tab on your Space
- Make sure `API_KEY` secret was added in Settings

### If you see "Error" status:
- Look at the logs to see what went wrong
- Usually it's a missing dependency in `requirements.txt`

---

## 🆘 If Web Upload Doesn't Work

Try the git command with your HF token embedded:

```bash
git push https://hf_YOUR_TOKEN@huggingface.co/spaces/Pandaisop/voice-detection-api main --force
```

Get your token from: https://huggingface.co/settings/tokens
