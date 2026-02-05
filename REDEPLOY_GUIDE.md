# Quick Redeploy Guide to Hugging Face

## Your Deployment Info
- **Space URL**: https://huggingface.co/spaces/Pandaisop/voice-detection-api
- **API Endpoint**: https://Pandaisop-voice-detection-api.hf.space/detect

## Step 1: Commit Your Changes

```bash
git add .
git commit -m "Fix: Accept both x-api-key and X-API-Key headers for hackathon compatibility"
```

## Step 2: Push to Hugging Face

```bash
git push hf main
```

**Note:** If you get an error, you may need to force push:
```bash
git push hf main --force
```

## Step 3: Wait for Deployment (2-3 minutes)

1. Visit: https://huggingface.co/spaces/Pandaisop/voice-detection-api
2. Watch the "Building" status at the top
3. Wait until it shows "Running"

## Step 4: Test Your Deployment

```bash
python test_deployed_api_quick.py
```

## Alternative: Manual Git Setup (if needed)

If you don't have the `hf` remote configured:

```bash
git remote add hf https://huggingface.co/spaces/Pandaisop/voice-detection-api
git push hf main
```

## Troubleshooting

### If push fails with authentication error:
1. Get your Hugging Face token from: https://huggingface.co/settings/tokens
2. Use it as password when prompted

### If Space shows error:
1. Check logs on the Space page
2. Verify all files are committed
3. Check that `requirements.txt` is up to date
