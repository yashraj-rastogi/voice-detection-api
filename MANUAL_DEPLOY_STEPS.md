# Manual Deployment Steps for Hugging Face

## ⚠️ Git Push Issue Detected

The automated git push is encountering an error. Here's how to deploy manually:

## Option 1: Use Hugging Face Web Interface (EASIEST)

### Step 1: Go to Your Space
Visit: https://huggingface.co/spaces/Pandaisop/voice-detection-api

### Step 2: Click "Files" Tab
Click on the "Files" tab at the top

### Step 3: Edit `app/main.py`
1. Click on `app/main.py`
2. Click the "Edit" button
3. Find the `verify_api_key` function (around line 106)
4. Replace it with this updated version:

```python
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
```

5. Click "Commit changes to main"
6. Wait 2-3 minutes for rebuild

## Option 2: Fix Git Authentication

### Check if you need to authenticate:
```bash
git config credential.helper store
```

### Then try pushing again:
```bash
git push hf main --force
```

When prompted:
- **Username**: Pandaisop
- **Password**: [Your Hugging Face token from https://huggingface.co/settings/tokens]

## Option 3: Create New Remote

```bash
# Remove old remote
git remote remove hf

# Add new remote with token
git remote add hf https://YOUR_HF_TOKEN@huggingface.co/spaces/Pandaisop/voice-detection-api

# Push
git push hf main --force
```

## After Deployment

Wait 2-3 minutes, then test:
```bash
python test_deployed_api_quick.py
```

## Verify Space is Running

Visit: https://huggingface.co/spaces/Pandaisop/voice-detection-api

Look for "Running" status (green indicator)
