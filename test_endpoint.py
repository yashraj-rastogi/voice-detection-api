"""
Test the /detect endpoint with sample audio data.
"""
import requests
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

# API Configuration
BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/detect"

def create_sample_mp3_base64():
    """Create a minimal valid MP3 base64 string for testing"""
    # Minimal MP3 header (ID3v2.3)
    mp3_bytes = bytes([
        0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    ])
    return base64.b64encode(mp3_bytes).decode('utf-8')

def test_detect_endpoint(language="English", with_auth=True):
    """Test the /detect endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing /detect endpoint - Language: {language}")
    print(f"Authentication: {'Enabled' if with_auth else 'Disabled'}")
    print('='*60)
    
    # Prepare request
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": create_sample_mp3_base64()
    }
    
    headers = {"Content-Type": "application/json"}
    if with_auth and API_KEY:
        headers["X-API-Key"] = API_KEY
        print(f"Using API Key: {API_KEY[:10]}...")
    
    try:
        response = requests.post(ENDPOINT, json=payload, headers=headers, timeout=30)
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response:")
        print(response.json())
        
        return response
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_all_languages():
    """Test all supported languages"""
    languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    
    for lang in languages:
        test_detect_endpoint(language=lang, with_auth=True)

def test_health_check():
    """Test health check endpoint"""
    print(f"\n{'='*60}")
    print("Testing /health endpoint")
    print('='*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI VOICE DETECTION API - ENDPOINT TESTER")
    print("="*60)
    
    # Test health check
    test_health_check()
    
    # Test all supported languages
    test_all_languages()
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)
