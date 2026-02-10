"""
Quick test script for your deployed API on Hugging Face
"""
import requests
import base64
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Your deployed API endpoint
API_URL = "https://Pandaisop-voice-detection-api.hf.space/detect"
HEALTH_URL = "https://Pandaisop-voice-detection-api.hf.space/"

def create_test_audio():
    """Create minimal MP3 base64"""
    mp3_bytes = bytes([
        0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    ])
    return base64.b64encode(mp3_bytes).decode('utf-8')

def test_health():
    """Test if API is running"""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    try:
        response = requests.get(HEALTH_URL, timeout=10)
        print(f"✅ Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_with_lowercase_header():
    """Test with lowercase x-api-key (hackathon format)"""
    print("\n" + "="*70)
    print("TEST 2: Lowercase 'x-api-key' Header (Hackathon Format)")
    print("="*70)
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": create_test_audio()
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY  # Lowercase
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS! Response:")
            print(f"   - Status: {data.get('status')}")
            print(f"   - Language: {data.get('language')}")
            print(f"   - Classification: {data.get('classification')}")
            print(f"   - Confidence: {data.get('confidenceScore')}")
            return True
        else:
            print(f"❌ FAILED! Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_with_uppercase_header():
    """Test with uppercase X-API-Key header"""
    print("\n" + "="*70)
    print("TEST 3: Uppercase 'X-API-Key' Header (Standard Format)")
    print("="*70)
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": create_test_audio()
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY  # Uppercase
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ SUCCESS!")
            return True
        else:
            print(f"❌ FAILED! Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DEPLOYED API QUICK TEST")
    print("="*70)
    print(f"API URL: {API_URL}")
    print(f"API Key: {API_KEY[:10]}...")
    
    # Run tests
    health_ok = test_health()
    lowercase_ok = test_with_lowercase_header()
    uppercase_ok = test_with_uppercase_header()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"{'✅' if health_ok else '❌'} Health Check")
    print(f"{'✅' if lowercase_ok else '❌'} Lowercase Header (x-api-key) - HACKATHON FORMAT")
    print(f"{'✅' if uppercase_ok else '❌'} Uppercase Header (X-API-Key)")
    
    if lowercase_ok and uppercase_ok:
        print("\n🎉 ALL TESTS PASSED! Your API is ready for hackathon submission!")
        print("\n📋 Use these details for hackathon tester:")
        print(f"   Endpoint: {API_URL}")
        print(f"   Header: x-api-key: {API_KEY}")
    else:
        print("\n⚠️  Some tests failed. You may need to redeploy.")
        print("\n💡 Check Space status: https://huggingface.co/spaces/Pandaisop/voice-detection-api")
