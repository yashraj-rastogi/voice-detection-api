"""
Test the deployed API with hackathon-compliant format.
This script tests your deployed API exactly as the hackathon tester will.
"""
import requests
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

# IMPORTANT: Update this with your deployed URL
DEPLOYED_URL = input("Enter your deployed API URL (e.g., https://your-app.com/detect): ").strip()

if not DEPLOYED_URL:
    print("❌ No URL provided. Using localhost for testing...")
    DEPLOYED_URL = "http://localhost:8000/detect"

def create_minimal_mp3_base64():
    """Create a minimal valid MP3 base64 string"""
    mp3_bytes = bytes([
        0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    ])
    return base64.b64encode(mp3_bytes).decode('utf-8')

def test_hackathon_format(language="English"):
    """
    Test with EXACT hackathon format.
    CRITICAL: Hackathon uses lowercase 'x-api-key' header!
    """
    print(f"\n{'='*70}")
    print(f"Testing Hackathon Format - Language: {language}")
    print('='*70)
    
    # EXACT hackathon request format
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": create_minimal_mp3_base64()
    }
    
    # CRITICAL: Hackathon uses lowercase 'x-api-key'
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY  # Lowercase as per hackathon spec!
    }
    
    print(f"\n📤 Request:")
    print(f"URL: {DEPLOYED_URL}")
    print(f"Headers: {headers}")
    print(f"Payload: {payload}")
    
    try:
        response = requests.post(DEPLOYED_URL, json=payload, headers=headers, timeout=30)
        
        print(f"\n📥 Response:")
        print(f"Status Code: {response.status_code}")
        print(f"Response Body:")
        print(response.json())
        
        # Validate response format
        if response.status_code == 200:
            data = response.json()
            required_fields = ["status", "language", "classification", "confidenceScore", "explanation"]
            missing_fields = [f for f in required_fields if f not in data]
            
            if missing_fields:
                print(f"\n❌ Missing required fields: {missing_fields}")
            else:
                print(f"\n✅ All required fields present!")
                
                # Validate field values
                if data["status"] == "success":
                    if data["classification"] in ["AI_GENERATED", "HUMAN"]:
                        print(f"✅ Valid classification: {data['classification']}")
                    else:
                        print(f"❌ Invalid classification: {data['classification']}")
                    
                    if 0.0 <= data["confidenceScore"] <= 1.0:
                        print(f"✅ Valid confidence score: {data['confidenceScore']}")
                    else:
                        print(f"❌ Invalid confidence score: {data['confidenceScore']}")
        else:
            print(f"\n❌ Request failed with status {response.status_code}")
            
        return response
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request Error: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

def test_all_languages():
    """Test all 5 supported languages"""
    languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    results = {}
    
    for lang in languages:
        response = test_hackathon_format(language=lang)
        results[lang] = response.status_code if response else None
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    for lang, status in results.items():
        status_icon = "✅" if status == 200 else "❌"
        print(f"{status_icon} {lang}: {status}")

def test_with_uppercase_header():
    """Test with uppercase X-API-Key header (current implementation)"""
    print(f"\n{'='*70}")
    print("Testing with Uppercase X-API-Key Header")
    print('='*70)
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": create_minimal_mp3_base64()
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY  # Uppercase
    }
    
    try:
        response = requests.post(DEPLOYED_URL, json=payload, headers=headers, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("\n" + "="*70)
    print("HACKATHON API COMPLIANCE TESTER")
    print("="*70)
    
    if not API_KEY:
        print("\n❌ ERROR: No API_KEY found in .env file")
        exit(1)
    
    print(f"\nUsing API Key: {API_KEY[:10]}...")
    print(f"Testing URL: {DEPLOYED_URL}")
    
    # Test with hackathon format (lowercase header)
    print("\n" + "="*70)
    print("TEST 1: Hackathon Format (lowercase 'x-api-key')")
    print("="*70)
    test_hackathon_format(language="English")
    
    # Test with uppercase header (current implementation)
    print("\n" + "="*70)
    print("TEST 2: Current Implementation (uppercase 'X-API-Key')")
    print("="*70)
    test_with_uppercase_header()
    
    # Test all languages
    choice = input("\n\nTest all 5 languages? (y/n): ").strip().lower()
    if choice == 'y':
        test_all_languages()
    
    print("\n" + "="*70)
    print("Testing Complete!")
    print("="*70)
    print("\n⚠️  IMPORTANT: If lowercase 'x-api-key' failed, your API needs to be updated!")
    print("   The hackathon tester uses lowercase 'x-api-key' header.")
