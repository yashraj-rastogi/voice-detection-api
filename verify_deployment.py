"""Quick verification that the deployed API is working"""
import requests
import json
import base64

API_URL = "https://Pandaisop-voice-detection-api.hf.space/detect"
API_KEY = "uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY"

# Create minimal MP3 test data
mp3_bytes = bytes([
    0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
])
test_audio = base64.b64encode(mp3_bytes).decode('utf-8')

print("=" * 70)
print("DEPLOYED API VERIFICATION")
print("=" * 70)
print(f"API URL: {API_URL}")
print()

# Test with lowercase x-api-key (HACKATHON FORMAT)
print("Testing with lowercase 'x-api-key' header (Hackathon Format)...")

payload = {
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": test_audio
}

headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY  # lowercase - hackathon format
}

try:
    response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ SUCCESS! API Response:")
        print(json.dumps(data, indent=2))
        print("\n🎉 Your API is READY for the hackathon!")
        print("\nUse these values in the hackathon tester:")
        print(f"  - Endpoint: {API_URL}")
        print(f"  - Header: x-api-key: {API_KEY}")
        print(f"  - Language: English")
        print(f"  - Audio Format: mp3")
        print(f"  - Audio Base64: {test_audio}")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"\n❌ Error: {e}")
