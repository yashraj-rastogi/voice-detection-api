"""
Final validation script - Tests your deployed API with real audio sample.
Download the sample from: https://drive.google.com/file/d/1n2RsLy-jfY025IbbaRQMex-KVgePG3zV/view?usp=drive_link
"""
import requests
import base64
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
API_KEY = os.getenv("API_KEY")

def load_audio_base64(file_path):
    """Load MP3 file and convert to base64"""
    try:
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
            return base64.b64encode(audio_bytes).decode('utf-8')
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None

def test_with_real_audio(api_url, audio_file, language="English"):
    """Test API with real audio sample"""
    print(f"\n{'='*70}")
    print(f"Testing with Real Audio Sample")
    print('='*70)
    
    # Load audio
    base64_audio = load_audio_base64(audio_file)
    if not base64_audio:
        return None
    
    print(f"✅ Loaded audio file: {audio_file}")
    print(f"   Base64 length: {len(base64_audio)} characters")
    
    # Prepare request (EXACT hackathon format)
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": base64_audio
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY  # Lowercase as per hackathon
    }
    
    print(f"\n📤 Sending request to: {api_url}")
    
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        
        print(f"\n📥 Response:")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ SUCCESS!")
            print(f"   Status: {data.get('status')}")
            print(f"   Language: {data.get('language')}")
            print(f"   Classification: {data.get('classification')}")
            print(f"   Confidence Score: {data.get('confidenceScore')}")
            print(f"   Explanation: {data.get('explanation')}")
            return data
        else:
            print(f"\n❌ FAILED!")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FINAL VALIDATION - Real Audio Test")
    print("="*70)
    
    # Get API URL
    api_url = input("\nEnter your deployed API URL (or press Enter for localhost): ").strip()
    if not api_url:
        api_url = "http://localhost:8000/detect"
    
    print(f"\nAPI URL: {api_url}")
    print(f"API Key: {API_KEY[:10]}...")
    
    # Check for audio files
    audio_files = list(Path(".").glob("*.mp3"))
    
    if not audio_files:
        print("\n⚠️  No MP3 files found in current directory")
        print("\nPlease:")
        print("1. Download sample from: https://drive.google.com/file/d/1n2RsLy-jfY025IbbaRQMex-KVgePG3zV/view?usp=drive_link")
        print("2. Place the MP3 file in this directory")
        print("3. Run this script again")
        
        manual_path = input("\nOr enter path to MP3 file: ").strip()
        if manual_path and os.path.exists(manual_path):
            audio_files = [Path(manual_path)]
        else:
            print("\n❌ No valid audio file provided")
            exit(1)
    
    print(f"\n📁 Found {len(audio_files)} MP3 file(s)")
    
    for audio_file in audio_files:
        print(f"\n{'='*70}")
        print(f"Testing: {audio_file.name}")
        print('='*70)
        
        # Ask for language
        print("\nSupported languages:")
        print("1. Tamil")
        print("2. English")
        print("3. Hindi")
        print("4. Malayalam")
        print("5. Telugu")
        
        lang_choice = input("\nSelect language (1-5, default=2): ").strip() or "2"
        languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        language = languages[int(lang_choice) - 1] if lang_choice.isdigit() and 1 <= int(lang_choice) <= 5 else "English"
        
        # Test
        result = test_with_real_audio(api_url, audio_file, language)
        
        if result:
            print(f"\n{'='*70}")
            print("✅ VALIDATION PASSED!")
            print('='*70)
            print("\nYour API is ready for hackathon submission!")
        else:
            print(f"\n{'='*70}")
            print("❌ VALIDATION FAILED!")
            print('='*70)
            print("\nPlease check:")
            print("1. API is deployed and accessible")
            print("2. API key is correct")
            print("3. Audio file is valid MP3")
