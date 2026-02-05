"""
Create test audio samples and convert to base64 for hackathon testing.
This script helps you prepare audio samples for the hackathon tester endpoint.
"""
import base64
import os
from pathlib import Path

def convert_audio_to_base64(audio_file_path):
    """Convert an MP3 file to base64 string"""
    try:
        with open(audio_file_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            return base64_audio
    except FileNotFoundError:
        print(f"❌ File not found: {audio_file_path}")
        return None
    except Exception as e:
        print(f"❌ Error converting file: {e}")
        return None

def save_base64_to_file(base64_string, output_file):
    """Save base64 string to a text file"""
    try:
        with open(output_file, 'w') as f:
            f.write(base64_string)
        print(f"✅ Base64 saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

def create_test_payload(language, base64_audio):
    """Create a complete test payload for the API"""
    return {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": base64_audio
    }

def main():
    print("=" * 60)
    print("Audio to Base64 Converter for Hackathon Testing")
    print("=" * 60)
    
    # Check for audio files in current directory
    audio_dir = Path(".")
    mp3_files = list(audio_dir.glob("*.mp3"))
    
    if not mp3_files:
        print("\n⚠️  No MP3 files found in current directory")
        print("\nTo use this script:")
        print("1. Download sample audio from hackathon")
        print("2. Place MP3 files in this directory")
        print("3. Run this script again")
        print("\nOr specify a file path manually:")
        
        file_path = input("\nEnter MP3 file path (or press Enter to skip): ").strip()
        if file_path and os.path.exists(file_path):
            mp3_files = [Path(file_path)]
        else:
            print("\n❌ No valid audio files to process")
            return
    
    print(f"\n📁 Found {len(mp3_files)} MP3 file(s):\n")
    
    for idx, mp3_file in enumerate(mp3_files, 1):
        print(f"{idx}. {mp3_file.name}")
        
        # Convert to base64
        base64_audio = convert_audio_to_base64(mp3_file)
        
        if base64_audio:
            # Save to text file
            output_file = mp3_file.stem + "_base64.txt"
            save_base64_to_file(base64_audio, output_file)
            
            # Show preview
            preview_length = 100
            print(f"   Preview: {base64_audio[:preview_length]}...")
            print(f"   Length: {len(base64_audio)} characters")
            
            # Ask for language
            print("\n   Supported languages:")
            print("   1. Tamil")
            print("   2. English")
            print("   3. Hindi")
            print("   4. Malayalam")
            print("   5. Telugu")
            
            lang_choice = input("\n   Select language (1-5) or press Enter to skip: ").strip()
            
            languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
            if lang_choice.isdigit() and 1 <= int(lang_choice) <= 5:
                language = languages[int(lang_choice) - 1]
                
                # Create complete payload
                payload = create_test_payload(language, base64_audio)
                
                # Save payload to JSON file
                import json
                payload_file = mp3_file.stem + f"_{language.lower()}_payload.json"
                with open(payload_file, 'w') as f:
                    json.dump(payload, f, indent=2)
                print(f"   ✅ Test payload saved to: {payload_file}")
            
            print()
    
    print("=" * 60)
    print("✅ Conversion Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Copy the base64 string from the .txt file")
    print("2. Use it in the hackathon tester endpoint")
    print("3. Or use the JSON payload file for automated testing")

if __name__ == "__main__":
    main()
