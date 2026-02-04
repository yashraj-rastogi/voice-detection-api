"""
Quick checker to see if Hugging Face deployment is ready
"""
import requests
import time

SPACE_URL = "https://Pandaisop-voice-detection-api.hf.space/"

print("🔍 Checking if Hugging Face Space is ready...\n")

try:
    response = requests.get(SPACE_URL, timeout=10)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            print("✅ SUCCESS! Your Space is RUNNING! (Web UI is live)")
            print(f"   The root URL now serves the new Voice Detection Interface.")
        else:
            print("✅ Deployment Live (JSON API)")
            print(response.json())
            
        print("\n🎉 You can now test your API!")
        print(f"🔗 Space URL: https://huggingface.co/spaces/Pandaisop/voice-detection-api")
    else:
        print(f"⚠️  Space responded with status code: {response.status_code}")
        print("It might still be building...")
        
except requests.exceptions.ConnectionError:
    print("❌ Connection failed!")
    print("The Space is likely still building or starting up.")
    print("\n💡 Wait 1-2 more minutes and try again.")
    
except requests.exceptions.Timeout:
    print("⏱️  Request timed out!")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")

print(f"\n🌐 Check manually at: https://huggingface.co/spaces/Pandaisop/voice-detection-api")
