#!/usr/bin/env python3
"""
Simple test script for DriaClaude2 API
"""

import requests
import json
import time
import sys

API_BASE = "http://localhost:4144"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_tts():
    """Test TTS generation"""
    print("\nTesting TTS generation...")
    try:
        payload = {
            "text": "[S1] Hello, this is a test. [S2] Testing the TTS API! [S1] (laughs) It works!",
            "guidance_scale": 3.0,
            "temperature": 1.8,
            "output_format": "mp3"
        }
        
        response = requests.post(
            f"{API_BASE}/api/tts",
            json=payload
        )
        
        if response.status_code == 200:
            # Save the audio file
            with open("test_output.mp3", "wb") as f:
                f.write(response.content)
            print("✅ TTS generated successfully! Saved as test_output.mp3")
            return True
        else:
            print(f"❌ TTS failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_voices():
    """Test voice listing"""
    print("\nTesting voice listing...")
    try:
        response = requests.get(f"{API_BASE}/api/voices")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            voices = response.json()
            print(f"Found {len(voices['voices'])} voices")
            return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_nonverbal_tags():
    """Test nonverbal tags endpoint"""
    print("\nTesting nonverbal tags...")
    try:
        response = requests.get(f"{API_BASE}/api/nonverbal-tags")
        if response.status_code == 200:
            tags = response.json()
            print(f"Available nonverbal tags: {', '.join(tags['tags'][:5])}...")
            return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("🚀 DriaClaude2 API Test Suite")
    print("=" * 40)
    
    # Wait a bit for the service to be ready
    print("Waiting for service to be ready...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health),
        ("Voice Listing", test_voices),
        ("Nonverbal Tags", test_nonverbal_tags),
        ("TTS Generation", test_tts),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print(f"{'='*40}")
        result = test_func()
        results.append((test_name, result))
    
    print(f"\n{'='*40}")
    print("Test Results:")
    print(f"{'='*40}")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())