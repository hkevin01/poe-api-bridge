#!/usr/bin/env python3
"""
Test script for the Poe Code Chat backend
"""

import json
import time

import requests


def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_models():
    """Test models endpoint"""
    print("\n🔍 Testing models endpoint...")
    response = requests.get("http://localhost:8000/v1/models")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        models = response.json()
        print(f"Available models: {len(models['data'])}")
        for model in models['data'][:5]:  # Show first 5
            print(f"  - {model['id']}")
    return response.status_code == 200


def test_chat():
    """Test chat completions"""
    print("\n🔍 Testing chat completions...")
    
    payload = {
        "model": "claude-3-haiku",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful coding assistant. The user is working on a VS Code extension project."
            },
            {
                "role": "user", 
                "content": "Hello! Can you help me understand how to improve my code?"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['choices'][0]['message']['content'][:100]}...")
        print(f"Usage: {result.get('usage', 'N/A')}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def main():
    """Run all tests"""
    print("🧪 Testing Poe Code Chat Backend")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Models Endpoint", test_models), 
        ("Chat Completions", test_chat)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✅ PASS" if success else "❌ FAIL"))
        except Exception as e:
            results.append((name, f"❌ ERROR: {e}"))
        
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    for name, result in results:
        print(f"  {name}: {result}")


if __name__ == "__main__":
    main() 