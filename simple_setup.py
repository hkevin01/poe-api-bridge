#!/usr/bin/env python3
"""
Simple Poe token setup
"""

import subprocess
from pathlib import Path


def main():
    print("🦊 Poe Token Setup")
    print("=" * 50)
    
    # Open Firefox to poe.com
    print("📱 Opening Firefox to poe.com...")
    try:
        subprocess.Popen(["firefox", "https://poe.com"])
        print("✅ Firefox opened!")
    except Exception as e:
        print(f"⚠️  Could not open Firefox: {e}")
        print("📱 Please open Firefox manually and go to https://poe.com")
    
    print("\n📋 Instructions:")
    print("1. Sign in to your Poe account (or create one)")
    print("2. Press F12 to open Developer Tools")
    print("3. Click on 'Storage' tab (or 'Application' in newer versions)")
    print("4. Expand 'Cookies' → click on 'https://poe.com'")
    print("5. Find and copy the values for 'p-b' and 'p-lat'")
    print("\n" + "=" * 50)
    
    print("🔑 When you have your tokens, run this command:")
    print("python enter_tokens.py")
    print("\nOr manually create a .env file with:")
    print("POE_P_B_TOKEN=your_p_b_token")
    print("POE_P_LAT_TOKEN=your_p_lat_token")


if __name__ == "__main__":
    main() 