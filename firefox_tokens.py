#!/usr/bin/env python3
"""
Firefox token extraction helper
"""

import subprocess
import time
from pathlib import Path


def main():
    print("🦊 Firefox Poe Token Setup")
    print("=" * 50)
    
    # Check if Firefox is running
    try:
        result = subprocess.run(["pgrep", "firefox"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Firefox is running")
        else:
            print("📱 Starting Firefox...")
            subprocess.Popen(["firefox", "https://poe.com"])
            time.sleep(3)
    except Exception as e:
        print(f"⚠️  Could not check Firefox: {e}")
        print("📱 Please open Firefox manually and go to https://poe.com")
    
    print("\n📋 Instructions:")
    print("1. Make sure you're logged into poe.com in Firefox")
    print("2. Press F12 to open Developer Tools")
    print("3. Click on 'Storage' tab (or 'Application' in newer versions)")
    print("4. Expand 'Cookies' → click on 'https://poe.com'")
    print("5. Find and copy the values for 'p-b' and 'p-lat'")
    print("\n" + "=" * 50)
    
    # Wait for user to get tokens
    input("Press Enter when you have your tokens ready...")
    
    # Get tokens
    print("\n🔑 Enter your tokens:")
    p_b = input("p-b token: ").strip()
    p_lat = input("p-lat token: ").strip()
    
    if not p_b or not p_lat:
        print("❌ Both p-b and p-lat tokens are required!")
        return
    
    formkey = input("formkey (optional, press Enter to skip): ").strip()
    
    # Create .env file
    env_content = f"""# Required Poe tokens
POE_P_B_TOKEN={p_b}
POE_P_LAT_TOKEN={p_lat}

# Optional tokens for enhanced functionality
POE_FORMKEY={formkey}
POE_CF_BM=
POE_CF_CLEARANCE=
"""
    
    env_path = Path(__file__).parent / ".env"
    with open(env_path, "w") as f:
        f.write(env_content)
    
    print(f"\n✅ Tokens saved to {env_path}")
    print("🎉 Setup complete!")
    
    # Ask if user wants to start the server
    response = input("\nWould you like to start the server now? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        print("\n🚀 Starting server...")
        subprocess.run(["python", "start.py"])


if __name__ == "__main__":
    main() 