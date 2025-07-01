#!/usr/bin/env python3
"""
Quick setup script for Poe tokens
"""

import subprocess
import webbrowser
from pathlib import Path


def main():
    print("🚀 Quick Poe Token Setup")
    print("=" * 50)
    
    # Open poe.com in default browser
    print("📱 Opening poe.com in your browser...")
    webbrowser.open("https://poe.com")
    
    print("\n📋 Instructions:")
    print("1. Sign in to your Poe account (or create one)")
    print("2. Once logged in, press F12 to open Developer Tools")
    print("3. Go to 'Application' tab (or 'Storage' in some browsers)")
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