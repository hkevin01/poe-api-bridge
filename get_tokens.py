#!/usr/bin/env python3
"""
Flexible Poe token extraction from various browsers
"""

import glob
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path


def find_browser_profiles():
    """Find browser profiles from various browsers"""
    profiles = {}
    
    # Firefox profiles
    firefox_paths = [
        Path.home() / ".mozilla" / "firefox",
        Path.home() / "Library" / "Application Support" / "Firefox" / "Profiles",
        Path.home() / "AppData" / "Roaming" / "Mozilla" / "Firefox" / "Profiles"
    ]
    
    for firefox_path in firefox_paths:
        if firefox_path.exists():
            for profile in firefox_path.glob("*.default*"):
                profiles[f"firefox_{profile.name}"] = profile
            break
    
    # Chrome profiles
    chrome_paths = [
        Path.home() / ".config" / "google-chrome" / "Default",
        Path.home() / "Library" / "Application Support" / "Google" / "Chrome" / "Default",
        Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data" / "Default"
    ]
    
    for chrome_path in chrome_paths:
        if chrome_path.exists():
            profiles["chrome_default"] = chrome_path
            break
    
    return profiles


def extract_cookies_from_browser(profile_path, browser_type):
    """Extract cookies from browser database"""
    try:
        if browser_type.startswith("firefox"):
            cookies_db = profile_path / "cookies.sqlite"
            if not cookies_db.exists():
                return {}
            
            # Create temporary copy
            with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as temp_db:
                temp_db_path = temp_db.name
            
            shutil.copy2(cookies_db, temp_db_path)
            
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name, value FROM moz_cookies 
                WHERE host LIKE '%poe.com' 
                AND name IN ('p-b', 'p-lat', 'formkey')
            """)
            
            cookies = {}
            for name, value in cursor.fetchall():
                cookies[name] = value
            
            conn.close()
            os.unlink(temp_db_path)
            
            return cookies
            
        elif browser_type == "chrome_default":
            cookies_db = profile_path / "Cookies"
            if not cookies_db.exists():
                return {}
            
            # Create temporary copy
            with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as temp_db:
                temp_db_path = temp_db.name
            
            shutil.copy2(cookies_db, temp_db_path)
            
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name, value FROM cookies 
                WHERE host_key LIKE '%poe.com' 
                AND name IN ('p-b', 'p-lat', 'formkey')
            """)
            
            cookies = {}
            for name, value in cursor.fetchall():
                cookies[name] = value
            
            conn.close()
            os.unlink(temp_db_path)
            
            return cookies
            
    except Exception as e:
        print(f"Error extracting from {browser_type}: {e}")
        return {}


def manual_token_input():
    """Manual token input fallback"""
    print("\n" + "="*60)
    print("🔧 Manual Token Input")
    print("="*60)
    print("\nPlease get your tokens from poe.com:")
    print("1. Go to https://poe.com/ and sign in")
    print("2. Press F12 to open Developer Tools")
    print("3. Go to Application/Storage > Cookies > poe.com")
    print("4. Copy the values for 'p-b' and 'p-lat'")
    print("\n" + "="*60)
    
    p_b = input("Enter your p-b token: ").strip()
    p_lat = input("Enter your p-lat token: ").strip()
    
    if not p_b or not p_lat:
        print("❌ p-b and p-lat tokens are required!")
        return {}
    
    formkey = input("Enter your formkey (optional, press Enter to skip): ").strip()
    
    return {
        'p-b': p_b,
        'p-lat': p_lat,
        'formkey': formkey
    }


def create_env_file(tokens):
    """Create .env file with the extracted tokens"""
    env_content = f"""# Required Poe tokens
POE_P_B_TOKEN={tokens.get('p-b', '')}
POE_P_LAT_TOKEN={tokens.get('p-lat', '')}

# Optional tokens for enhanced functionality
POE_FORMKEY={tokens.get('formkey', '')}
POE_CF_BM={tokens.get('__cf_bm', '')}
POE_CF_CLEARANCE={tokens.get('cf_clearance', '')}
"""
    
    env_path = Path(__file__).parent / ".env"
    with open(env_path, "w") as f:
        f.write(env_content)
    
    return env_path


def main():
    """Main function to extract tokens"""
    print("🔍 Automatically extracting Poe tokens from browsers...")
    print("=" * 60)
    
    # Find browser profiles
    profiles = find_browser_profiles()
    
    if not profiles:
        print("❌ No browser profiles found!")
        print("Falling back to manual input...")
        tokens = manual_token_input()
    else:
        print(f"✅ Found {len(profiles)} browser profile(s):")
        for name, path in profiles.items():
            print(f"  - {name}: {path}")
        
        # Try to extract from each browser
        tokens = {}
        for browser_name, profile_path in profiles.items():
            print(f"\n🔍 Trying {browser_name}...")
            browser_tokens = extract_cookies_from_browser(profile_path, browser_name)
            
            if browser_tokens:
                print(f"✅ Found tokens in {browser_name}:")
                for name, value in browser_tokens.items():
                    if name in ['p-b', 'p-lat']:
                        print(f"  {name}: {value[:20]}...{value[-10:]}")
                    else:
                        print(f"  {name}: {value}")
                tokens.update(browser_tokens)
                break
            else:
                print(f"❌ No Poe tokens found in {browser_name}")
        
        # If no tokens found, fall back to manual input
        if not tokens:
            print("\n❌ No tokens found in any browser!")
            tokens = manual_token_input()
    
    # Check if we have required tokens
    if not tokens.get('p-b') or not tokens.get('p-lat'):
        print("\n❌ Missing required tokens (p-b and p-lat)")
        print("Please make sure you are logged into poe.com")
        return
    
    # Create .env file
    env_path = create_env_file(tokens)
    
    print(f"\n✅ Tokens saved to {env_path}")
    print("🎉 You can now start the backend server!")
    
    # Ask if user wants to start the server
    response = input("\nWould you like to start the server now? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        print("\n🚀 Starting server...")
        subprocess.run(["python", "start.py"])


if __name__ == "__main__":
    main() 