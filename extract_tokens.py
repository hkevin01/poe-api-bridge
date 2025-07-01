#!/usr/bin/env python3
"""
Automated Poe token extraction from Firefox
"""

import json
import os
import platform
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path


def get_firefox_profile_path():
    """Get the default Firefox profile path"""
    system = platform.system()
    
    if system == "Linux":
        # Linux Firefox profiles
        firefox_path = Path.home() / ".mozilla" / "firefox"
    elif system == "Darwin":  # macOS
        firefox_path = Path.home() / "Library" / "Application Support" / "Firefox" / "Profiles"
    elif system == "Windows":
        firefox_path = Path.home() / "AppData" / "Roaming" / "Mozilla" / "Firefox" / "Profiles"
    else:
        raise OSError(f"Unsupported operating system: {system}")
    
    if not firefox_path.exists():
        raise FileNotFoundError(f"Firefox profile directory not found: {firefox_path}")
    
    # Find the default profile (ends with .default-release or .default)
    profiles = list(firefox_path.glob("*.default*"))
    if not profiles:
        raise FileNotFoundError("No Firefox profile found")
    
    # Prefer default-release over default
    default_profile = None
    for profile in profiles:
        if profile.name.endswith(".default-release"):
            default_profile = profile
            break
    
    if not default_profile:
        default_profile = profiles[0]  # Use first available profile
    
    return default_profile


def extract_cookies_from_firefox():
    """Extract cookies from Firefox's cookies.sqlite database"""
    try:
        profile_path = get_firefox_profile_path()
        cookies_db = profile_path / "cookies.sqlite"
        
        if not cookies_db.exists():
            raise FileNotFoundError(f"Cookies database not found: {cookies_db}")
        
        # Create a temporary copy of the database (Firefox might have it locked)
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as temp_db:
            temp_db_path = temp_db.name
        
        shutil.copy2(cookies_db, temp_db_path)
        
        # Connect to the temporary database
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # Query for poe.com cookies
        cursor.execute("""
            SELECT name, value FROM moz_cookies 
            WHERE host LIKE '%poe.com' 
            AND name IN ('p-b', 'p-lat', 'formkey')
        """)
        
        cookies = {}
        for name, value in cursor.fetchall():
            cookies[name] = value
        
        conn.close()
        os.unlink(temp_db_path)  # Clean up temporary file
        
        return cookies
        
    except Exception as e:
        print(f"Error extracting cookies: {e}")
        return {}


def extract_formkey_from_firefox():
    """Try to extract formkey using Firefox's storage"""
    try:
        profile_path = get_firefox_profile_path()
        storage_db = profile_path / "storage" / "default" / "https+++poe.com" / "ls" / "data.sqlite"
        
        if not storage_db.exists():
            return None
        
        # Create a temporary copy
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as temp_db:
            temp_db_path = temp_db.name
        
        shutil.copy2(storage_db, temp_db_path)
        
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # Look for formkey in localStorage
        cursor.execute("SELECT key, value FROM data WHERE key LIKE '%formkey%'")
        
        formkey = None
        for key, value in cursor.fetchall():
            if 'formkey' in key.lower():
                try:
                    data = json.loads(value)
                    if isinstance(data, dict) and 'formkey' in data:
                        formkey = data['formkey']
                        break
                except:
                    continue
        
        conn.close()
        os.unlink(temp_db_path)
        
        return formkey
        
    except Exception as e:
        print(f"Error extracting formkey: {e}")
        return None


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
    """Main function to extract tokens and create .env file"""
    print("🔍 Automatically extracting Poe tokens from Firefox...")
    print("=" * 60)
    
    # Extract cookies
    cookies = extract_cookies_from_firefox()
    
    if not cookies:
        print("❌ No Poe cookies found in Firefox!")
        print("\nMake sure you are logged into poe.com in Firefox")
        print("Then try again.")
        return
    
    print("✅ Found cookies:")
    for name, value in cookies.items():
        if name in ['p-b', 'p-lat']:
            print(f"  {name}: {value[:20]}...{value[-10:]}")
        else:
            print(f"  {name}: {value}")
    
    # Try to get formkey
    formkey = extract_formkey_from_firefox()
    if formkey:
        cookies['formkey'] = formkey
        print(f"  formkey: {formkey[:20]}...{formkey[-10:]}")
    
    # Check if we have required tokens
    if not cookies.get('p-b') or not cookies.get('p-lat'):
        print("\n❌ Missing required tokens (p-b and p-lat)")
        print("Please make sure you are logged into poe.com in Firefox")
        return
    
    # Create .env file
    env_path = create_env_file(cookies)
    
    print(f"\n✅ Tokens saved to {env_path}")
    print("🎉 You can now start the backend server!")
    
    # Ask if user wants to start the server
    response = input("\nWould you like to start the server now? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        print("\n🚀 Starting server...")
        subprocess.run(["python", "start.py"])


if __name__ == "__main__":
    main() 