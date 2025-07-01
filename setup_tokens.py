#!/usr/bin/env python3
"""
Script to help set up Poe API tokens
Follow the instructions in the poe-api-wrapper documentation to get these tokens
"""

from pathlib import Path


def setup_tokens():
    """Interactive script to set up Poe tokens"""
    print("🚀 Poe API Token Setup")
    print("=" * 50)
    print("\nTo get your tokens, follow these steps:")
    print("1. Go to https://poe.com/")
    print("2. Sign in to your account")
    print("3. Press F12 to open Developer Tools")
    print("4. Go to Application > Cookies > poe.com")
    print("5. Copy the values for 'p-b' and 'p-lat' cookies")
    print("\nFor optional formkey:")
    print("6. Go to Network tab > Look for gql_POST > Headers > Poe-Formkey")
    print("   OR in Console tab, type: window.ereNdsRqhp2Rd3LEW()")
    
    print("\n" + "=" * 50)
    
    # Get required tokens
    p_b = input("Enter your p-b token: ").strip()
    p_lat = input("Enter your p-lat token: ").strip()
    
    if not p_b or not p_lat:
        print("❌ p-b and p-lat tokens are required!")
        return
    
    # Get optional tokens
    formkey = input("Enter your formkey (optional, press Enter to skip): ").strip()
    cf_bm = input("Enter your __cf_bm token (optional, press Enter to skip): ").strip()
    cf_clearance = input(
        "Enter your cf_clearance token (optional, press Enter to skip): "
    ).strip()
    
    # Create .env file
    env_content = f"""# Required Poe tokens
POE_P_B_TOKEN={p_b}
POE_P_LAT_TOKEN={p_lat}

# Optional tokens for enhanced functionality
POE_FORMKEY={formkey}
POE_CF_BM={cf_bm}
POE_CF_CLEARANCE={cf_clearance}
"""
    
    env_path = Path(__file__).parent / ".env"
    with open(env_path, "w") as f:
        f.write(env_content)
    
    print(f"\n✅ Tokens saved to {env_path}")
    print("🎉 You can now start the backend server!")


if __name__ == "__main__":
    setup_tokens() 