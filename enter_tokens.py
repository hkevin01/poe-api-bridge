#!/usr/bin/env python3
"""
Enter Poe tokens manually
"""

from pathlib import Path


def main():
    print("🔑 Enter Your Poe Tokens")
    print("=" * 30)
    
    # Get tokens
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
    response = input("\nStart the server now? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        print("\n🚀 Starting server...")
        import subprocess
        subprocess.run(["python", "start.py"])


if __name__ == "__main__":
    main() 