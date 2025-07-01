#!/usr/bin/env python3
"""
Startup script for Poe Code Chat Backend
"""

import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


def main():
    # Load environment variables
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from {env_path}")
    else:
        print("⚠️  No .env file found. Please run setup_tokens.py first.")
        sys.exit(1)
    
    # Check required tokens
    if not os.getenv('POE_P_B_TOKEN') or not os.getenv('POE_P_LAT_TOKEN'):
        print("❌ Required Poe tokens not found in environment!")
        print("Please run: python setup_tokens.py")
        sys.exit(1)
    
    print("🚀 Starting Poe Code Chat Backend...")
    print("📡 Server will be available at: http://127.0.0.1:8000")
    print("📖 API docs will be available at: http://127.0.0.1:8000/docs")
    
    # Start the server
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)],
        loop="asyncio"
    )


if __name__ == "__main__":
    main() 