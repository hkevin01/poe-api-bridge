#!/usr/bin/env python3
"""
Run Poe API wrapper integration tests
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path


def check_server_running():
    """Check if the server is running on localhost:8000"""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def start_server():
    """Start the server in the background"""
    print("🚀 Starting server...")
    try:
        # Start server in background
        process = subprocess.Popen(
            [sys.executable, "server.py"],
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        if process.poll() is None:  # Still running
            print("✅ Server started successfully")
            return process
        else:
            print("❌ Server failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

def stop_server(process):
    """Stop the server process"""
    if process:
        print("🛑 Stopping server...")
        process.terminate()
        process.wait()
        print("✅ Server stopped")

async def main():
    """Main test runner"""
    print("🧪 Poe API Wrapper Test Suite Runner")
    print("=" * 50)
    
    # Check if server is already running
    if check_server_running():
        print("✅ Server is already running")
        server_process = None
    else:
        print("⚠️  Server not running, starting it...")
        server_process = start_server()
        if not server_process:
            print("❌ Cannot run tests without server")
            sys.exit(1)
    
    try:
        # Import and run the test suite
        from test_poe_wrapper_integration import PoeWrapperTestSuite
        
        test_suite = PoeWrapperTestSuite()
        passed, failed, total = await test_suite.run_all_tests()
        
        print(f"\n🎯 Final Result: {passed}/{total} tests passed")
        
        if failed > 0:
            print("❌ Some tests failed - check the output above")
            sys.exit(1)
        else:
            print("✅ All tests passed!")
            sys.exit(0)
            
    except ImportError as e:
        print(f"❌ Failed to import test suite: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements-prod.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test suite crashed: {e}")
        sys.exit(1)
    finally:
        # Stop server if we started it
        if server_process:
            stop_server(server_process)

if __name__ == "__main__":
    asyncio.run(main()) 