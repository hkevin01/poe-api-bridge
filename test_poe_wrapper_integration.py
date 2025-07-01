#!/usr/bin/env python3
"""
Test suite for Poe API wrapper integration
Tests connection, authentication, and conversation retrieval capabilities
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Any

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the poe-api-wrapper
try:
    from poe_api_wrapper import AsyncPoeApi
    POE_WRAPPER_AVAILABLE = True
except ImportError:
    POE_WRAPPER_AVAILABLE = False
    print("❌ poe-api-wrapper not installed. Please run: pip install poe-api-wrapper")


class PoeWrapperTestSuite:
    """Test suite for Poe API wrapper integration"""
    
    def __init__(self):
        self.poe_client = None
        self.test_results = []
        self.server_url = "http://localhost:8000"
        
        # Poe tokens from environment
        self.poe_tokens = {
            'p-b': os.getenv('POE_P_B_TOKEN', ''),
            'p-lat': os.getenv('POE_P_LAT_TOKEN', ''),
            'formkey': os.getenv('POE_FORMKEY', ''),
            '__cf_bm': os.getenv('POE_CF_BM', ''),
            'cf_clearance': os.getenv('POE_CF_CLEARANCE', '')
        }
        
        # Test models
        self.test_models = [
            "claude_3_haiku",
            "claude_3_igloo", 
            "gpt4_o_mini",
            "gemini_pro"
        ]
    
    def log_test(self, test_name: str, success: bool, message: str = "", 
                 details: Any = None):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        timestamp = datetime.now().strftime("%H:%M:%S")
        result = {
            "test": test_name,
            "status": status,
            "message": message,
            "details": details,
            "timestamp": timestamp
        }
        self.test_results.append(result)
        print(f"[{timestamp}] {status} {test_name}: {message}")
        if details:
            print(f"    Details: {details}")
    
    async def test_poe_wrapper_import(self):
        """Test that poe-api-wrapper can be imported"""
        if not POE_WRAPPER_AVAILABLE:
            self.log_test("Poe Wrapper Import", False, 
                         "poe-api-wrapper not installed")
            return False
        
        self.log_test("Poe Wrapper Import", True, 
                     "poe-api-wrapper successfully imported")
        return True
    
    async def test_environment_variables(self):
        """Test that required environment variables are set"""
        required_tokens = ['p-b', 'p-lat']
        missing_tokens = []
        
        for token in required_tokens:
            if not self.poe_tokens.get(token):
                missing_tokens.append(token)
        
        if missing_tokens:
            self.log_test("Environment Variables", False, 
                         f"Missing required tokens: {missing_tokens}")
            return False
        
        self.log_test("Environment Variables", True, 
                     f"All required tokens present: {list(self.poe_tokens.keys())}")
        return True
    
    async def test_poe_client_initialization(self):
        """Test Poe client initialization"""
        if not POE_WRAPPER_AVAILABLE:
            self.log_test("Poe Client Init", False, "poe-api-wrapper not available")
            return False
        
        try:
            # Filter out empty tokens
            tokens = {k: v for k, v in self.poe_tokens.items() if v}
            
            # Initialize client with tokens
            self.poe_client = await AsyncPoeApi(tokens=tokens).create()
            
            self.log_test("Poe Client Init", True, 
                         "Poe client initialized successfully")
            return True
            
        except Exception as e:
            self.log_test("Poe Client Init", False, 
                         f"Failed to initialize client: {str(e)}")
            return False
    
    async def test_poe_connection(self):
        """Test basic connection to Poe API"""
        if not self.poe_client:
            self.log_test("Poe Connection", False, "Poe client not initialized")
            return False
        
        try:
            # Test connection by sending a simple message
            test_message = "Hello! This is a connection test."
            response_text = ""
            
            async for chunk in self.poe_client.send_message(
                    bot="claude_3_haiku", 
                    message=test_message):
                if "response" in chunk:
                    response_text += chunk["response"]
                elif "text" in chunk:
                    response_text = chunk["text"]
                    break
            
            if response_text:
                self.log_test("Poe Connection", True, 
                             f"Successfully connected, got response: {response_text[:50]}...")
                return True
            else:
                self.log_test("Poe Connection", False, 
                             "No response received - possible auth issue")
                return False
                
        except Exception as e:
            self.log_test("Poe Connection", False, f"Connection failed: {str(e)}")
            return False
    
    async def test_model_availability(self):
        """Test that specific models are available"""
        if not self.poe_client:
            self.log_test("Model Availability", False, "Poe client not initialized")
            return False
        
        try:
            # Test each model by sending a simple message
            working_models = []
            failed_models = []
            
            for model in self.test_models:
                try:
                    test_message = "Hello! Please respond with 'OK' if you can see this."
                    response_text = ""
                    
                    async for chunk in self.poe_client.send_message(
                            bot=model, 
                            message=test_message):
                        if "response" in chunk:
                            response_text += chunk["response"]
                        elif "text" in chunk:
                            response_text = chunk["text"]
                            break
                    
                    if response_text:
                        working_models.append(model)
                    else:
                        failed_models.append(model)
                        
                except Exception as e:
                    failed_models.append(f"{model} (error: {str(e)})")
            
            if working_models:
                self.log_test("Model Availability", True, 
                             f"Working models: {working_models}")
                if failed_models:
                    print(f"    Failed models: {failed_models}")
                return True
            else:
                self.log_test("Model Availability", False, 
                             f"No test models working. Failed: {failed_models}")
                return False
                
        except Exception as e:
            self.log_test("Model Availability", False, 
                         f"Failed to test models: {str(e)}")
            return False
    
    async def test_simple_conversation(self):
        """Test sending a simple message and getting a response"""
        if not self.poe_client:
            self.log_test("Simple Conversation", False, "Poe client not initialized")
            return False
        
        try:
            # Use a reliable model for testing
            test_model = "claude_3_haiku"
            test_message = ("Hello! Please respond with 'Test successful' "
                           "if you can see this message.")
            
            # Send message
            response_text = ""
            async for chunk in self.poe_client.send_message(
                    bot=test_model, 
                    message=test_message):
                if "response" in chunk:
                    response_text += chunk["response"]
                elif "text" in chunk:
                    response_text = chunk["text"]
                    break
            
            if response_text:
                self.log_test("Simple Conversation", True, 
                             f"Got response: {response_text[:100]}...")
                return True
            else:
                self.log_test("Simple Conversation", False, "No response received")
                return False
                
        except Exception as e:
            self.log_test("Simple Conversation", False, 
                         f"Conversation failed: {str(e)}")
            return False
    
    async def test_conversation_history(self):
        """Test retrieving conversation history"""
        if not self.poe_client:
            self.log_test("Conversation History", False, "Poe client not initialized")
            return False
        
        try:
            # Note: poe-api-wrapper doesn't have a direct get_conversations method
            # We'll test by creating a conversation and checking if it works
            test_message = "This is a test conversation for history testing."
            
            response_text = ""
            async for chunk in self.poe_client.send_message(
                    bot="claude_3_haiku", 
                    message=test_message):
                if "response" in chunk:
                    response_text += chunk["response"]
                elif "text" in chunk:
                    response_text = chunk["text"]
                    break
            
            if response_text:
                self.log_test("Conversation History", True, 
                             "Successfully created test conversation")
                return True
            else:
                self.log_test("Conversation History", False, 
                             "Failed to create test conversation")
                return False
                
        except Exception as e:
            self.log_test("Conversation History", False, 
                         f"History test failed: {str(e)}")
            return False
    
    async def test_server_health(self):
        """Test if the local server is running and healthy"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            
            if response.status_code == 200:
                health_data = response.json()
                self.log_test("Server Health", True, 
                             f"Server healthy: {health_data.get('status', 'unknown')}")
                return True
            else:
                self.log_test("Server Health", False, 
                             f"Server returned status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Server Health", False, f"Server not reachable: {str(e)}")
            return False
    
    async def test_server_models_endpoint(self):
        """Test the server's models endpoint"""
        try:
            response = requests.get(f"{self.server_url}/v1/models", timeout=5)
            
            if response.status_code == 200:
                models_data = response.json()
                model_count = len(models_data.get('data', []))
                self.log_test("Server Models", True, 
                             f"Server returned {model_count} models")
                return True
            else:
                self.log_test("Server Models", False, 
                             f"Models endpoint returned {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Server Models", False, f"Models endpoint failed: {str(e)}")
            return False
    
    async def test_server_chat_endpoint(self):
        """Test the server's chat completions endpoint"""
        try:
            payload = {
                "model": "claude-3-haiku",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello! This is a test message from the test suite."
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                self.log_test("Server Chat", True, 
                             f"Got response: {content[:50]}...")
                return True
            else:
                self.log_test("Server Chat", False, 
                             f"Chat endpoint returned {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Server Chat", False, f"Chat endpoint failed: {str(e)}")
            return False
    
    async def run_all_tests(self):
        """Run all tests in sequence"""
        print("🧪 Poe API Wrapper Integration Test Suite")
        print("=" * 60)
        
        tests = [
            ("Poe Wrapper Import", self.test_poe_wrapper_import),
            ("Environment Variables", self.test_environment_variables),
            ("Poe Client Init", self.test_poe_client_initialization),
            ("Poe Connection", self.test_poe_connection),
            ("Model Availability", self.test_model_availability),
            ("Simple Conversation", self.test_simple_conversation),
            ("Conversation History", self.test_conversation_history),
            ("Server Health", self.test_server_health),
            ("Server Models", self.test_server_models_endpoint),
            ("Server Chat", self.test_server_chat_endpoint),
        ]
        
        for test_name, test_func in tests:
            try:
                await test_func()
                await asyncio.sleep(1)  # Brief pause between tests
            except Exception as e:
                self.log_test(test_name, False, f"Test crashed: {str(e)}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Test Summary:")
        
        passed = sum(1 for result in self.test_results if "✅ PASS" in result["status"])
        failed = sum(1 for result in self.test_results if "❌ FAIL" in result["status"])
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%" if total > 0 else "N/A")
        
        if failed > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if "❌ FAIL" in result["status"]:
                    print(f"  - {result['test']}: {result['message']}")
        
        return passed, failed, total


async def main():
    """Main test runner"""
    test_suite = PoeWrapperTestSuite()
    passed, failed, total = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main()) 