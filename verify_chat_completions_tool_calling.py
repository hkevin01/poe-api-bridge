import os
from dotenv import load_dotenv
from openai import OpenAI


def test_tool_calling_basic():
    """Test basic tool calling functionality"""
    load_dotenv()
    
    BASE_URL = os.environ.get("OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080")
    API_KEY = os.environ.get("POE_API_KEY")
    
    if not API_KEY:
        print("❌ Error: POE_API_KEY environment variable not set")
        return
    
    print("\n=== Testing Basic Tool Calling ===")
    print(f"Base URL: {BASE_URL}")
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=API_KEY,
        base_url=f"{BASE_URL}/v1"
    )
    
    # Define a simple tool
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
    
    try:
        print("\n--- Testing Non-Streaming Tool Call ---")
        response = client.chat.completions.create(
            model="Claude-3.5-Sonnet",
            messages=[
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ],
            tools=[weather_tool],
            tool_choice="auto"
        )
        
        print("✅ Tool calling request successful!")
        
        choice = response.choices[0]
        message = choice.message
        
        # Check for tool calls
        if message.tool_calls:
            tool_calls = message.tool_calls
            print(f"✅ Tool calls detected: {len(tool_calls)} call(s)")
            
            for i, tool_call in enumerate(tool_calls):
                print(f"  Tool call {i+1}:")
                print(f"    ID: {tool_call.id}")
                print(f"    Function: {tool_call.function.name}")
                print(f"    Arguments: {tool_call.function.arguments}")
                
                # Validate structure
                assert tool_call.type == "function", "Tool call type should be 'function'"
                assert tool_call.id, "Tool call should have an ID"
                assert tool_call.function, "Tool call should have function details"
                
            print("✅ Tool call structure validation passed!")
        else:
            print("ℹ️  No tool calls in response (model chose not to use tools)")
            if message.content:
                print("Response content:")
                print("-" * 30)
                print(message.content)
                print("-" * 30)
                
        # Check finish reason
        finish_reason = choice.finish_reason
        if finish_reason == "tool_calls":
            print("✅ Finish reason correctly set to 'tool_calls'")
        elif finish_reason == "stop":
            print("ℹ️  Finish reason is 'stop' (no tool calls made)")
            
    except Exception as e:
        print(f"❌ Error during basic tool calling test: {e}")


def test_tool_choice_required():
    """Test tool_choice='required' functionality"""
    load_dotenv()
    
    BASE_URL = os.environ.get("OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080")
    API_KEY = os.environ.get("POE_API_KEY")
    
    if not API_KEY:
        print("❌ Error: POE_API_KEY environment variable not set")
        return
    
    print("\n--- Testing tool_choice='required' ---")
    
    client = OpenAI(
        api_key=API_KEY,
        base_url=f"{BASE_URL}/v1"
    )
    
    calculator_tool = {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
    
    try:
        response = client.chat.completions.create(
            model="Claude-3.5-Sonnet",
            messages=[
                {"role": "user", "content": "Calculate 15 multiplied by 23"}
            ],
            tools=[calculator_tool],
            tool_choice="required"
        )
        
        choice = response.choices[0]
        message = choice.message
        
        # With required, should always have tool calls
        if message.tool_calls:
            print("✅ tool_choice='required' working - tool call was made")
            tool_call = message.tool_calls[0]
            print(f"  Function called: {tool_call.function.name}")
            print(f"  Arguments: {tool_call.function.arguments}")
        else:
            print("❌ tool_choice='required' failed - no tool calls made")
            assert False, "tool_choice='required' should force tool usage"
            
    except Exception as e:
        print(f"❌ Error during required tool choice test: {e}")


def test_tool_choice_none():
    """Test tool_choice='none' functionality"""
    load_dotenv()
    
    BASE_URL = os.environ.get("OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080")
    API_KEY = os.environ.get("POE_API_KEY")
    
    if not API_KEY:
        print("❌ Error: POE_API_KEY environment variable not set")
        return
    
    print("\n--- Testing tool_choice='none' ---")
    
    client = OpenAI(
        api_key=API_KEY,
        base_url=f"{BASE_URL}/v1"
    )
    
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
    
    try:
        response = client.chat.completions.create(
            model="Claude-3.5-Sonnet",
            messages=[
                {"role": "user", "content": "What's the weather in NYC?"}
            ],
            tools=[weather_tool],
            tool_choice="none"
        )
        
        choice = response.choices[0]
        message = choice.message
        
        # Should not use tools
        if not message.tool_calls:
            print("✅ tool_choice='none' working - no tool calls made")
            if message.content:
                print("Response content provided instead of tool calls")
        else:
            print("❌ tool_choice='none' failed - tool calls were made")
            assert False, "tool_choice='none' should prevent tool usage"
            
    except Exception as e:
        print(f"❌ Error during none tool choice test: {e}")


def test_streaming_tool_calls():
    """Test streaming with tool calls"""
    load_dotenv()
    
    BASE_URL = os.environ.get("OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080")
    API_KEY = os.environ.get("POE_API_KEY")
    
    if not API_KEY:
        print("❌ Error: POE_API_KEY environment variable not set")
        return
    
    print("\n--- Testing Streaming Tool Calls ---")
    
    client = OpenAI(
        api_key=API_KEY,
        base_url=f"{BASE_URL}/v1"
    )
    
    search_tool = {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
    
    try:
        stream = client.chat.completions.create(
            model="Claude-3.5-Sonnet",
            messages=[
                {"role": "user", "content": "Search for information about Python programming"}
            ],
            tools=[search_tool],
            tool_choice="auto",
            stream=True
        )
        
        print("Processing streaming response...")
        
        chunk_count = 0
        tool_calls_found = False
        
        for chunk in stream:
            chunk_count += 1
            if chunk.choices[0].delta.tool_calls:
                tool_calls_found = True
                print("✅ Tool calls detected in streaming response")
        
        print(f"✅ Streaming completed - received {chunk_count} chunks")
        if tool_calls_found:
            print("✅ Tool calls successfully streamed")
        else:
            print("ℹ️  No tool calls in streaming response")
            
    except Exception as e:
        print(f"❌ Error during streaming tool calls test: {e}")


def test_no_tools_fallback():
    """Test that requests without tools work normally (fallback)"""
    load_dotenv()
    
    BASE_URL = os.environ.get("OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080")
    API_KEY = os.environ.get("POE_API_KEY")
    
    if not API_KEY:
        print("❌ Error: POE_API_KEY environment variable not set")
        return
    
    print("\n--- Testing No Tools Fallback ---")
    
    client = OpenAI(
        api_key=API_KEY,
        base_url=f"{BASE_URL}/v1"
    )
    
    try:
        response = client.chat.completions.create(
            model="Claude-3.5-Sonnet",
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ]
            # No tools provided
        )
        
        choice = response.choices[0]
        message = choice.message
        
        # Should work normally without tools
        if message.content and not message.tool_calls:
            print("✅ No tools fallback working - normal response provided")
            print("Response content preview:")
            content_preview = message.content[:100] + "..." if len(message.content) > 100 else message.content
            print(content_preview)
        else:
            print("❌ No tools fallback failed")
            
    except Exception as e:
        print(f"❌ Error during no tools fallback test: {e}")


if __name__ == "__main__":
    print("🚀 Starting Fake Tool Calling Verification")
    
    # Test basic tool calling
    test_tool_calling_basic()
    
    # Test tool_choice options
    test_tool_choice_required()
    test_tool_choice_none()
    
    # Test streaming
    test_streaming_tool_calls()
    
    # Test fallback
    test_no_tools_fallback()
    
    print("\n🏁 Tool calling verification complete!")