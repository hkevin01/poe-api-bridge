import requests
import json
import os
from dotenv import load_dotenv
import httpx


def test_function_calling():
    # Load environment variables from .env file
    load_dotenv()
    
    BASE_URL = "http://localhost:80"
    API_KEY = os.environ["POE_API_KEY"]
    
    if not API_KEY:
        print("❌ Error: POE_API_KEY environment variable not set")
        return
    
    # Simple weather function definition
    weather_function = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state/country"
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

    # Test data with a simple weather query
    test_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. When asked about weather, use the get_weather function."
            },
            {
                "role": "user",
                "content": "What's the weather like in San Francisco and New York? Compare the temperatures."
            }
        ],
        "stream": False,
        "temperature": 0.7,
        "tools": [{
            "type": "function",
            "function": weather_function["function"]
        }],
        "tool_choice": "auto"  # Let the model decide when to use the function
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        # Test non-streaming call
        print("\n=== Testing Weather Function Call ===")
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=test_data
        )
    
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nResponse:")
            print(json.dumps(result, indent=2))
            
            # Check for tool calls
            message = result["choices"][0]["message"]
            if "tool_calls" in message:
                tool_calls = message["tool_calls"]
                print(f"\n✅ Successfully generated {len(tool_calls)} tool call(s)!")
                
                # Print each tool call
                for i, call in enumerate(tool_calls, 1):
                    print(f"\nTool Call {i}:")
                    print(json.dumps({
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"]
                    }, indent=2))
            else:
                print("\n❌ No tool calls found in response")
        else:
            print("\n❌ Request failed:")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

    # Test with forced multiple function calls
    print("\n=== Testing Multiple Tool Calls ===")
    test_data["messages"] = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. When asked about weather, use the get_weather function for EACH city mentioned."
        },
        {
            "role": "user",
            "content": "Compare the weather between Tokyo, London, and Paris."
        }
    ]

    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=test_data
        )

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            message = result["choices"][0]["message"]
            if "tool_calls" in message:
                print(f"\n✅ Generated {len(message['tool_calls'])} tool calls!")
                print(json.dumps(message["tool_calls"], indent=2))
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

    # Test tool call response handling
    print("\n=== Testing Tool Call Response Flow ===")
    
    # Initial request
    test_data["messages"] = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. When asked about weather, use the get_weather function for each city mentioned."
        },
        {
            "role": "user",
            "content": "Compare the weather between New York and Tokyo right now."
        }
    ]

    try:
        # Step 1: Make initial request
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=test_data
        )
        
        if response.status_code == 200:
            result = response.json()
            message = result["choices"][0]["message"]
            
            if "tool_calls" in message:
                tool_calls = message["tool_calls"]
                print("\n✅ Initial tool call successful")

                # Step 2: Add tool results for each city
                tool_results = []
                for tool_call in tool_calls:
                    args = tool_call["function"]["arguments"]
                    city = args["location"] if isinstance(args, dict) else json.loads(args)["location"]
                    # Generate different weather data for each city
                    weather_data = {
                        "New York": {
                            "temperature": 5,
                            "condition": "snowy",
                            "humidity": 85
                        },
                        "Tokyo": {
                            "temperature": 15,
                            "condition": "sunny",
                            "humidity": 60
                        }
                    }.get(city, {
                        "temperature": 20,
                        "condition": "cloudy",
                        "humidity": 70,
                    })

                    tool_results.append({
                        "role": "user",
                        "content": f"""Tool call result for {tool_call["function"]["name"]} (ID: {tool_call["id"]}):
```json
{{
    "tool_call_id": "{tool_call["id"]}",
    "name": "{tool_call["function"]["name"]}",
    "result": {{
            "location": "{tool_call["function"]["arguments"]["location"]}",
            "time": "2024-02-21T14:21:00Z",
                            "temperature": {str(weather_data['temperature'])},
                            "condition": "{str(weather_data['condition'])}",
                            "humidity": {str(weather_data['humidity'])}
    }}
}}
```\n"""
                    })
                
                # Make follow-up request with tool results
                follow_up_data = {
                    **test_data,
                    "messages": [
                        *test_data["messages"],
                        message,  # Assistant's message with tool calls
                        *tool_results  # Tool response messages
                    ]
                }
                
                print("\nSending tool results...")
                response = requests.post(                    
                    f"{BASE_URL}/v1/chat/completions",
                    headers=headers, timeout=30,
                    json=follow_up_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "error" in result:
                        print(f"\nRequest data: {json.dumps(follow_up_data, indent=2)}")
                        print("\n❌ Error in follow-up response:")
                        print(json.dumps(result["error"], indent=2))
                        return
                    print("\nFinal Response:")
                    print(json.dumps(result["choices"][0]["message"], indent=2))
                    
    except Exception as e:
        print(f"\nRequest data: {json.dumps(test_data, indent=2)}")
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
# Set up basic logging for requests
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("weather_test")
    
    try:
        test_function_calling()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")