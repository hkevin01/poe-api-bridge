#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["POE_API_KEY"],
    base_url=os.environ["OPENAI_COMPATIBLE_API_BASE_URL"] + "/v1",
)

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather information for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state"},
            },
            "required": ["location"],
        },
    },
}

# Basic tool calling
response = client.chat.completions.create(
    model="Claude-3.5-Sonnet",
    messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
    tools=[weather_tool],
    tool_choice="auto",
)

print(f"✅ Tool calling: {len(response.choices[0].message.tool_calls or [])} calls")
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Function: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")

# Tool choice required
response = client.chat.completions.create(
    model="Claude-3.5-Sonnet",
    messages=[{"role": "user", "content": "Calculate 15 multiplied by 23"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }],
    tool_choice="required",
)

print(f"✅ Tool required: {response.choices[0].message.tool_calls is not None}")
assert response.choices[0].message.tool_calls

# Tool choice none
response = client.chat.completions.create(
    model="Claude-3.5-Sonnet",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=[weather_tool],
    tool_choice="none",
)

print(f"✅ Tool none: {response.choices[0].message.tool_calls is None}")
assert not response.choices[0].message.tool_calls
