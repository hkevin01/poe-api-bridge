#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["POE_API_KEY"],
    base_url=os.environ["OPENAI_COMPATIBLE_API_BASE_URL"] + "/v1",
)

# Non-streaming
response = client.chat.completions.create(
    model="GPT-4o",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)

print(f"✅ Chat completion: {response.choices[0].message.content}")
assert response.choices[0].message.content

# Streaming
stream = client.chat.completions.create(
    model="GPT-4o",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    stream=True,
)

content = ""
for chunk in stream:
    if chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content

print(f"✅ Chat streaming: {content}")
assert content
