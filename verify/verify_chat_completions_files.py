#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["POE_API_KEY"],
    base_url=os.environ["OPENAI_COMPATIBLE_API_BASE_URL"] + "/v1",
)

image_url = "https://qph.cf2.quoracdn.net/main-qimg-3dd2a86604805445a043be92da5f87aa"

# Non-streaming
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ],
    stream=False,
)

content = response.choices[0].message.content
print(f"✅ File support: {content}")
assert "quora" in content.lower()

# Streaming
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ],
    stream=True,
)

full_content = ""
for chunk in stream:
    if chunk.choices[0].delta.content:
        full_content += chunk.choices[0].delta.content

print(f"✅ File streaming: {full_content}")
assert "quora" in full_content.lower()
