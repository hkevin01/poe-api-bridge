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
    model="Imagen-3-Fast",
    messages=[{"role": "user", "content": "Generate an image of a red cat sitting on a blue chair"}],
    stream=False,
)

content = response.choices[0].message.content
print(f"✅ Chat attachment: {content}")
assert "http" in content

# Streaming
stream = client.chat.completions.create(
    model="Imagen-3-Fast",
    messages=[{"role": "user", "content": "Create a simple drawing of a house"}],
    stream=True,
)

full_content = ""
for chunk in stream:
    if chunk.choices[0].delta.content:
        full_content += chunk.choices[0].delta.content

print(f"✅ Chat streaming attachment: {full_content}")
assert "http" in full_content
