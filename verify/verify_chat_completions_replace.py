#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["POE_API_KEY"],
    base_url=os.environ["OPENAI_COMPATIBLE_API_BASE_URL"] + "/v1",
)

stream = client.chat.completions.create(
    model="BetterDevBot",
    messages=[{"role": "user", "content": "openai_test_case_replace_response"}],
    stream=True,
)

full_content = ""
for chunk in stream:
    if chunk.choices[0].delta.content:
        full_content += chunk.choices[0].delta.content

print(f"✅ Replace response: {full_content}")
assert "Final Response" in full_content
