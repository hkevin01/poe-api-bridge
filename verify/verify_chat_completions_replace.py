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
    model="Mercury-Coder-Small", # Mercury-Coder-Small uses diffusion instead of streaming https://platform.inceptionlabs.ai/docs#getting-started
    messages=[{"role": "user", "content": "Repeat this: 'Hello'"}],
    stream=True,
)

full_content_replace = ""
replace_mode = True  # This would be determined by server logic

for chunk in stream:
    choice = chunk.choices[0]
    if choice.delta and choice.delta.content is not None:
        if replace_mode:
            # Replace content on each chunk (is_replace_response behavior)
            full_content_replace = choice.delta.content or ''
            print(f"🔄 Replacing with: '{full_content_replace}'")
        else:
            # Accumulate content (normal streaming behavior)
            full_content_replace += choice.delta.content
            print(f"➕ Accumulating: '{choice.delta.content}' (total: '{full_content_replace}')")

print(f"✅ Final replace response: {full_content_replace}")
assert "Hello" in full_content_replace
