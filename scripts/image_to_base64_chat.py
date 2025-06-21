#!/usr/bin/env python3
"""
Convert image to base64 and send to OpenAI chat completion API.

Usage:
    python3 scripts/image_to_base64_chat.py scripts/eaa3f4c4fa1a10fb233e0e9fac9ec25ce67d77e365093671892d181d519bbf49.jpeg
    python3 scripts/image_to_base64_chat.py scripts/eaa3f4c4fa1a10fb233e0e9fac9ec25ce67d77e365093671892d181d519bbf49.jpeg --prompt "What do you see in this image?"

Requirements:
    - Set POE_API_KEY in .env file
    - Optional: Set OPENAI_COMPATIBLE_API_BASE_URL in .env (defaults to http://localhost:8080)
"""
import argparse
import base64
import os
from dotenv import load_dotenv
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument("--prompt", default="Describe this image", help="Prompt text")
    args = parser.parse_args()
    
    load_dotenv()
    
    # Read and encode image
    with open(args.image_path, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode()
    
    # Setup client
    client = OpenAI(
        api_key=os.environ["POE_API_KEY"],
        base_url=os.environ.get("OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080") + "/v1"
    )
    
    # Send request
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }]
    )
    
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()