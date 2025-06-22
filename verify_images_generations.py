#!/usr/bin/env python3
"""
Verify image generation API functionality.

Usage:
    python3 verify_image_generation.py
    python3 verify_image_generation.py --prompt "A cat sitting on a table"

Requirements:
    - Set POE_API_KEY in .env file
    - Optional: Set OPENAI_COMPATIBLE_API_BASE_URL in .env (defaults to http://localhost:8080)
"""
import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="A beautiful sunset over mountains", help="Image generation prompt")
    parser.add_argument("--model", default="Imagen-3-Fast", help="Model to use for generation")
    args = parser.parse_args()
    
    load_dotenv()
    
    # Setup client
    client = OpenAI(
        api_key=os.environ["POE_API_KEY"],
        base_url=os.environ.get("OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080") + "/v1"
    )
    
    print(f"Generating image with prompt: {args.prompt}")
    print(f"Using model: {args.model}")
    
    # Test single image generation
    response = client.images.generate(
        model=args.model,
        prompt=args.prompt,
        n=1,
        response_format="url"
    )
    
    print(f"✅ Single image generation successful!")
    print(f"Generated image URL: {response.data[0].url}")
    
    # Test multiple image generation (n=3)
    print(f"\n--- Testing Multiple Images (n=3) ---")
    response_multi = client.images.generate(
        model=args.model,
        prompt=args.prompt,
        n=3,
        response_format="url"
    )
    
    print(f"✅ Multiple image generation successful!")
    for i, data in enumerate(response_multi.data):
        print(f"Generated image {i+1} URL: {data.url}")
    
    # Test with b64_json format
    response_b64 = client.images.generate(
        model=args.model,
        prompt=args.prompt,
        n=1,
        response_format="b64_json"
    )
    
    print(f"✅ Base64 image generation successful!")
    print(f"Generated image as base64 (length: {len(response_b64.data[0].b64_json)} characters)")


if __name__ == "__main__":
    main()