#!/usr/bin/env python3
"""
Generate images using the Poe API bridge.

Usage:
    python3 scripts/generate_image.py "A cat sitting on a table"
    python3 scripts/generate_image.py "A sunset" --model Imagen-3-Fast --format b64_json

Requirements:
    - Set POE_API_KEY in .env file
    - Optional: Set OPENAI_COMPATIBLE_API_BASE_URL in .env (defaults to http://localhost:8080)
"""
import argparse
import os
import base64
from dotenv import load_dotenv
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Image generation prompt")
    parser.add_argument(
        "--model", default="Imagen-3-Fast", help="Model to use for generation"
    )
    parser.add_argument(
        "--format", choices=["url", "b64_json"], default="url", help="Response format"
    )
    parser.add_argument(
        "--output", help="Save base64 image to file (only for b64_json format)"
    )
    args = parser.parse_args()

    load_dotenv()

    # Setup client
    client = OpenAI(
        api_key=os.environ["POE_API_KEY"],
        base_url=os.environ.get(
            "OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080"
        )
        + "/v1",
    )

    print(f"Generating image: {args.prompt}")
    print(f"Model: {args.model}")
    print(f"Format: {args.format}")

    # Generate image
    response = client.images.generate(
        model=args.model, prompt=args.prompt, n=1, response_format=args.format
    )

    if args.format == "url":
        print(f"Image URL: {response.data[0].url}")
    else:
        print(
            f"Base64 image generated (length: {len(response.data[0].b64_json)} characters)"
        )

        if args.output:
            # Decode and save image
            image_data = base64.b64decode(response.data[0].b64_json)
            with open(args.output, "wb") as f:
                f.write(image_data)
            print(f"Image saved to: {args.output}")


if __name__ == "__main__":
    main()
