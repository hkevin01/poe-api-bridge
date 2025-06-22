#!/usr/bin/env python3
"""
Verify chat completions attachment handling functionality.

This test verifies that when a bot returns files via message.attachment,
they are properly handled and included in the response.

Usage:
    python3 verify_chat_completions_attachments.py

Requirements:
    - Set POE_API_KEY in .env file
    - Optional: Set OPENAI_COMPATIBLE_API_BASE_URL in .env (defaults to http://localhost:8080)
"""
import os
from dotenv import load_dotenv
from openai import OpenAI


def main():
    load_dotenv()

    # Setup client
    client = OpenAI(
        api_key=os.environ["POE_API_KEY"],
        base_url=os.environ.get(
            "OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080"
        )
        + "/v1",
    )

    print("=== Testing Chat Completions with Bot Attachments ===")
    print("Testing whether bots can return files via message.attachment...")

    # Test with image generation bot via chat completions
    print("\n--- Test 1: Image Generation via Chat Completions (Imagen-3-Fast) ---")
    response = client.chat.completions.create(
        model="Imagen-3-Fast",
        messages=[
            {
                "role": "user",
                "content": "Generate an image of a red cat sitting on a blue chair",
            }
        ],
        stream=False,
    )

    content = response.choices[0].message.content
    print(f"✅ Chat completion successful!")
    print(f"Response content: {content}")

    # Check if attachment URL is included in content
    if "https://pfst.cf2.poecdn.net" in content or (
        "http" in content
        and ("jpg" in content or "png" in content or "webp" in content)
    ):
        print(
            "✅ Response appears to contain an attachment URL - message.attachment handling working!"
        )
    else:
        print("❌ Response doesn't appear to contain an attachment URL")
        print(f"   Content: {content}")

    # Test with streaming
    print("\n--- Test 2: Streaming Chat Completions with Attachments ---")
    stream = client.chat.completions.create(
        model="Imagen-3-Fast",
        messages=[{"role": "user", "content": "Create a simple drawing of a house"}],
        stream=True,
    )

    full_content = ""
    print("Streaming response:")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content_piece = chunk.choices[0].delta.content
            print(content_piece, end="", flush=True)
            full_content += content_piece

    print(f"\n✅ Streaming chat completion successful!")

    # Check streaming attachment handling
    if "https://pfst.cf2.poecdn.net" in full_content or (
        "http" in full_content
        and ("jpg" in full_content or "png" in full_content or "webp" in full_content)
    ):
        print(
            "✅ Streaming response contains attachment URL - message.attachment streaming working!"
        )
    else:
        print("❌ Streaming response doesn't appear to contain an attachment URL")
        print(f"   Content: {full_content}")


if __name__ == "__main__":
    main()
