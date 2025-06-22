#!/usr/bin/env python3
"""
Verify image editing API functionality.

Usage:
    python3 verify_image_edit.py
    python3 verify_image_edit.py --prompt "Add a blue sky background"
    python3 verify_image_edit.py --image scripts/eaa3f4c4fa1a10fb233e0e9fac9ec25ce67d77e365093671892d181d519bbf49.jpeg

Requirements:
    - Set POE_API_KEY in .env file
    - Optional: Set OPENAI_COMPATIBLE_API_BASE_URL in .env (defaults to http://localhost:8080)
"""
import argparse
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI


def test_image_edit_openai_client():
    """Test image editing using OpenAI client"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="Make the image more colorful and vibrant",
        help="Image edit prompt",
    )
    parser.add_argument(
        "--image",
        default="scripts/eaa3f4c4fa1a10fb233e0e9fac9ec25ce67d77e365093671892d181d519bbf49.jpeg",
        help="Path to image file",
    )
    parser.add_argument(
        "--model", default="StableDiffusionXL", help="Model to use for editing"
    )
    args = parser.parse_args()

    load_dotenv()

    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        return

    # Setup client
    client = OpenAI(
        api_key=os.environ["POE_API_KEY"],
        base_url=os.environ.get(
            "OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080"
        )
        + "/v1",
    )

    print(f"Editing image: {args.image}")
    print(f"Using prompt: {args.prompt}")
    print(f"Using model: {args.model}")

    try:
        # Test image editing with URL response format
        with open(args.image, "rb") as image_file:
            response = client.images.edit(
                model=args.model,
                image=image_file,
                prompt=args.prompt,
                n=1,
                response_format="url",
            )

        print(f"✅ Image edit successful!")
        print(f"Edited image URL: {response.data[0].url}")

        # Test with multiple images (n=2)
        print("\n--- Testing Multiple Images (n=2) ---")
        with open(args.image, "rb") as image_file:
            response_multi = client.images.edit(
                model=args.model,
                image=image_file,
                prompt=args.prompt,
                n=2,
                response_format="url",
            )

        print(f"✅ Multiple image edit successful!")
        for i, data in enumerate(response_multi.data):
            print(f"Edited image {i+1} URL: {data.url}")

        # Test with b64_json format
        with open(args.image, "rb") as image_file:
            response_b64 = client.images.edit(
                model=args.model,
                image=image_file,
                prompt=args.prompt,
                n=1,
                response_format="b64_json",
            )

        print(f"✅ Base64 image edit successful!")
        print(
            f"Generated image as base64 (length: {len(response_b64.data[0].b64_json)} characters)"
        )

    except Exception as e:
        print(f"❌ Error during image editing: {e}")


def test_image_edit_direct_api():
    """Test image editing using direct API calls with multipart/form-data"""
    load_dotenv()

    BASE_URL = os.environ.get("OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080")
    API_KEY = os.environ.get("POE_API_KEY")

    if not API_KEY:
        print("❌ Error: POE_API_KEY environment variable not set")
        return

    image_path = (
        "scripts/eaa3f4c4fa1a10fb233e0e9fac9ec25ce67d77e365093671892d181d519bbf49.jpeg"
    )

    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return

    print("\n--- Testing Direct API Image Edit ---")
    print(f"Using image: {image_path}")

    # Prepare multipart form data
    with open(image_path, "rb") as image_file:
        files = {"image": ("image.jpg", image_file, "image/jpeg")}
        data = {
            "model": "StableDiffusionXL",
            "prompt": "Transform this into a futuristic cyberpunk scene",
            "n": "1",
            "response_format": "url",
        }
        headers = {"Authorization": f"Bearer {API_KEY}"}

        try:
            response = requests.post(
                f"{BASE_URL}/v1/images/edits",
                headers=headers,
                files=files,
                data=data,
                timeout=60,
            )

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print("✅ Direct API image edit successful!")
                print(f"Response: {result}")

                if "data" in result and result["data"]:
                    for i, item in enumerate(result["data"]):
                        if "url" in item:
                            print(f"Edited image {i+1} URL: {item['url']}")
                        elif "b64_json" in item:
                            print(
                                f"Edited image {i+1} base64 length: {len(item['b64_json'])}"
                            )

                # Test multiple images with direct API
                print("\n--- Testing Direct API Multiple Images (n=3) ---")
                data_multi = {
                    "model": "StableDiffusionXL",
                    "prompt": "Transform this into a vibrant cartoon style",
                    "n": "3",
                    "response_format": "url",
                }

                with open(image_path, "rb") as image_file:
                    files_multi = {"image": ("image.jpg", image_file, "image/jpeg")}

                    response_multi = requests.post(
                        f"{BASE_URL}/v1/images/edits",
                        headers=headers,
                        files=files_multi,
                        data=data_multi,
                        timeout=120,  # Longer timeout for multiple images
                    )

                    print(f"Status Code: {response_multi.status_code}")

                    if response_multi.status_code == 200:
                        result_multi = response_multi.json()
                        print("✅ Direct API multiple image edit successful!")
                        print(f"Generated {len(result_multi['data'])} images:")

                        for i, item in enumerate(result_multi["data"]):
                            if "url" in item:
                                print(f"  Image {i+1} URL: {item['url']}")
                    else:
                        print(
                            f"❌ Multiple image request failed with status code: {response_multi.status_code}"
                        )
                        print("Response:")
                        print(response_multi.text)
            else:
                print(f"❌ Request failed with status code: {response.status_code}")
                print("Response:")
                print(response.text)

        except requests.exceptions.Timeout:
            print("❌ Request timed out")
        except Exception as e:
            print(f"❌ Error during direct API request: {e}")


def main():
    print("🚀 Starting Image Edit Verification")

    # Test using OpenAI client
    print("\n=== Testing OpenAI Client Image Edit ===")
    test_image_edit_openai_client()

    # Test using direct API calls
    print("\n=== Testing Direct API Image Edit ===")
    test_image_edit_direct_api()

    print("\n🏁 Image edit verification complete!")


if __name__ == "__main__":
    main()
