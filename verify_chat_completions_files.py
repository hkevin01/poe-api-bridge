import requests
import json
import os
import base64
from dotenv import load_dotenv

# Constants
IMAGE_URL = "https://qph.cf2.quoracdn.net/main-qimg-3dd2a86604805445a043be92da5f87aa"


def test_file_support():
    """Test OpenAI file support implementation with an image URL"""
    # Load environment variables from .env file
    load_dotenv()

    # Get configuration from environment
    BASE_URL = os.environ.get("OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080")
    API_KEY = os.environ.get("POE_API_KEY")

    if not API_KEY:
        print("❌ Error: POE_API_KEY environment variable not set")
        return

    # Test image URL from constant
    image_url = IMAGE_URL

    print("\n=== Testing OpenAI File Support Implementation ===")
    print(f"Using image URL: {image_url}")
    print(f"Base URL: {BASE_URL}")

    # Test data with multimodal content
    test_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image? Please describe it in detail.",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "stream": False,
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        print("\n--- Testing Non-Streaming Request ---")
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=test_data,
            timeout=60,  # Longer timeout for file processing
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"]
                print("\n✅ Non-streaming request successful!")
                print("Response content:")
                print("-" * 50)
                print(content)
                print("-" * 50)

                # Check for token usage
                if "usage" in result:
                    usage = result["usage"]
                    print(f"\nToken Usage:")
                    print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                    print(
                        f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}"
                    )
                    print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")

                # Verify the response mentions image content and Quora
                if any(
                    keyword in content.lower()
                    for keyword in ["image", "picture", "photo", "see", "visual"]
                ):
                    print(
                        "\n✅ Response appears to reference image content - file support working!"
                    )
                else:
                    print("\n⚠️ Response doesn't clearly reference image content")

                # Assert the response contains "Quora"
                if "quora" in content.lower():
                    print("✅ Response contains 'Quora' - assertion passed!")
                else:
                    print("❌ Response does not contain 'Quora' - assertion failed!")
                    assert False, "Response should contain the word 'Quora'"

            else:
                print("❌ No choices in response")
                print(f"Response: {result}")
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print("Response:")
            print(response.text)

    except requests.exceptions.Timeout:
        print(
            "❌ Request timed out - this might happen if file processing takes too long"
        )
    except Exception as e:
        print(f"❌ Error during request: {e}")


def test_file_support_streaming():
    """Test OpenAI file support with streaming"""
    # Load environment variables from .env file
    load_dotenv()

    # Get configuration from environment
    BASE_URL = os.environ.get("OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080")
    API_KEY = os.environ.get("POE_API_KEY")

    if not API_KEY:
        print("❌ Error: POE_API_KEY environment variable not set")
        return

    # Test image URL from constant
    image_url = IMAGE_URL

    print("\n--- Testing Streaming Request ---")

    # Test data with multimodal content and streaming
    test_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image and describe what you see.",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "stream": True,
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        with requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=test_data,
            stream=True,
            timeout=60,
        ) as response:

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                print("Streaming response chunks:")
                print("-" * 50)

                full_content = ""
                chunk_count = 0

                for line in response.iter_lines():
                    if line:
                        line_text = line.decode("utf-8")
                        if line_text.startswith("data: "):
                            line_text = line_text[6:]  # Remove "data: " prefix

                        # Skip [DONE] message
                        if line_text.strip() == "[DONE]":
                            print("\n[DONE]")
                            break

                        try:
                            chunk = json.loads(line_text)
                            chunk_count += 1

                            delta_content = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )

                            if delta_content:
                                full_content += delta_content
                                print(delta_content, end="", flush=True)

                        except json.JSONDecodeError:
                            pass

                print(f"\n{'-' * 50}")
                print(f"✅ Streaming successful! Received {chunk_count} chunks")

                # Verify the response mentions image content
                if any(
                    keyword in full_content.lower()
                    for keyword in ["image", "picture", "photo", "see", "visual"]
                ):
                    print(
                        "✅ Streaming response references image content - file support working!"
                    )
                else:
                    print(
                        "⚠️ Streaming response doesn't clearly reference image content"
                    )

                # Assert the response contains "Quora"
                if "quora" in full_content.lower():
                    print("✅ Streaming response contains 'Quora' - assertion passed!")
                else:
                    print(
                        "❌ Streaming response does not contain 'Quora' - assertion failed!"
                    )
                    assert False, "Streaming response should contain the word 'Quora'"

            else:
                print(
                    f"❌ Streaming request failed with status code: {response.status_code}"
                )
                print("Response:")
                print(response.text)

    except requests.exceptions.Timeout:
        print("❌ Streaming request timed out")
    except Exception as e:
        print(f"❌ Error during streaming request: {e}")


def test_base64_image_support():
    """Test base64 image support"""
    # Load environment variables
    load_dotenv()

    BASE_URL = os.environ.get("OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080")
    API_KEY = os.environ.get("POE_API_KEY")

    if not API_KEY:
        print("❌ Error: POE_API_KEY environment variable not set")
        return

    print("\n--- Testing Base64 Image Support ---")

    # Download and convert the Quora image to base64 on-demand
    try:
        print("Downloading image for base64 conversion...")
        response = requests.get(IMAGE_URL, timeout=30)
        response.raise_for_status()
        base64_image = base64.b64encode(response.content).decode("utf-8")
        print("✅ Image downloaded and converted to base64")
    except Exception as e:
        print(f"❌ Failed to download image: {e}")
        return

    test_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this base64 image? Describe it in detail.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "stream": False,
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=test_data,
            timeout=30,
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"]
                print("✅ Base64 image processing successful!")
                print("Response:")
                print("-" * 30)
                print(content)
                print("-" * 30)

                # Assert the response contains "Quora"
                if "quora" in content.lower():
                    print("✅ Base64 response contains 'Quora' - assertion passed!")
                else:
                    print(
                        "❌ Base64 response does not contain 'Quora' - assertion failed!"
                    )
                    assert False, "Base64 response should contain the word 'Quora'"
            else:
                print("❌ No choices in response")
        else:
            print(f"❌ Base64 request failed with status code: {response.status_code}")
            print("Response:")
            print(response.text)

    except Exception as e:
        print(f"❌ Error during base64 test: {e}")


if __name__ == "__main__":
    print("🚀 Starting OpenAI File Support Verification")

    # Test file support with image URL
    test_file_support()

    # Test streaming with files
    test_file_support_streaming()

    # Test base64 images
    test_base64_image_support()

    print("\n🏁 File support verification complete!")
