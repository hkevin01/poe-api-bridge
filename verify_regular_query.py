import requests
import json
import os
from dotenv import load_dotenv


def test_regular_query():
    # Load environment variables from .env file
    load_dotenv()

    BASE_URL = "http://localhost:80"
    API_KEY = os.environ["POE_API_KEY"]

    if not API_KEY:
        print("❌ Error: POE_API_KEY environment variable not set")
        return

    # Test data with a simple query
    test_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    # Test non-streaming call
    print("\n=== Testing Regular Query ===")
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions", headers=headers, json=test_data
    )

    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))

    if response.status_code == 200:
        result = response.json()
        if "choices" in result and result["choices"][0].get("message", {}).get(
            "content"
        ):
            print("\n✅ Query successful!")
            content = result["choices"][0]["message"]["content"]
            print("\nResponse Content:")
            print(content)
        else:
            print("\n❌ Query failed - no content in response")
    else:
        print("\n❌ Query failed")


def test_streaming_query():
    # Load environment variables from .env file
    load_dotenv()

    BASE_URL = "http://localhost:80"
    API_KEY = os.environ["POE_API_KEY"]

    if not API_KEY:
        print("❌ Error: POE_API_KEY environment variable not set")
        return

    # Test data with a simple query
    test_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "stream": True,
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    # Test streaming call
    print("\n=== Testing Streaming Query ===")
    full_content = ""
    chunk_count = 0
    tokens_received = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    try:
        with requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=test_data,
            stream=True,
        ) as response:

            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("Streaming response chunks:")
                for line in response.iter_lines():
                    if line:
                        # Remove the "data: " prefix if present and parse JSON
                        line_text = line.decode("utf-8")
                        if line_text.startswith("data: "):
                            line_text = line_text[6:]  # Remove "data: " prefix

                        # Skip [DONE] message
                        if line_text.strip() == "[DONE]":
                            continue

                        try:
                            chunk = json.loads(line_text)
                            chunk_count += 1

                            # Extract token information - normally appears in final chunk
                            if "usage" in chunk:
                                prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                                completion_tokens = chunk["usage"].get(
                                    "completion_tokens", 0
                                )
                                total_tokens = chunk["usage"].get("total_tokens", 0)
                                tokens_received = completion_tokens

                            delta_content = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta_content:
                                full_content += delta_content
                                print(delta_content, end="", flush=True)
                        except json.JSONDecodeError:
                            print(f"Could not decode JSON: {line_text}")

                print("\n\n✅ Streaming query successful!")
                print(f"\nChunks received: {chunk_count}")
                print("\nToken statistics:")
                print(f"Prompt tokens: {prompt_tokens}")
                print(f"Completion tokens: {completion_tokens}")
                print(f"Total tokens: {total_tokens}")
                print("\nFull assembled content:")
                print(full_content)
            else:
                print(
                    f"\n❌ Streaming query failed with status code: {response.status_code}"
                )
                print("Response:")
                print(response.text)
    except Exception as e:
        print(f"\n❌ Error during streaming request: {e}")


if __name__ == "__main__":
    test_regular_query()
    test_streaming_query()
