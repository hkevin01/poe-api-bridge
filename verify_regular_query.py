import requests
import json
import os
from dotenv import load_dotenv


def test_regular_query():
    # Load environment variables from .env file
    load_dotenv()

    # Try to get base URL from environment variable, fall back to local URL if not present
    BASE_URL = os.environ["OPENAI_COMPATIBLE_API_BASE_URL"]
    API_KEY = os.environ["POE_API_KEY"]

    if not API_KEY:
        return

    # Test data with a simple query
    test_data = {
        "model": "GPT-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    # Test non-streaming call
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions", headers=headers, json=test_data
    )


    if response.status_code == 200:
        result = response.json()
        if "choices" in result and result["choices"][0].get("message", {}).get(
            "content"
        ):
            content = result["choices"][0]["message"]["content"]


def test_streaming_query():
    # Load environment variables from .env file
    load_dotenv()

    # Get port from environment or use default 8080
    port = os.environ.get("SERVER_PORT", "8080")
    # Try to get base URL from environment variable, fall back to local URL if not present
    BASE_URL = os.environ.get("OPENAI_COMPATIBLE_API_BASE_URL", f"http://localhost:{port}")
    API_KEY = os.environ["POE_API_KEY"]

    if not API_KEY:
        return

    # Test data with a simple query
    test_data = {
        "model": "GPT-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "stream": True,
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    # Test streaming call
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

            if response.status_code == 200:
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
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        pass


if __name__ == "__main__":
    test_regular_query()
    test_streaming_query()
