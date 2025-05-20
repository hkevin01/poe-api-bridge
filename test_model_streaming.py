import pytest
import httpx
import json
import os
import logging
from typing import List, Dict, AsyncGenerator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model-streaming-tests")

# Configuration
OPENAI_COMPATIBLE_API_BASE_URL = os.getenv(
    "OPENAI_COMPATIBLE_API_BASE_URL", "http://localhost:8080"
)
API_KEY = os.getenv("POE_API_KEY", "test-key")  # Replace with your API key

# Test configuration
TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "1"))

# Get models from environment variable, fallback to default list
DEFAULT_MODELS = ["Claude-3.5-Sonnet", "Claude-3.7-Sonnet", "GPT-4o"]
AVAILABLE_MODELS = os.getenv("TEST_MODELS", ",".join(DEFAULT_MODELS)).split(",")

# Configure client defaults
client_defaults = {
    "timeout": TIMEOUT_SECONDS,
    "transport": httpx.AsyncHTTPTransport(retries=MAX_RETRIES),
}


def get_headers():
    return {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}


async def process_streaming_response(response: AsyncGenerator) -> List[Dict]:
    """Process streaming response and return list of decoded chunks."""
    chunks = []
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            data = line[6:]  # Remove 'data: ' prefix
            if data.strip() == "[DONE]":
                logger.debug("Received [DONE] message")
                break
            try:
                chunk = json.loads(data)
                chunks.append(chunk)

                # Log content if present
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    content = chunk["choices"][0].get("delta", {}).get("content")
                    if content:
                        logger.debug(f"Received content chunk: {content}")

                # Log finish reason if present
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    finish_reason = chunk["choices"][0].get("finish_reason")
                    if finish_reason:
                        logger.debug(f"Received finish_reason: {finish_reason}")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode chunk: {data}. Error: {str(e)}")
                continue
    return chunks


def extract_content_from_chunks(chunks: List[Dict]) -> str:
    """Extract and combine content from streaming chunks."""
    content_parts = []
    for chunk in chunks:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content:
                content_parts.append(content)
    return "".join(content_parts)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_streaming_response(model: str):
    """Test streaming response for each model."""
    logger.info(f"Testing streaming response for model: {model}")
    async with httpx.AsyncClient(**client_defaults) as client:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Write 'hello world' and nothing else."}
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "stream": True,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }

        response = await client.post(
            f"{OPENAI_COMPATIBLE_API_BASE_URL}/v1/chat/completions",
            json=payload,
            headers=get_headers(),
        )

        assert (
            response.status_code == 200
        ), f"Request failed with status {response.status_code}"
        chunks = await process_streaming_response(response)
        assert len(chunks) > 0, "No chunks received"

        # Verify chunk structure
        first_chunk = chunks[0]
        assert "id" in first_chunk, "First chunk missing 'id'"
        assert "choices" in first_chunk, "First chunk missing 'choices'"
        assert (
            first_chunk["object"] == "chat.completion.chunk"
        ), "Invalid chunk object type"

        # Verify response content
        complete_response = extract_content_from_chunks(chunks)
        assert (
            "hello world" in complete_response.lower()
        ), f"Expected 'hello world', got: {complete_response}"

        # Verify final chunk
        final_chunk = chunks[-1]
        assert (
            final_chunk["choices"][0]["finish_reason"] == "stop"
        ), "Missing or invalid finish_reason in final chunk"


@pytest.mark.asyncio
async def test_streaming_parameters():
    """Test streaming with different parameter combinations."""
    logger.info("Testing streaming with different parameter combinations")
    model = AVAILABLE_MODELS[0]  # Use first model for parameter testing
    test_cases = [
        {"temperature": 0.0, "top_p": 1.0},
        {"temperature": 0.7, "top_p": 0.9},
        {"temperature": 1.0, "top_p": 0.8},
    ]

    async with httpx.AsyncClient(**client_defaults) as client:
        for params in test_cases:
            logger.debug(f"Testing with parameters: {params}")
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Count from 1 to 3."}],
                "stream": True,
                "n": 1,
                **params,
            }

            response = await client.post(
                f"{OPENAI_COMPATIBLE_API_BASE_URL}/v1/chat/completions",
                json=payload,
                headers=get_headers(),
            )

            assert (
                response.status_code == 200
            ), f"Request failed with status {response.status_code}"
            chunks = await process_streaming_response(response)
            assert len(chunks) > 0, "No chunks received"

            # Verify content contains numbers
            complete_response = extract_content_from_chunks(chunks)
            assert any(
                str(i) in complete_response for i in range(1, 4)
            ), f"Expected numbers 1-3 in response, got: {complete_response}"

            # Verify final chunk
            final_chunk = chunks[-1]
            assert (
                final_chunk["choices"][0]["finish_reason"] == "stop"
            ), "Missing or invalid finish_reason in final chunk"


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling for streaming requests."""
    logger.info("Testing error handling for streaming requests")
    async with httpx.AsyncClient(**client_defaults) as client:
        # Test with invalid model
        payload = {
            "model": "invalid-model",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        }
        response = await client.post(
            f"{OPENAI_COMPATIBLE_API_BASE_URL}/v1/chat/completions",
            json=payload,
            headers=get_headers(),
        )
        assert response.status_code in [
            400,
            404,
        ], f"Expected 400/404 status for invalid model, got {response.status_code}"
        error_data = response.json()
        assert "error" in error_data.get(
            "detail", {}
        ), "Error response missing error details"

        # Test with empty messages
        payload = {"model": AVAILABLE_MODELS[0], "messages": [], "stream": True}
        response = await client.post(
            f"{OPENAI_COMPATIBLE_API_BASE_URL}/v1/chat/completions",
            json=payload,
            headers=get_headers(),
        )
        assert (
            response.status_code == 400
        ), f"Expected 400 status for empty messages, got {response.status_code}"
        error_data = response.json()
        assert "error" in error_data.get(
            "detail", {}
        ), "Error response missing error details"


@pytest.mark.asyncio
async def test_long_conversation_streaming():
    """Test streaming with a longer conversation context."""
    logger.info("Testing streaming with long conversation context")
    async with httpx.AsyncClient(**client_defaults) as client:
        payload = {
            "model": AVAILABLE_MODELS[0],
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "Write a simple hello world in Python."},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
        }

        response = await client.post(
            f"{OPENAI_COMPATIBLE_API_BASE_URL}/v1/chat/completions",
            json=payload,
            headers=get_headers(),
        )

        assert (
            response.status_code == 200
        ), f"Request failed with status {response.status_code}"
        chunks = await process_streaming_response(response)
        assert len(chunks) > 0, "No chunks received"

        # Verify response content
        complete_response = extract_content_from_chunks(chunks)
        assert (
            "print" in complete_response.lower()
        ), f"Expected print statement in response, got: {complete_response}"

        # Verify final chunk
        final_chunk = chunks[-1]
        assert (
            final_chunk["choices"][0]["finish_reason"] == "stop"
        ), "Missing or invalid finish_reason in final chunk"


def test_verify_base_url():
    """Verify that BASE_URL is properly configured."""
    assert OPENAI_COMPATIBLE_API_BASE_URL is not None, "BASE_URL must be set"
    assert OPENAI_COMPATIBLE_API_BASE_URL.startswith(
        ("http://", "https://")
    ), "BASE_URL must include protocol"
    assert not OPENAI_COMPATIBLE_API_BASE_URL.endswith(
        "/"
    ), "BASE_URL should not end with a slash"
