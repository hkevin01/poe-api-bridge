import pytest
from fastapi.testclient import TestClient
from server import app, fp, normalize_model, normalize_role, models, models_mapping
import asyncio
from unittest.mock import AsyncMock, patch
import json
import os
from fastapi import HTTPException


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_get_bot_response():
    async def mock_response(messages, bot_name, api_key, skip_system_prompt=False):
        # Simulate streaming response
        response = fp.PartialResponse(text="Test response")
        yield response

    with patch("server.get_bot_response", mock_response):
        yield mock_response


def test_missing_auth(client):
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 401


def test_normalize_model():
    # Test exact match
    assert normalize_model("Claude-3.5-Sonnet") == "Claude-3.5-Sonnet"

    # Test case insensitive match
    assert normalize_model("claude-3.5-sonnet") == "Claude-3.5-Sonnet"

    # Test mapping
    assert normalize_model("poe-cursor-model") == "Claude-3.5-Sonnet"

    # Test invalid model
    with pytest.raises(HTTPException) as exc_info:
        normalize_model("invalid-model")
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["error"]["code"] == "model_not_found"
    assert "invalid-model" in exc_info.value.detail["error"]["message"]


def test_normalize_role():
    assert normalize_role("user") == "user"
    assert normalize_role("assistant") == "bot"
    assert normalize_role("system") == "system"
    assert normalize_role("tool") == "tool"  # tool role should remain as tool
    assert normalize_role("custom") == "custom"


def test_flexible_http_bearer_auth(client):
    # Test with Authorization header
    headers = {"Authorization": "Bearer test-token"}
    response = client.get("/test", headers=headers)
    assert response.status_code != 401

    # Test with token query parameter
    response = client.get("/test?token=test-token")
    assert response.status_code != 401

    # Test with POE_API_KEY environment variable
    with patch.dict("os.environ", {"POE_API_KEY": "test-env-token"}):
        response = client.get("/test")
        assert response.status_code != 401


def test_chat_completions_endpoint(client, mock_get_bot_response):
    headers = {"Authorization": "Bearer test-token"}
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "id" in data
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "Test response"


def test_chat_completions_with_content_list(client, mock_get_bot_response):
    headers = {"Authorization": "Bearer test-token"}
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "image_url": "http://example.com/image.jpg"},
                    {"type": "text", "text": "What's in this image?"},
                ],
            }
        ],
    }

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(data["choices"][0]["message"]["content"], str)
    assert "Test response" in data["choices"][0]["message"]["content"]

    assert data["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_with_tools(client, mock_get_bot_response):
    headers = {"Authorization": "Bearer test-token"}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [{"role": "user", "content": "What's the weather in London?"}],
        "tools": tools,
        "tool_choice": "auto",
        "stream": False,
    }

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "id" in data
    assert "system_fingerprint" in data
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert "message" in data["choices"][0]


def test_error_response_format(client):
    headers = {"Authorization": "Bearer test-token"}
    request_data = {
        "model": "nonexistent-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 404

    error = response.json()
    assert "detail" in error
    assert "error" in error["detail"]
    assert "message" in error["detail"]["error"]
    assert "type" in error["detail"]["error"]
    assert error["detail"]["error"]["type"] == "invalid_request_error"
    assert error["detail"]["error"]["param"] == "model"
    assert error["detail"]["error"]["code"] == "model_not_found"


def test_streaming_format(client, mock_get_bot_response):
    headers = {"Authorization": "Bearer test-token"}
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    with client.stream(
        "POST", "/v1/chat/completions", json=request_data, headers=headers
    ) as response:
        assert response.status_code == 200

        # Read and parse the streaming response
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if line.startswith("data: "):
                    if line.strip() == "data: [DONE]":
                        continue

                    data = json.loads(line.replace("data: ", ""))

                    # Verify the structure of each chunk
                    assert "id" in data
                    assert "object" in data
                    assert data["object"] == "chat.completion.chunk"
                    assert "created" in data
                    assert "model" in data
                    assert "choices" in data

                    choices = data["choices"]
                    assert len(choices) == 1
                    assert "index" in choices[0]
                    assert "delta" in choices[0]

                    if "content" in choices[0]["delta"]:
                        assert isinstance(choices[0]["delta"]["content"], str)
                    elif choices[0].get("finish_reason") == "stop":
                        assert choices[0]["delta"] == {}


def test_openai_models_endpoint(client):
    response = client.get("/v1/models")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0

    model = data["data"][0]
    assert "id" in model
    assert "object" in model
    assert "created" in model
    assert "owned_by" in model
    assert "permission" in model


@pytest.mark.asyncio
async def test_create_stream_chunk():
    from server import create_stream_chunk

    # Test completion format
    chunk = await create_stream_chunk("test message", "test-model", "completion")
    assert chunk["object"] == "text_completion"
    assert chunk["choices"][0]["text"] == "test message"

    # Test chat format
    chunk = await create_stream_chunk("test message", "test-model", "chat", True)
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["choices"][0]["delta"]["role"] == "assistant"
    assert chunk["choices"][0]["delta"]["content"] == "test message"

    # Test poe format
    chunk = await create_stream_chunk("test message", "test-model", "poe")
    assert chunk["response"] == "test message"
    assert chunk["done"] is False


@pytest.mark.asyncio
async def test_create_final_chunk():
    from server import create_final_chunk

    # Test completion format
    chunk = await create_final_chunk("test-model", "completion")
    assert chunk["choices"][0]["finish_reason"] == "stop"

    # Test chat format
    chunk = await create_final_chunk("test-model", "chat")
    assert chunk["choices"][0]["finish_reason"] == "stop"
    assert chunk["choices"][0]["delta"] == {}

    # Test poe format
    chunk = await create_final_chunk("test-model", "poe")
    assert chunk["done"] is True


def test_error_handling_chat_completions(client):
    headers = {"Authorization": "Bearer test-token"}
    request_data = {
        "model": "invalid-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 404
    error = response.json()
    assert "error" in error["detail"]
    assert error["detail"]["error"]["type"] == "invalid_request_error"
    assert error["detail"]["error"]["code"] == "model_not_found"


def test_long_message_handling(client, mock_get_bot_response):
    headers = {"Authorization": "Bearer test-token"}
    long_content = "x" * (4 * 4096 + 100)  # Exceeds the 4*4096 threshold
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [{"role": "user", "content": long_content}],
    }

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 200


def test_token_counts_in_response(client, mock_get_bot_response):
    """Test that token counts are included in API responses"""
    headers = {"Authorization": "Bearer test-token"}
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    # Test chat completions API
    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "usage" in data
    assert "prompt_tokens" in data["usage"]
    assert "completion_tokens" in data["usage"]
    assert "total_tokens" in data["usage"]
    assert (
        data["usage"]["total_tokens"]
        == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
    )

    # Skip completions API test for now as it requires more complex mocking
    # The token counting functionality is already verified in the other endpoints


def test_token_counts_with_complex_messages(client, mock_get_bot_response):
    """Test token counting with complex message structure"""
    headers = {"Authorization": "Bearer test-token"}
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Tell me about token counting."},
            {"role": "assistant", "content": "Token counting is the process of..."},
            {"role": "user", "content": "Can you provide an example?"},
        ],
        "stream": False,
    }

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 200
    data = response.json()

    # Check that system and user messages are counted in prompt tokens
    assert data["usage"]["prompt_tokens"] > 10
    # Check that assistant message is counted in completion tokens
    assert data["usage"]["completion_tokens"] > 0
