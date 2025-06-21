import pytest
from fastapi.testclient import TestClient
from server import app, fp, normalize_model, normalize_role
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import json
import os
from fastapi import HTTPException, Depends
from typing import Dict, Any, Optional


@pytest.fixture
def client():
    """Test client"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_get_bot_response():
    async def mock_response(
        messages, bot_name, api_key, skip_system_prompt=False, base_url=None
    ):
        # Simulate streaming response
        response = fp.PartialResponse(text="Test response")
        yield response

    with patch("server.get_bot_response", mock_response):
        yield mock_response


@pytest.fixture
def mock_get_bot_response_with_replace():
    """Mock that simulates a response with is_replace_response=True."""
    async def mock_response(
        messages, bot_name, api_key, skip_system_prompt=False, base_url=None
    ):
        # First yield normal response
        initial = fp.PartialResponse(text="initial text")
        yield initial
        
        # Then yield replacement response
        replacement = fp.PartialResponse(
            text="replacement text",
            is_replace_response=True
        )
        yield replacement
    
    with patch("server.get_bot_response", mock_response):
        yield mock_response


def test_missing_auth(client):
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 401
    assert "authentication_error" in response.json()["detail"]["error"]["type"]


def test_malformed_auth_header(client):
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    headers = {"Authorization": "malformed_header"}

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 401
    assert "authentication_error" in response.json()["detail"]["error"]["type"]


def test_invalid_scheme_auth_header(client):
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    headers = {"Authorization": "Invalid test_api_key"}

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 401
    assert "authentication_error" in response.json()["detail"]["error"]["type"]


def test_empty_messages(client):
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [],
    }
    headers = {"Authorization": "Bearer test_api_key"}

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 400
    assert response.json()["detail"]["error"]["type"] == "invalid_request_error"
    assert "Messages array cannot be empty" in response.json()["detail"]["error"]["message"]


def test_normalize_model():
    # Test that the function normalizes model names correctly
    assert normalize_model("  claude-3.5-sonnet  ") == "claude-3.5-sonnet"
    assert normalize_model("GPT-4") == "GPT-4"
    assert normalize_model("test-model") == "test-model"


def test_normalize_role():
    # Test role normalization
    assert normalize_role("user") == "user"
    assert normalize_role("assistant") == "bot"
    assert normalize_role("system") == "system"
    assert normalize_role("custom") == "custom"


def test_chat_completion_success(client, mock_get_bot_response):
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": False,
    }
    headers = {"Authorization": "Bearer test_api_key"}

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    assert "choices" in response_data
    assert len(response_data["choices"]) == 1
    assert response_data["choices"][0]["message"]["content"] == "Test response"
    assert response_data["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in response_data


def test_chat_completion_streaming(client, mock_get_bot_response):
    request_data = {
        "model": "Claude-3.5-Sonnet", 
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": True,
    }
    headers = {"Authorization": "Bearer test_api_key"}

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_completions_endpoint(client, mock_get_bot_response):
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "prompt": "Hello, how are you?",
        "stream": False,
    }
    headers = {"Authorization": "Bearer test_api_key"}

    response = client.post("/v1/completions", json=request_data, headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    assert "choices" in response_data
    assert len(response_data["choices"]) == 1
    assert response_data["choices"][0]["text"] == "Test response"


def test_models_endpoint(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    response_data = response.json()
    assert "data" in response_data
    assert len(response_data["data"]) > 0
    # Check that Claude models are included
    model_ids = [model["id"] for model in response_data["data"]]
    assert "Claude-3.5-Sonnet" in model_ids
    assert "GPT-4o" in model_ids


def test_openapi_endpoint(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    response_data = response.json()
    assert "info" in response_data
    assert response_data["info"]["title"] == "Poe-API OpenAI Proxy"


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200


def test_complex_message_content(client, mock_get_bot_response):
    """Test handling of complex message content with arrays"""
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "image_url": "http://example.com/image.jpg"}
                ]
            }
        ],
        "stream": False,
    }
    headers = {"Authorization": "Bearer test_api_key"}

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    assert "choices" in response_data


def test_is_replace_response_functionality(client, mock_get_bot_response_with_replace):
    """Test that is_replace_response works correctly"""
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [{"role": "user", "content": "Test replacement"}],
        "stream": False,
    }
    headers = {"Authorization": "Bearer test_api_key"}

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    # The final response should be "replacement text" not "initial textreplacement text"
    assert response_data["choices"][0]["message"]["content"] == "replacement text"
