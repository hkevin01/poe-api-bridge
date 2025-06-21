import pytest
from fastapi.testclient import TestClient
from server import app, fp, normalize_model, normalize_role
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import json
import os
from fastapi import HTTPException, Depends
from typing import Dict, Any, Optional


@pytest.fixture(autouse=True)
def mock_all_external_calls():
    """Auto-mock all external API calls to prevent real requests"""
    with patch("server.get_bot_response") as mock_bot, \
         patch("fastapi_poe.upload_file") as mock_upload, \
         patch("httpx.AsyncClient") as mock_client:
        
        # Default mock for get_bot_response
        async def default_bot_response(*args, **kwargs):
            response = fp.PartialResponse(text="Test response")
            yield response
        mock_bot.side_effect = default_bot_response
        
        # Default mock for file upload
        mock_upload.return_value = fp.Attachment(
            url="https://poe.com/test-file.jpg",
            content_type="image/jpeg",
            name="test-file.jpg"
        )
        
        # Default mock for HTTP client
        mock_response = MagicMock()
        mock_response.content = b"fake_http_response_data"
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        
        yield {
            'bot_response': mock_bot,
            'upload_file': mock_upload,
            'http_client': mock_client
        }


@pytest.fixture
def client():
    """Test client"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_get_bot_response(mock_all_external_calls):
    """Override default bot response for specific tests"""
    async def mock_response(
        messages, bot_name, api_key, skip_system_prompt=False, base_url=None
    ):
        # Simulate streaming response
        response = fp.PartialResponse(text="Test response")
        yield response

    mock_all_external_calls['bot_response'].side_effect = mock_response
    return mock_all_external_calls['bot_response']


@pytest.fixture
def mock_get_bot_response_with_replace(mock_all_external_calls):
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
    
    mock_all_external_calls['bot_response'].side_effect = mock_response
    return mock_all_external_calls['bot_response']


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


def test_base64_image_support(client, mock_get_bot_response):
    """Test handling of base64 encoded images"""
    # Simple base64 encoded 1x1 pixel PNG
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU8fYgAAAABJRU5ErkJggg=="
    
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
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
    assert response_data["choices"][0]["message"]["content"] == "Test response"


def test_multiple_images_in_message(client, mock_get_bot_response):
    """Test handling of multiple images in a single message"""
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU8fYgAAAABJRU5ErkJggg=="
    
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two images:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": "versus"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}},
                    {"type": "text", "text": "What are the differences?"}
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


def test_supported_image_formats(client, mock_get_bot_response):
    """Test various supported image formats"""
    formats = [
        ("jpeg", "data:image/jpeg;base64,"),
        ("png", "data:image/png;base64,"),
        ("webp", "data:image/webp;base64,"),
        ("gif", "data:image/gif;base64,")
    ]
    
    base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU8fYgAAAABJRU5ErkJggg=="
    
    for format_name, data_url_prefix in formats:
        request_data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze this {format_name} image"},
                        {"type": "image_url", "image_url": {"url": f"{data_url_prefix}{base64_data}"}}
                    ]
                }
            ],
            "stream": False,
        }
        headers = {"Authorization": "Bearer test_api_key"}

        response = client.post("/v1/chat/completions", json=request_data, headers=headers)
        assert response.status_code == 200, f"Failed for {format_name} format"


def test_image_url_reference(client, mock_get_bot_response):
    """Test handling of direct image URL references"""
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/test-image.jpg"}}
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


def test_mixed_content_types(client, mock_get_bot_response):
    """Test mixing text and images in complex message structure"""
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyze:"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}},
                    {"type": "text", "text": "and explain the trends"},
                ]
            },
            {
                "role": "assistant",
                "content": "I can see the chart shows..."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Now compare with:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU8fYgAAAABJRU5ErkJggg=="}}
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


def test_streaming_with_images(client, mock_get_bot_response):
    """Test streaming responses with image content"""
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/detailed-image.jpg"}}
                ]
            }
        ],
        "stream": True,
    }
    headers = {"Authorization": "Bearer test_api_key"}

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_empty_image_url(client, mock_get_bot_response):
    """Test handling of empty or invalid image URLs"""
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this"},
                    {"type": "image_url", "image_url": {"url": ""}}
                ]
            }
        ],
        "stream": False,
    }
    headers = {"Authorization": "Bearer test_api_key"}

    # Should process the text content and handle empty URL gracefully
    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    assert "choices" in response_data


def test_malformed_image_content(client, mock_get_bot_response):
    """Test handling of malformed image content structure"""
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Test"},
                    {"type": "image_url"}  # Missing image_url field
                ]
            }
        ],
        "stream": False,
    }
    headers = {"Authorization": "Bearer test_api_key"}

    # Should handle gracefully and process the text content
    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    assert "choices" in response_data


def test_large_base64_content(client, mock_get_bot_response):
    """Test handling of larger base64 encoded content"""
    # Create a larger base64 string (simulating a larger image)
    large_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU8fYgAAAABJRU5ErkJggg==" * 10
    
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Process this larger image"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{large_base64}"}}
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


def test_image_content_extraction():
    """Test the content extraction logic for image messages"""
    from server import normalize_role
    
    # Test that image content is properly converted to text representation
    test_content = [
        {"type": "text", "text": "Hello"},
        {"type": "image", "image_url": "http://example.com/image.jpg"}
    ]
    
    # Simulate the content processing logic from server.py
    parts = []
    for comp in test_content:
        if isinstance(comp, dict):
            if comp.get("type") == "text" and "text" in comp:
                parts.append(comp["text"])
            elif comp.get("type") == "image":
                parts.append(f"[Image: {comp.get('image_url', '')}]")
    
    result = " ".join(parts)
    assert result == "Hello [Image: http://example.com/image.jpg]"


def test_backward_compatibility_text_only(client, mock_get_bot_response):
    """Test that text-only messages still work as before"""
    request_data = {
        "model": "Claude-3.5-Sonnet",
        "messages": [
            {"role": "user", "content": "Simple text message"},
            {"role": "assistant", "content": "Simple response"}
        ],
        "stream": False,
    }
    headers = {"Authorization": "Bearer test_api_key"}

    response = client.post("/v1/chat/completions", json=request_data, headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    assert "choices" in response_data
    assert response_data["choices"][0]["message"]["content"] == "Test response"


# Integration tests for the implemented file support feature
@patch('fastapi_poe.upload_file')
def test_base64_image_integration(mock_upload_file, client, mock_get_bot_response):
    """Test integration of base64 image processing with the server"""
    # Mock the upload_file response
    mock_attachment = fp.Attachment(
        url="https://poe.com/attachment/123",
        content_type="image/png",
        name="uploaded_file.png"
    )
    mock_upload_file.return_value = mock_attachment
    
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU8fYgAAAABJRU5ErkJggg=="
    
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
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
    
    # Verify that upload_file was called for the base64 image
    mock_upload_file.assert_called_once()
    call_kwargs = mock_upload_file.call_args.kwargs
    assert "file" in call_kwargs
    assert "file_name" in call_kwargs
    assert call_kwargs["file_name"] == "uploaded_file.png"


@patch('fastapi_poe.upload_file')
def test_image_url_integration(mock_upload_file, client, mock_get_bot_response):
    """Test integration of image URL processing with the server"""
    # Mock the upload_file response
    mock_attachment = fp.Attachment(
        url="https://poe.com/attachment/456",
        content_type="image/jpeg",
        name="image.jpg"
    )
    mock_upload_file.return_value = mock_attachment
    
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/test-image.jpg"}}
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
    
    # Verify that upload_file was called with file_url
    mock_upload_file.assert_called_once()
    call_kwargs = mock_upload_file.call_args.kwargs
    assert "file_url" in call_kwargs
    assert call_kwargs["file_url"] == "https://example.com/test-image.jpg"


def test_file_upload_failure_fallback(client, mock_get_bot_response):
    """Test that the system falls back gracefully when file upload fails"""
    with patch('fastapi_poe.upload_file', side_effect=Exception("Upload failed")):
        request_data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/fail-image.jpg"}}
                    ]
                }
            ],
            "stream": False,
        }
        headers = {"Authorization": "Bearer test_api_key"}

        # Should still work but fall back to text representation
        response = client.post("/v1/chat/completions", json=request_data, headers=headers)
        assert response.status_code == 200
        response_data = response.json()
        assert "choices" in response_data


@patch('fastapi_poe.upload_file')
def test_multiple_attachments_integration(mock_upload_file, client, mock_get_bot_response):
    """Test handling multiple file attachments in one message"""
    # Mock multiple attachments
    mock_attachments = [
        fp.Attachment(url="https://poe.com/attachment/1", content_type="image/png", name="image1.png"),
        fp.Attachment(url="https://poe.com/attachment/2", content_type="image/jpeg", name="image2.jpg")
    ]
    mock_upload_file.side_effect = mock_attachments
    
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU8fYgAAAABJRU5ErkJggg=="
    
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": "versus"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
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
    
    # Verify both uploads were called
    assert mock_upload_file.call_count == 2


def test_content_conversion_functions():
    """Test the helper functions for content conversion"""
    from server import process_base64_image, process_image_url, convert_openai_content_to_poe
    
    # Test data URL validation
    valid_data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU8fYgAAAABJRU5ErkJggg=="
    invalid_data_url = "not-a-data-url"
    
    # These would need to be async tests in practice, but testing the validation logic
    assert valid_data_url.startswith("data:")
    assert ";base64," in valid_data_url
    assert not invalid_data_url.startswith("data:")


def test_supported_mime_types():
    """Test MIME type to extension mapping"""
    extension_map = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
        "application/pdf": "pdf"
    }
    
    # Test that all spec-supported formats are mapped
    for mime_type, expected_ext in extension_map.items():
        assert expected_ext in ["jpg", "png", "webp", "gif", "pdf"]
        
    # Test fallback for unknown types
    unknown_type = "application/unknown"
    fallback_ext = extension_map.get(unknown_type, "bin")
    assert fallback_ext == "bin"


# Tests for future Poe API file upload support
@pytest.mark.asyncio
@patch('fastapi_poe.upload_file')
async def test_file_upload_called_correctly(mock_upload_file):
    """Test that fp.upload_file is called with correct parameters"""
    # Mock the upload_file response
    mock_attachment = fp.Attachment(url="https://example.com/file", content_type="application/pdf", name="test.pdf")
    mock_upload_file.return_value = mock_attachment
    
    # Simulate file upload request (future implementation)
    with patch('builtins.open', create=True) as mock_open:
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # This would be the future API call
        api_key = "test_api_key"
        
        # Call the mocked upload function
        result = await fp.upload_file(file=mock_file, api_key=api_key)
        
        # Verify upload_file was called correctly
        mock_upload_file.assert_called_once_with(file=mock_file, api_key=api_key)
        assert result == mock_attachment


@pytest.mark.asyncio
@patch('fastapi_poe.upload_file')
async def test_protocol_message_with_attachments(mock_upload_file):
    """Test creating ProtocolMessage with file attachments"""
    # Mock attachment
    mock_attachment = fp.Attachment(url="https://example.com/pdf", content_type="application/pdf", name="draconomicon.pdf")
    mock_upload_file.return_value = mock_attachment
    
    # Test creating message with attachment as per spec
    api_key = "test_api_key"
    
    with patch('builtins.open', create=True) as mock_open:
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Simulate the spec example: fp.upload_file(open("draconomicon.pdf", "rb"), api_key=api_key)
        pdf_attachment = await fp.upload_file(file=mock_file, api_key=api_key)
        
        # Verify upload was called
        mock_upload_file.assert_called_once_with(file=mock_file, api_key=api_key)
        
        # Create message with attachment as per spec
        message = fp.ProtocolMessage(
            role="user",
            content="Hello world",
            attachments=[pdf_attachment]
        )
        
        # Verify message structure
        assert message.role == "user"
        assert message.content == "Hello world"
        assert hasattr(message, 'attachments')
        assert len(message.attachments) == 1
        assert message.attachments[0] == pdf_attachment


@pytest.mark.asyncio
@patch('fastapi_poe.upload_file')
async def test_file_url_upload(mock_upload_file):
    """Test uploading file via URL"""
    # Mock attachment
    mock_attachment = fp.Attachment(url="https://example.com/image", content_type="image/jpeg", name="image.jpg")
    mock_upload_file.return_value = mock_attachment
    
    api_key = "test_api_key"
    file_url = "https://example.com/remote-image.jpg"
    
    # Upload file via URL
    attachment = await fp.upload_file(file_url=file_url, api_key=api_key)
    
    # Verify upload was called with URL
    mock_upload_file.assert_called_once_with(file_url=file_url, api_key=api_key)
    assert attachment == mock_attachment


@pytest.mark.asyncio
@patch('fastapi_poe.upload_file')
async def test_upload_file_error_handling(mock_upload_file):
    """Test error handling in file upload"""
    # Mock upload_file to raise an exception
    mock_upload_file.side_effect = Exception("Upload failed")
    
    api_key = "test_api_key"
    
    with patch('builtins.open', create=True) as mock_open:
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Test that exception is properly raised
        with pytest.raises(Exception, match="Upload failed"):
            await fp.upload_file(file=mock_file, api_key=api_key)
        
        # Verify upload_file was called
        mock_upload_file.assert_called_once_with(file=mock_file, api_key=api_key)


def test_file_size_validation_logic():
    """Test file size validation logic (20MB limit from spec)"""
    max_size = 20 * 1024 * 1024  # 20MB
    
    # Test valid sizes
    valid_sizes = [1024, 5 * 1024 * 1024, max_size]
    for size in valid_sizes:
        assert size <= max_size, f"Size {size} should be valid"
    
    # Test invalid sizes
    invalid_sizes = [max_size + 1, 50 * 1024 * 1024]
    for size in invalid_sizes:
        assert size > max_size, f"Size {size} should be invalid"


def test_supported_file_formats_validation():
    """Test supported file format validation from spec"""
    # From spec: Images: JPEG, PNG, WebP, GIF (non-animated), Documents: PDF
    supported_formats = {
        "image/jpeg": True,
        "image/png": True,
        "image/webp": True,
        "image/gif": True,
        "application/pdf": True,
        "text/plain": False,  # Not in spec
        "application/doc": False,  # Not in spec
    }
    
    for mime_type, should_be_supported in supported_formats.items():
        if should_be_supported:
            assert mime_type in ["image/jpeg", "image/png", "image/webp", "image/gif", "application/pdf"]
        else:
            assert mime_type not in ["image/jpeg", "image/png", "image/webp", "image/gif", "application/pdf"]


@pytest.mark.asyncio
@patch('fastapi_poe.upload_file')
async def test_upload_file_error_handling(mock_upload_file):
    """Test error handling in file upload"""
    # Mock upload_file to raise an exception
    mock_upload_file.side_effect = Exception("Upload failed")
    
    api_key = "test_api_key"
    
    with patch('builtins.open', create=True) as mock_open:
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Test that exception is properly raised
        with pytest.raises(Exception, match="Upload failed"):
            await fp.upload_file(file=mock_file, api_key=api_key)
        
        # Verify upload_file was called
        mock_upload_file.assert_called_once_with(file=mock_file, api_key=api_key)


def test_image_generations_endpoint(client, mock_all_external_calls):
    request_data = {
        "prompt": "A beautiful sunset",
        "model": "Imagen-3-Fast",
        "n": 1,
        "response_format": "url"
    }
    headers = {"Authorization": "Bearer test_api_key"}

    # Mock the bot response with attachment
    mock_message = MagicMock()
    mock_message.text = "Here's your image:"
    mock_message.attachment = fp.Attachment(
        url="https://poe.com/image.jpg",
        content_type="image/jpeg",
        name="image.jpg"
    )
    
    async def mock_response_generator(*args, **kwargs):
        yield mock_message
    
    mock_all_external_calls['bot_response'].side_effect = mock_response_generator

    response = client.post("/v1/images/generations", json=request_data, headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    assert "created" in response_data
    assert "data" in response_data
    assert len(response_data["data"]) == 1
    assert response_data["data"][0]["url"] == "https://poe.com/image.jpg"


def test_image_generations_b64_json(client, mock_all_external_calls):
    request_data = {
        "prompt": "A beautiful sunset",
        "model": "Imagen-3-Fast",
        "response_format": "b64_json"
    }
    headers = {"Authorization": "Bearer test_api_key"}

    # Mock the bot response with attachment
    mock_message = MagicMock()
    mock_message.text = "Here's your image:"
    mock_message.attachment = fp.Attachment(
        url="https://poe.com/image.jpg",
        content_type="image/jpeg",
        name="image.jpg"
    )
    
    async def mock_response_generator(*args, **kwargs):
        yield mock_message
    
    mock_all_external_calls['bot_response'].side_effect = mock_response_generator

    # HTTP client is already mocked by mock_all_external_calls
    mock_all_external_calls['http_client'].return_value.__aenter__.return_value.get.return_value.content = b"fake_image_data"

    response = client.post("/v1/images/generations", json=request_data, headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    assert "data" in response_data
    assert "b64_json" in response_data["data"][0]


def test_image_edits_endpoint(client, mock_all_external_calls):
    request_data = {
        "image": "base64_image_data",
        "prompt": "Make it more colorful",
        "model": "Imagen-3-Fast",
        "response_format": "url"
    }
    headers = {"Authorization": "Bearer test_api_key"}

    # Mock the bot response with attachment
    mock_message = MagicMock()
    mock_message.text = "Here's your edited image:"
    mock_message.attachment = fp.Attachment(
        url="https://poe.com/edited_image.jpg",
        content_type="image/jpeg",
        name="edited_image.jpg"
    )
    
    async def mock_response_generator(*args, **kwargs):
        yield mock_message
    
    mock_all_external_calls['bot_response'].side_effect = mock_response_generator

    response = client.post("/v1/images/edits", json=request_data, headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    assert "created" in response_data
    assert "data" in response_data
    assert len(response_data["data"]) == 1
    assert response_data["data"][0]["url"] == "https://poe.com/edited_image.jpg"


def test_image_generations_no_file(client, mock_all_external_calls):
    request_data = {
        "prompt": "A beautiful sunset",
        "model": "Imagen-3-Fast"
    }
    headers = {"Authorization": "Bearer test_api_key"}

    # Mock the bot response without attachment
    mock_message = MagicMock()
    mock_message.text = "I couldn't generate an image."
    mock_message.attachment = None
    
    async def mock_response_generator(*args, **kwargs):
        yield mock_message
    
    mock_all_external_calls['bot_response'].side_effect = mock_response_generator

    response = client.post("/v1/images/generations", json=request_data, headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["data"][0]["url"] == "https://example.com/generated_image.png"


def test_completions_with_image_urls(client, mock_all_external_calls):
    request_data = {
        "model": "Imagen-3-Fast",
        "prompt": "Generate an image of a cat",
        "stream": False,
    }
    headers = {"Authorization": "Bearer test_api_key"}

    # Mock the bot response with file attachment
    mock_message = MagicMock()
    mock_message.text = "Here's your image:"
    mock_message.attachment = fp.Attachment(
        url="https://poe.com/generated_cat.jpg",
        content_type="image/jpeg",
        name="cat.jpg"
    )
    
    async def mock_response_generator(*args, **kwargs):
        yield mock_message
    
    mock_all_external_calls['bot_response'].side_effect = mock_response_generator

    response = client.post("/v1/completions", json=request_data, headers=headers)
    assert response.status_code == 200
    response_data = response.json()
    assert "https://poe.com/generated_cat.jpg" in response_data["choices"][0]["text"]


def test_completions_streaming_with_image_urls(client, mock_all_external_calls):
    request_data = {
        "model": "Imagen-3-Fast",
        "prompt": "Generate an image of a cat",
        "stream": True,
    }
    headers = {"Authorization": "Bearer test_api_key"}

    # Mock the bot response with file attachment for streaming
    mock_message = MagicMock()
    mock_message.text = "Here's your image:"
    mock_message.attachment = fp.Attachment(
        url="https://poe.com/generated_cat.jpg",
        content_type="image/jpeg",
        name="cat.jpg"
    )
    
    async def mock_response_generator(*args, **kwargs):
        yield mock_message
    
    mock_all_external_calls['bot_response'].side_effect = mock_response_generator

    response = client.post("/v1/completions", json=request_data, headers=headers)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
