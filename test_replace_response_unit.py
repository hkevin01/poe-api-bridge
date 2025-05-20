import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
import fastapi_poe as fp

@pytest.mark.asyncio
async def test_stream_response_with_replace():
    """Test that stream_response correctly handles is_replace_response=True."""
    # Import the function under test
    from server import stream_response
    
    # Create mock messages for testing
    messages = []
    
    # Create a logger mock
    logger_mock = MagicMock()
    logger_mock.debug = MagicMock()
    
    # Mock the get_logger function to return our mock logger
    with patch("server.get_logger", return_value=logger_mock):
        # Set up our response generator to yield messages with is_replace_response
        async def mock_get_bot_response(*args, **kwargs):
            # First message - normal
            msg1 = fp.PartialResponse(text="Initial content")
            yield msg1
            
            # Second message - with replace flag
            msg2 = fp.PartialResponse(text="Replacement content", is_replace_response=True)
            yield msg2
            
            # Third message - normal append
            msg3 = fp.PartialResponse(text=" with addition")
            yield msg3
        
        # Mock the get_bot_response function
        with patch("server.get_bot_response", mock_get_bot_response):
            # Test with chat format
            chunks = []
            async for chunk_bytes in stream_response("test-model", messages, "test-key", "chat"):
                chunk_str = chunk_bytes.decode("utf-8")
                if chunk_str.startswith("data: ") and chunk_str.strip() != "data: [DONE]":
                    chunk_data = json.loads(chunk_str.replace("data: ", ""))
                    chunks.append(chunk_data)
            
            # Assert that we got 3 chunks (plus possibly a final one)
            assert len(chunks) >= 3
            
            # Check for reset behavior in the chunks
            content_chunks = []
            for chunk in chunks:
                if "choices" in chunk and chunk["choices"][0].get("delta", {}).get("content") is not None:
                    content_chunks.append(chunk["choices"][0]["delta"]["content"])
            
            # We should have received these content chunks in order
            assert content_chunks[0] == "Initial content"
            assert content_chunks[1] == "Replacement content"
            assert content_chunks[2] == " with addition"
            
            # Verify the logger was called with the expected debug message
            logger_mock.debug.assert_any_call("Replacing response with: Replacement content")

@pytest.mark.asyncio
async def test_generate_poe_bot_response_with_replace():
    """Test that generate_poe_bot_response correctly handles is_replace_response=True."""
    # Import the function under test
    from server import generate_poe_bot_response
    
    # Create mock messages for testing
    messages = []
    
    # Create a logger mock
    logger_mock = MagicMock()
    logger_mock.info = MagicMock()
    logger_mock.debug = MagicMock()
    
    # Mock the get_logger function to return our mock logger
    with patch("server.get_logger", return_value=logger_mock):
        # Set up our response generator to yield messages with is_replace_response
        async def mock_get_bot_response(*args, **kwargs):
            # First message - normal
            msg1 = fp.PartialResponse(text="Initial content")
            yield msg1
            
            # Second message - with replace flag
            msg2 = fp.PartialResponse(text="Replacement content", is_replace_response=True)
            yield msg2
            
            # Third message - normal append
            msg3 = fp.PartialResponse(text=" with addition")
            yield msg3
        
        # Mock the get_bot_response function and other necessary functions
        with patch("server.get_bot_response", mock_get_bot_response), \
             patch("server.get_bot_query_base_url", return_value="https://api.example.com"):
            
            # Call the function
            response = await generate_poe_bot_response("test-model", messages, "test-key")
            
            # Verify the response content
            assert response["role"] == "assistant"
            assert response["content"] == "Replacement content with addition"
            
            # Verify the logger was called with the expected debug message
            logger_mock.debug.assert_any_call(ANY)  # Using ANY for the request_id part of the message


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_stream_response_with_replace())
    asyncio.run(test_generate_poe_bot_response_with_replace())
