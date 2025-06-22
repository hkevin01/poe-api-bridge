import pytest
import json
from unittest.mock import MagicMock, AsyncMock
from fake_tool_calling import FakeToolCallHandler


@pytest.fixture
def handler():
    """Create a FakeToolCallHandler instance for testing."""
    return FakeToolCallHandler()


@pytest.fixture
def sample_tools():
    """Sample tools for testing."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    message_class = MagicMock()
    message_class.role = "user"
    message_class.content = "What's the weather in San Francisco?"
    return [message_class]


class TestToolPromptBuilding:
    """Test XML prompt building functionality."""
    
    def test_build_tools_xml_single_tool(self, handler, sample_tools):
        """Test building XML for a single tool."""
        single_tool = [sample_tools[0]]
        xml = handler._build_tools_xml(single_tool)
        
        assert "<tools>" in xml
        assert "</tools>" in xml
        assert 'name="get_weather"' in xml
        assert "<description>Get current weather information</description>" in xml
        assert "<parameters>" in xml
        assert "location" in xml
        
    def test_build_tools_xml_multiple_tools(self, handler, sample_tools):
        """Test building XML for multiple tools."""
        xml = handler._build_tools_xml(sample_tools)
        
        assert xml.count('<tool name=') == 2
        assert 'name="get_weather"' in xml
        assert 'name="calculate"' in xml
        
    def test_build_tools_xml_empty(self, handler):
        """Test building XML with no tools."""
        xml = handler._build_tools_xml([])
        assert xml == ""
        
    def test_build_tool_instructions_auto(self, handler):
        """Test building instructions for auto tool choice."""
        instructions = handler._build_tool_instructions("auto")
        assert "Use tools when appropriate" in instructions
        
    def test_build_tool_instructions_none(self, handler):
        """Test building instructions for none tool choice."""
        instructions = handler._build_tool_instructions("none")
        assert "FORBIDDEN from using any tools" in instructions
        
    def test_build_tool_instructions_required(self, handler):
        """Test building instructions for required tool choice."""
        instructions = handler._build_tool_instructions("required")
        assert "MUST use at least one tool" in instructions
        
    def test_build_tool_instructions_specific_function(self, handler):
        """Test building instructions for specific function choice."""
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}
        instructions = handler._build_tool_instructions(tool_choice)
        assert "MUST use the 'get_weather' function" in instructions


class TestToolCallParsing:
    """Test XML tool call parsing functionality."""
    
    def test_parse_single_tool_call(self, handler):
        """Test parsing a single tool call from response."""
        content = """Here's the weather info:
        
<tool_call>
<name>get_weather</name>
<arguments>{"location": "San Francisco", "unit": "celsius"}</arguments>
</tool_call>

Let me get that information for you."""
        
        cleaned_content, tool_calls = handler._parse_tool_calls(content)
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert "San Francisco" in tool_calls[0]["function"]["arguments"]
        assert tool_calls[0]["type"] == "function"
        assert "id" in tool_calls[0]
        assert "<tool_call>" not in cleaned_content
        
    def test_parse_multiple_tool_calls(self, handler):
        """Test parsing multiple tool calls from response."""
        content = """I'll help you with both requests:

<tool_call>
<name>get_weather</name>
<arguments>{"location": "NYC"}</arguments>
</tool_call>

<tool_call>
<name>calculate</name>
<arguments>{"expression": "2 + 2"}</arguments>
</tool_call>

Here are the results."""
        
        cleaned_content, tool_calls = handler._parse_tool_calls(content)
        
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[1]["function"]["name"] == "calculate"
        assert all("id" in call for call in tool_calls)
        
    def test_parse_no_tool_calls(self, handler):
        """Test parsing content with no tool calls."""
        content = "This is a regular response without any tool calls."
        
        cleaned_content, tool_calls = handler._parse_tool_calls(content)
        
        assert len(tool_calls) == 0
        assert cleaned_content == content
        
    def test_parse_invalid_json_arguments(self, handler):
        """Test parsing tool call with invalid JSON arguments."""
        content = """<tool_call>
<name>test_function</name>
<arguments>invalid json here</arguments>
</tool_call>"""
        
        cleaned_content, tool_calls = handler._parse_tool_calls(content)
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["arguments"] == '"invalid json here"'
        
    def test_parse_tool_call_with_whitespace(self, handler):
        """Test parsing tool call with extra whitespace."""
        content = """
<tool_call>
  <name>  get_weather  </name>
  <arguments>  {"location": "Boston"}  </arguments>
</tool_call>
"""
        
        cleaned_content, tool_calls = handler._parse_tool_calls(content)
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert "Boston" in tool_calls[0]["function"]["arguments"]


class TestMessageInjection:
    """Test tool injection into messages."""
    
    def test_inject_tools_no_existing_system_message(self, handler, sample_tools, sample_messages):
        """Test injecting tools when no system message exists."""
        enhanced = handler._inject_tools_into_messages(sample_messages, sample_tools, "auto")
        
        # Should have added a system message at the beginning
        assert len(enhanced) == len(sample_messages) + 1
        assert enhanced[0].role == "system"
        assert "<tools>" in enhanced[0].content
        assert "get_weather" in enhanced[0].content
        
    def test_inject_tools_existing_system_message(self, handler, sample_tools):
        """Test injecting tools when system message already exists."""
        system_msg = MagicMock()
        system_msg.role = "system"
        system_msg.content = "You are a helpful assistant."
        
        user_msg = MagicMock()
        user_msg.role = "user"
        user_msg.content = "Hello"
        
        messages = [system_msg, user_msg]
        enhanced = handler._inject_tools_into_messages(messages, sample_tools, "auto")
        
        # Should not add a new message, just enhance existing
        assert len(enhanced) == len(messages)
        assert enhanced[0].role == "system"
        assert "You are a helpful assistant." in enhanced[0].content
        assert "<tools>" in enhanced[0].content
        
    def test_inject_no_tools(self, handler, sample_messages):
        """Test that no injection happens when no tools provided."""
        enhanced = handler._inject_tools_into_messages(sample_messages, [], "auto")
        assert enhanced == sample_messages
        
    def test_inject_tools_with_different_tool_choices(self, handler, sample_tools, sample_messages):
        """Test tool injection with different tool_choice values."""
        # Test 'none'
        enhanced = handler._inject_tools_into_messages(sample_messages, sample_tools, "none")
        assert "FORBIDDEN from using any tools" in enhanced[0].content
        
        # Test 'required'
        enhanced = handler._inject_tools_into_messages(sample_messages, sample_tools, "required")
        assert "MUST use at least one tool" in enhanced[0].content
        
        # Test specific function
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}
        enhanced = handler._inject_tools_into_messages(sample_messages, sample_tools, tool_choice)
        assert "MUST use the 'get_weather' function" in enhanced[0].content


class TestRegexPattern:
    """Test the regex pattern used for parsing tool calls."""
    
    def test_regex_pattern_basic(self, handler):
        """Test basic regex pattern matching."""
        content = "<tool_call><name>test</name><arguments>{}</arguments></tool_call>"
        matches = handler.tool_call_pattern.findall(content)
        
        assert len(matches) == 1
        assert matches[0] == ("test", "{}")
        
    def test_regex_pattern_multiline(self, handler):
        """Test regex with multiline content."""
        content = """<tool_call>
<name>multiline_test</name>
<arguments>{
  "param": "value",
  "nested": {"key": "val"}
}</arguments>
</tool_call>"""
        
        matches = handler.tool_call_pattern.findall(content)
        
        assert len(matches) == 1
        assert matches[0][0] == "multiline_test"
        assert "nested" in matches[0][1]
        
    def test_regex_pattern_no_match(self, handler):
        """Test regex with no matching content."""
        content = "No tool calls here, just regular text."
        matches = handler.tool_call_pattern.findall(content)
        
        assert len(matches) == 0


class TestUtilityFunctions:
    """Test utility and helper functions."""
    
    def test_tool_call_id_generation(self, handler):
        """Test that tool call IDs are generated correctly."""
        content = "<tool_call><name>test</name><arguments>{}</arguments></tool_call>"
        _, tool_calls = handler._parse_tool_calls(content)
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"].startswith("call_")
        assert len(tool_calls[0]["id"]) == 13  # "call_" + 8 hex chars
        
    def test_multiple_tool_calls_unique_ids(self, handler):
        """Test that multiple tool calls get unique IDs."""
        content = """
<tool_call><name>test1</name><arguments>{}</arguments></tool_call>
<tool_call><name>test2</name><arguments>{}</arguments></tool_call>
"""
        _, tool_calls = handler._parse_tool_calls(content)
        
        assert len(tool_calls) == 2
        assert tool_calls[0]["id"] != tool_calls[1]["id"]
        
    def test_json_validation_in_arguments(self, handler):
        """Test JSON validation in tool call arguments."""
        # Valid JSON
        content1 = '<tool_call><name>test</name><arguments>{"valid": "json"}</arguments></tool_call>'
        _, tool_calls1 = handler._parse_tool_calls(content1)
        assert tool_calls1[0]["function"]["arguments"] == '{"valid": "json"}'
        
        # Invalid JSON gets quoted
        content2 = '<tool_call><name>test</name><arguments>invalid json</arguments></tool_call>'
        _, tool_calls2 = handler._parse_tool_calls(content2)
        assert tool_calls2[0]["function"]["arguments"] == '"invalid json"'
class TestStreamingToolCallFiltering:
    """Test streaming tool call detection and filtering."""
    
    @pytest.mark.asyncio
    async def test_streaming_tool_call_detection(self, handler):
        """Test that streaming properly detects and filters tool calls."""
        
        # Mock streaming chunks that contain a tool call
        mock_chunks = [
            "data: {\"choices\":[{\"delta\":{\"content\":\"I'll help you with that. \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"<tool_call>\"}}]}\n\n", 
            "data: {\"choices\":[{\"delta\":{\"content\":\"<name>get_weather</name>\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"<arguments>{\\\"location\\\": \\\"SF\\\"}</arguments>\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"</tool_call>\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" The weather is sunny.\"}}]}\n\n",
            "data: [DONE]\n\n"
        ]
        
        # Mock the stream_openai_format function
        async def mock_stream(*args, **kwargs):
            for chunk in mock_chunks:
                yield chunk
        
        # Mock server module
        import sys
        from unittest.mock import MagicMock
        mock_server = MagicMock()
        mock_server.stream_openai_format = mock_stream
        sys.modules['server'] = mock_server
        
        # Create mock protocol messages
        mock_messages = [MagicMock()]
        
        # Test the streaming
        result_chunks = []
        async for chunk in handler._stream_tool_aware_response("test-model", mock_messages, "test-key", MagicMock()):
            result_chunks.append(chunk)
        
        # Verify results
        assert len(result_chunks) > 0
        
        # Should have filtered out tool call content
        content_chunks = [chunk for chunk in result_chunks if "I'll help you with that." in chunk]
        assert len(content_chunks) > 0
        
        # Should have tool call chunks
        tool_call_chunks = [chunk for chunk in result_chunks if "tool_calls" in chunk]
        assert len(tool_call_chunks) > 0
        
        # Should not contain raw tool call XML in content
        raw_xml_chunks = [chunk for chunk in result_chunks if "<tool_call>" in chunk and "tool_calls" not in chunk]
        assert len(raw_xml_chunks) == 0
    
    def test_tool_call_boundary_detection(self, handler):
        """Test detection of tool call boundaries in streaming content."""
        
        # Test content with tool call
        content_with_tool = "Hello <tool_call><name>test</name><arguments>{}</arguments></tool_call> world"
        
        # The content should be detected as having tool calls
        _, tool_calls = handler._parse_tool_calls(content_with_tool)
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "test"
    
    def test_partial_tool_call_handling(self, handler):
        """Test handling of partial tool calls in streaming."""
        
        # Test various partial states
        partial_states = [
            "<tool",
            "<tool_call",
            "<tool_call>",
            "<tool_call><name>test",
            "<tool_call><name>test</name><arg",
        ]
        
        for partial in partial_states:
            # Partial tool calls should not be parsed as complete
            _, tool_calls = handler._parse_tool_calls(partial)
            assert len(tool_calls) == 0, f"Partial state '{partial}' should not parse as complete tool call"