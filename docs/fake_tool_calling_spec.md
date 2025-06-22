# Fake Tool Calling Specification

## Overview
Modular fake tool calling for Poe API bridge using prompt engineering. Provides full OpenAI tools API compatibility while being easily removable.

## Components

### File Structure
```
fake_tool_calling.py    # Single file implementation
```

### Core Classes

**`FakeToolCallHandler`** - Main entry point
- `process_request()` - Handle tool-enhanced requests
- `inject_tools_prompt()` - Add tool definitions to system prompt
- `parse_tool_calls()` - Extract XML tool calls and format response

## Integration

### Request Model Enhancement
Extend existing [`ChatCompletionRequest`](../server.py:68) with optional fields:
```python
# Add to existing ChatCompletionRequest class
tools: Optional[List[Dict[str, Any]]] = None
tool_choice: Optional[Union[str, Dict[str, Any]]] = None
```

### Endpoint Modification
```python
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key: str):
    if request.tools:
        return await FakeToolCallHandler().process_request(request, api_key)
    # ... existing logic unchanged (no tool parsing when tools=None)
```

## XML Format

### Prompt Injection
```xml
<tools>
<tool name="function_name">
<description>Function description</description>
<parameters>{"type": "object", "properties": {...}}</parameters>
</tool>
</tools>

To use tools, respond with:
<tool_call>
<name>function_name</name>
<arguments>{"param": "value"}</arguments>
</tool_call>
```

### Response Parsing
Extract `<tool_call>` blocks and convert to OpenAI `tool_calls` format.

## OpenAI Compatibility

### Supported Features
- ✅ `tools` parameter
- ✅ `tool_choice`: `"none"`, `"auto"`, `"required"`, specific function
- ✅ Parallel tool calls
- ✅ Streaming support
- ✅ Standard OpenAI response format

### Example Response
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_123",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"SF\"}"
          }
        }
      ]
    }
  }]
}
```

## Testing
- Unit tests in [`test_server.py`](../test_server.py)
- E2E verification in `verify_chat_completions_tool_calling.py`
- OpenAI API compatibility validation