"""
Fake Tool Calling Implementation for Poe API Bridge

Provides OpenAI tools API compatibility using prompt engineering and XML parsing.
"""

import asyncio
import json
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import fastapi_poe as fp
from fastapi.responses import StreamingResponse

# Import only safe types/constants at the top
from server import normalize_model, normalize_role


class FakeToolCallHandler:
    """Main handler for fake tool calling functionality."""

    def __init__(self):
        # Regex pattern to extract tool calls from responses
        self.tool_call_pattern = re.compile(
            r"<tool_call>\s*<name>([^<]+)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>",
            re.DOTALL,
        )

    async def process_request(
        self, request, api_key: str
    ) -> Union[Dict[str, Any], StreamingResponse]:
        """
        Process a chat completion request with tools.

        Args:
            request: ChatCompletionRequest with tools
            api_key: Poe API key

        Returns:
            OpenAI-compatible response or StreamingResponse
        """
        # Inject tools into the system prompt
        enhanced_messages = self._inject_tools_into_messages(
            request.messages, request.tools, request.tool_choice
        )

        model = normalize_model(request.model)

        # Convert messages to Poe format
        poe_messages = []
        for msg in enhanced_messages:
            role = normalize_role(msg.role)
            # Handle different content types
            if isinstance(msg.content, str):
                content = msg.content or ""
            elif isinstance(msg.content, dict):
                content = str(msg.content)
            else:
                content = ""
            poe_messages.append(fp.ProtocolMessage(role=role, content=content))

        if request.stream:
            # Handle streaming response
            return StreamingResponse(
                self._stream_tool_aware_response(
                    model, poe_messages, api_key, request
                ),
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Transfer-Encoding": "chunked",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Handle regular response
            return await self._generate_tool_aware_response(
                model, poe_messages, api_key, request
            )

    def _inject_tools_into_messages(
        self, messages: List, tools: List[Dict], 
        tool_choice: Optional[Union[str, Dict]]
    ) -> List:
        """Inject tool definitions into the system message."""
        if not tools:
            return messages

        # Build tools section
        tools_xml = self._build_tools_xml(tools)

        # Build instructions based on tool_choice
        instructions = self._build_tool_instructions(tool_choice)

        # Create enhanced system message
        tool_prompt = (
            f"""
{tools_xml}

{instructions}

When using tools, respond with XML in this exact format:
<tool_call>
<name>function_name</name>
<arguments>{{"param": "value"}}</arguments>
</tool_call>

You can make multiple tool calls by using multiple <tool_call> blocks.
"""
        )

        # Separate system message from other messages
        system_content = None
        non_system_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                non_system_messages.append(msg)

        # Create the combined system message: tools first, then user system content
        if system_content:
            combined_system_content = f"{tool_prompt}\n\n{system_content}"
        else:
            combined_system_content = tool_prompt

        # Import ChatCompletionMessage only here to avoid circular import
        from server import ChatCompletionMessage

        # Create a new system message
        if messages:
            # Use the first message as a template for structure
            first_msg = messages[0]
            system_msg = type(first_msg)(
                **{
                    **first_msg.__dict__,
                    "role": "system",
                    "content": combined_system_content,
                }
            )
        else:
            # Fallback if no messages (shouldn't happen)
            system_msg = ChatCompletionMessage(
                role="system", content=combined_system_content
            )

        # Build final message list: system message first, then all non-system messages
        enhanced_messages = [system_msg] + non_system_messages

        return enhanced_messages

    def _build_tools_xml(self, tools: List[Dict]) -> str:
        """Build XML representation of tools."""
        if not tools:
            return ""

        tools_parts = ["<tools>"]

        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                name = func.get("name", "")
                description = func.get("description", "")
                parameters = func.get("parameters", {})

                tools_parts.append(f'<tool name="{name}">')
                if description:
                    tools_parts.append(f"<description>{description}</description>")
                if parameters:
                    params_json = json.dumps(parameters, separators=(",", ":"))
                    tools_parts.append(f"<parameters>{params_json}</parameters>")
                tools_parts.append("</tool>")

        tools_parts.append("</tools>")
        return "\n".join(tools_parts)

    def _build_tool_instructions(
        self, tool_choice: Optional[Union[str, Dict]]
    ) -> str:
        """Build tool usage instructions based on tool_choice."""
        if tool_choice == "none":
            return (
                "IMPORTANT: You are FORBIDDEN from using any tools. "
                "Do NOT use <tool_call> tags. Respond directly with natural language only. "
                "Never format your response with XML tool calls."
            )
        elif tool_choice == "required":
            return (
                "You MUST use at least one tool to answer this request. "
                "Do not respond without making a tool call."
            )
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            func_name = tool_choice.get("function", {}).get("name", "")
            return f"You MUST use the '{func_name}' function to answer this request."
        else:  # "auto" or None
            return "Use tools when appropriate to help answer the user's request."

    def _parse_tool_calls(self, content: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Parse tool calls from response content.

        Returns:
            (cleaned_content, tool_calls_list)
        """
        tool_calls = []

        # Find all tool call matches
        matches = list(self.tool_call_pattern.finditer(content))

        if not matches:
            return content, []

        # Process matches and build tool calls
        for i, match in enumerate(matches):
            tool_name = match.group(1).strip()
            arguments_str = match.group(2).strip()

            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a simple string argument
                arguments = {"raw_content": arguments_str}

            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments, separators=(",", ":")),
                },
            }
            tool_calls.append(tool_call)

        # Remove tool calls from content
        cleaned_content = self.tool_call_pattern.sub("", content).strip()

        return cleaned_content, tool_calls

    async def _generate_tool_aware_response(
        self, model: str, messages: List[fp.ProtocolMessage], api_key: str, request
    ) -> Dict[str, Any]:
        """
        Generate a non-streaming response with tool calling support.

        Args:
            model: Model name
            messages: List of Poe protocol messages
            api_key: Poe API key
            request: Original request object

        Returns:
            OpenAI-compatible response dict
        """
        try:
            from server import generate_poe_bot_response
            response = await generate_poe_bot_response(model, messages, api_key)

            # Parse tool calls from response
            content = response.get("content", "")
            cleaned_content, tool_calls = self._parse_tool_calls(content)

            # Build OpenAI-compatible response
            response_data = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": cleaned_content if cleaned_content else None,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

            # Add tool calls if present
            if tool_calls:
                response_data["choices"][0]["message"]["tool_calls"] = tool_calls
                response_data["choices"][0]["finish_reason"] = "tool_calls"

            return response_data

        except Exception as e:
            # Handle errors
            return {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                }
            }

    async def _stream_tool_aware_response(
        self, model: str, messages: List[fp.ProtocolMessage], api_key: str, request
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response with tool calling support.

        Args:
            model: Model name
            messages: List of Poe protocol messages
            api_key: Poe API key
            request: Original request object

        Yields:
            Server-sent events in OpenAI format
        """
        try:
            from server import stream_openai_format
            async for chunk in stream_openai_format(model, messages, api_key):
                # Convert chunk to string if it's bytes
                if isinstance(chunk, bytes):
                    chunk_str = chunk.decode('utf-8')
                else:
                    chunk_str = str(chunk)
                
                # Parse tool calls from chunk if it contains content
                if "data: " in chunk_str:
                    # Extract the JSON data from the chunk
                    if chunk_str.startswith("data: "):
                        data_part = chunk_str[6:]  # Remove "data: " prefix
                        if data_part.strip() == "[DONE]":
                            yield chunk_str
                            continue

                        try:
                            # Parse the JSON data
                            data = json.loads(data_part)
                            
                            # Check if this chunk contains content
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    content = choice["delta"]["content"]
                                    if content:
                                        # Parse tool calls from content
                                        cleaned_content, tool_calls = self._parse_tool_calls(
                                            content
                                        )
                                        
                                        # Update the content in the chunk
                                        choice["delta"]["content"] = cleaned_content
                                        
                                        # Add tool calls if present
                                        if tool_calls:
                                            choice["delta"]["tool_calls"] = tool_calls
                                        
                                        # Re-serialize the chunk
                                        yield f"data: {json.dumps(data, separators=(',', ':'))}\n\n"
                                        continue
                        except (json.JSONDecodeError, KeyError):
                            # If parsing fails, yield the original chunk
                            pass
                
                yield chunk_str

        except Exception as e:
            # Handle errors
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                }
            }
            yield f"data: {json.dumps(error_data, separators=(',', ':'))}\n\n"
            yield "data: [DONE]\n\n"

    async def _stream_tool_aware_response_legacy(
        self, model: str, messages: List[fp.ProtocolMessage], api_key: str, request
    ) -> AsyncGenerator[str, None]:
        """
        Legacy streaming response method (kept for reference).

        Args:
            model: Model name
            messages: List of Poe protocol messages
            api_key: Poe API key
            request: Original request object

        Yields:
            Server-sent events in OpenAI format
        """
        try:
            from server import generate_poe_bot_response
            response = await generate_poe_bot_response(model, messages, api_key)

            # Parse tool calls from response
            content = response.get("content", "")
            cleaned_content, tool_calls = self._parse_tool_calls(content)

            # Stream the response
            if cleaned_content:
                # Stream content chunks
                chunk_size = 10
                for i in range(0, len(cleaned_content), chunk_size):
                    chunk = cleaned_content[i:i + chunk_size]
                    data = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data, separators=(',', ':'))}\n\n"
                    await asyncio.sleep(0.01)  # Small delay for streaming effect

            # Add tool calls if present
            if tool_calls:
                data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": tool_calls},
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
                yield f"data: {json.dumps(data, separators=(',', ':'))}\n\n"

            # Final chunk
            data = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop" if not tool_calls else "tool_calls",
                    }
                ],
            }
            yield f"data: {json.dumps(data, separators=(',', ':'))}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            # Handle errors
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                }
            }
            yield f"data: {json.dumps(error_data, separators=(',', ':'))}\n\n"
            yield "data: [DONE]\n\n"
