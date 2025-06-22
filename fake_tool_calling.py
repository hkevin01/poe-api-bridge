"""
Fake Tool Calling Implementation for Poe API Bridge

Provides OpenAI tools API compatibility using prompt engineering and XML parsing.
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Union, Iterator
import asyncio

import fastapi_poe as fp
from fastapi.responses import StreamingResponse


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

        # Import here to avoid circular imports
        from server import normalize_model, normalize_role, count_message_tokens

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
                self._stream_tool_aware_response(model, poe_messages, api_key, request),
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
        self, messages: List, tools: List[Dict], tool_choice: Optional[Union[str, Dict]]
    ) -> List:
        """Inject tool definitions into the system message."""
        if not tools:
            return messages

        # Build tools section
        tools_xml = self._build_tools_xml(tools)

        # Build instructions based on tool_choice
        instructions = self._build_tool_instructions(tool_choice)

        # Create enhanced system message
        tool_prompt = f"""
{tools_xml}

{instructions}

When using tools, respond with XML in this exact format:
<tool_call>
<name>function_name</name>
<arguments>{{"param": "value"}}</arguments>
</tool_call>

You can make multiple tool calls by using multiple <tool_call> blocks.
"""

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
            from server import ChatCompletionMessage

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

    def _build_tool_instructions(self, tool_choice: Optional[Union[str, Dict]]) -> str:
        """Build tool usage instructions based on tool_choice."""
        if tool_choice == "none":
            return "IMPORTANT: You are FORBIDDEN from using any tools. Do NOT use <tool_call> tags. Respond directly with natural language only. Never format your response with XML tool calls."
        elif tool_choice == "required":
            return "You MUST use at least one tool to answer this request. Do not respond without making a tool call."
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

        # Extract tool calls
        for match in matches:
            name = match.group(1).strip()
            arguments_str = match.group(2).strip()

            # Generate unique ID for this tool call
            call_id = f"call_{uuid.uuid4().hex[:8]}"

            # Validate JSON arguments
            try:
                json.loads(arguments_str)  # Validate JSON
            except json.JSONDecodeError:
                # If invalid JSON, wrap in quotes as fallback
                arguments_str = f'"{arguments_str}"'

            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments_str},
                }
            )

        # Remove tool call XML from content
        cleaned_content = self.tool_call_pattern.sub("", content).strip()

        return cleaned_content, tool_calls

    async def _generate_tool_aware_response(
        self, model: str, messages: List[fp.ProtocolMessage], api_key: str, request
    ) -> Dict[str, Any]:
        """Generate a non-streaming response with tool call parsing."""
        from server import generate_poe_bot_response, count_message_tokens

        # Get response from Poe
        response_content = await generate_poe_bot_response(model, messages, api_key)

        # Handle case where generate_poe_bot_response returns a dict instead of string
        if isinstance(response_content, dict):
            # Extract text content from the response dict
            response_text = (
                response_content.get("content", "")
                or response_content.get("text", "")
                or str(response_content)
            )
        else:
            response_text = response_content or ""

        # Parse for tool calls
        cleaned_content, tool_calls = self._parse_tool_calls(response_text)

        # Count tokens
        token_counts = count_message_tokens(messages, model)
        completion_tokens = len(response_text.split()) if response_text else 0

        # Build OpenAI-compatible response
        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": cleaned_content if cleaned_content else None,
            },
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }

        # Add tool calls if present
        if tool_calls:
            choice["message"]["tool_calls"] = tool_calls

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": model,
            "choices": [choice],
            "usage": {
                "prompt_tokens": token_counts.get("prompt_tokens", 0),
                "completion_tokens": completion_tokens,
                "total_tokens": token_counts.get("total_tokens", 0) + completion_tokens,
            },
        }

    async def _stream_tool_aware_response(
        self, model: str, messages: List[fp.ProtocolMessage], api_key: str, request
    ) -> Iterator[str]:
        """Generate a streaming response with real-time tool call filtering."""
        from server import stream_openai_format

        # State for tool call detection and filtering
        content_buffer = ""
        tool_call_buffer = ""
        in_tool_call = False

        async for chunk in stream_openai_format(model, messages, api_key):
            if chunk.strip():
                try:
                    # Convert bytes to string if needed
                    if isinstance(chunk, bytes):
                        chunk_str = chunk.decode("utf-8")
                    else:
                        chunk_str = chunk

                    # Extract content from chunk
                    content_delta = ""
                    modified_chunk = chunk_str

                    if chunk_str.startswith("data: "):
                        chunk_data = chunk_str[6:].strip()
                        if chunk_data and chunk_data != "[DONE]":
                            try:
                                parsed = json.loads(chunk_data)
                                delta = parsed.get("choices", [{}])[0].get("delta", {})
                                content_delta = delta.get("content", "")

                                if content_delta:
                                    content_buffer += content_delta

                                    # Process character by character to detect tool calls
                                    filtered_content = ""
                                    i = 0
                                    while i < len(content_delta):
                                        char = content_delta[i]

                                        if not in_tool_call:
                                            # Check if we're starting a tool call
                                            temp_buffer = content_buffer
                                            if "<tool_call>" in temp_buffer:
                                                # Found start of tool call
                                                start_pos = temp_buffer.rfind(
                                                    "<tool_call>"
                                                )
                                                # Keep content before tool call
                                                before_tool = temp_buffer[:start_pos]
                                                if len(before_tool) > len(
                                                    content_buffer
                                                ) - len(content_delta):
                                                    # This content was in the current delta
                                                    chars_before = start_pos - (
                                                        len(content_buffer)
                                                        - len(content_delta)
                                                    )
                                                    if chars_before > 0:
                                                        filtered_content += (
                                                            content_delta[:chars_before]
                                                        )

                                                in_tool_call = True
                                                tool_call_buffer = temp_buffer[
                                                    start_pos:
                                                ]
                                                break
                                            else:
                                                filtered_content += char
                                        else:
                                            # We're in a tool call, add to buffer and check for end
                                            tool_call_buffer += char
                                            if tool_call_buffer.endswith(
                                                "</tool_call>"
                                            ):
                                                # Tool call complete
                                                _, tool_calls = self._parse_tool_calls(
                                                    tool_call_buffer
                                                )

                                                if tool_calls:
                                                    # Emit tool call chunks
                                                    for idx, tool_call in enumerate(
                                                        tool_calls
                                                    ):
                                                        tool_chunk = {
                                                            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                                                            "object": "chat.completion.chunk",
                                                            "created": int(
                                                                asyncio.get_event_loop().time()
                                                            ),
                                                            "model": model,
                                                            "choices": [
                                                                {
                                                                    "index": 0,
                                                                    "delta": {
                                                                        "tool_calls": [
                                                                            {
                                                                                "index": idx,
                                                                                "id": tool_call[
                                                                                    "id"
                                                                                ],
                                                                                "type": "function",
                                                                                "function": {
                                                                                    "name": tool_call[
                                                                                        "function"
                                                                                    ][
                                                                                        "name"
                                                                                    ],
                                                                                    "arguments": tool_call[
                                                                                        "function"
                                                                                    ][
                                                                                        "arguments"
                                                                                    ],
                                                                                },
                                                                            }
                                                                        ]
                                                                    },
                                                                    "finish_reason": None,
                                                                }
                                                            ],
                                                        }
                                                        yield f"data: {json.dumps(tool_chunk)}\n\n"

                                                    # Final chunk
                                                    final_chunk = {
                                                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                                                        "object": "chat.completion.chunk",
                                                        "created": int(
                                                            asyncio.get_event_loop().time()
                                                        ),
                                                        "model": model,
                                                        "choices": [
                                                            {
                                                                "index": 0,
                                                                "delta": {},
                                                                "finish_reason": "tool_calls",
                                                            }
                                                        ],
                                                    }
                                                    yield f"data: {json.dumps(final_chunk)}\n\n"
                                                    yield "data: [DONE]\n\n"
                                                    return

                                                # Reset state and continue
                                                in_tool_call = False
                                                tool_call_buffer = ""

                                        i += 1

                                    # Update chunk with filtered content
                                    if (
                                        not in_tool_call
                                        and filtered_content != content_delta
                                    ):
                                        parsed["choices"][0]["delta"][
                                            "content"
                                        ] = filtered_content
                                        modified_chunk = (
                                            f"data: {json.dumps(parsed)}\n\n"
                                        )

                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass

                    # Only yield chunk if we're not filtering tool call content
                    if not in_tool_call or chunk_str.endswith("[DONE]\n\n"):
                        yield modified_chunk

                except UnicodeDecodeError:
                    yield chunk

        # Final done if not already sent
        if not in_tool_call:
            yield "data: [DONE]\n\n"
