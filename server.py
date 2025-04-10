# Standard library imports
import asyncio
import functools
import json
import logging
import os
import time
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third-party imports
import fastapi_poe as fp
import httpx
import tiktoken
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi_poe.client import get_bot_response
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("poe-openai-proxy")

app = FastAPI()


# Add request logging middleware with simplified logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    path = request.url.path
    method = request.method
    request_id = os.urandom(4).hex()

    logger.info(f"[{request_id}] Request received: {method} {path}")

    try:
        response = await call_next(request)

        process_time = (time.time() - start_time) * 1000
        formatted_process_time = f"{process_time:.2f}"
        status_code = response.status_code

        if status_code >= 400:
            logger.warning(
                f"[{request_id}] Error response: {method} {path} -> {status_code} (took {formatted_process_time} ms)"
            )
        else:
            logger.info(
                f"[{request_id}] Response: {method} {path} -> {status_code} (took {formatted_process_time} ms)"
            )

        return response
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        formatted_process_time = f"{process_time:.2f}"

        logger.exception(
            f"[{request_id}] Unhandled exception in {method} {path} (took {formatted_process_time} ms): {str(e)}"
        )

        # Re-raise the exception for FastAPI's exception handlers
        raise


# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

models = (
    "Claude-3.5-Sonnet",
    "Claude-3.7-Sonnet",
    "gpt-4o",
)

models_mapping = {"poe-cursor-model": "Claude-3.5-Sonnet"}


class ChatMessage(BaseModel):
    role: str  # role: the role of the message, either system, user, or assistant
    content: str


class ChatCompletionMessage(BaseModel):
    role: str
    content: Optional[Any] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    seed: Optional[int] = None
    response_format: Optional[Dict[str, str]] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[int, float]] = None
    user: Optional[str] = None


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None


class ModerationRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = "text-moderation-latest"


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"


class ErrorResponse(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


# Add a custom exception class for Poe API errors
class PoeAPIError(Exception):
    """Custom exception for Poe API errors."""

    def __init__(self, message, error_data=None, status_code=500, error_id=None):
        self.message = message
        self.error_data = error_data
        self.status_code = status_code
        self.error_id = error_id
        super().__init__(self.message)


def create_error_response(
    message: str, error_type: str, status_code: int, param: Optional[str] = None
) -> HTTPException:
    error_types = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
        500: "server_error",
    }
    error = {
        "message": message,
        "type": error_type or error_types.get(status_code, "server_error"),
    }
    if param:
        error["param"] = param
    return HTTPException(status_code=status_code, detail=error)


def normalize_model(model: str):
    # trim any whitespace from the model name
    model = model.strip()

    mappings_lowercase = {k.lower(): v for k, v in models_mapping.items()}

    if model.lower() in mappings_lowercase:
        model = mappings_lowercase[model.lower()]

    models_lowercase = [m.lower() for m in models]

    if model.lower() not in models_lowercase:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    model_index = models_lowercase.index(model.lower())

    return models[model_index]


# Custom HTTP Bearer authentication that returns 401 like OpenAI
class CustomHTTPBearer(HTTPBearer):
    async def __call__(
        self, request: Request
    ) -> Optional[HTTPAuthorizationCredentials]:
        authorization = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "Authentication error: No token provided",
                        "type": "authentication_error",
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            scheme, credentials = authorization.split()
            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": {
                            "message": f"Authentication error: Invalid scheme '{scheme}' - must be 'Bearer'",
                            "type": "authentication_error",
                        }
                    },
                    headers={"WWW-Authenticate": "Bearer"},
                )
        except ValueError:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "Authentication error: Malformed Authorization header",
                        "type": "authentication_error",
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        return HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)


security = CustomHTTPBearer(bearerFormat="Bearer", description="Your API key")


async def get_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Extracts and validates the API key from the authorization header"""
    return credentials.credentials


def normalize_role(role: str):
    if role == "user":
        return "user"
    elif role == "assistant":
        return "bot"
    elif role == "system":
        return "system"
    else:
        return role


def parse_poe_error(error: Exception) -> tuple[str, dict, str, str]:
    """
    Parse error information from Poe API errors.

    Returns a tuple of:
        - Error message
        - Error data (JSON object or None)
        - Error type
        - Error ID (if available)
    """
    error_message = str(error)
    error_data = None
    error_type = "server_error"
    error_id = None

    try:
        error_str = str(error)

        # Case 1: Error is a JSON object string
        if error_str.startswith("{") and error_str.endswith("}"):
            error_data = json.loads(error_str)
            error_message = error_data.get("text", error_str)
            error_type = "poe_api_error"

        # Case 2: Error is BotError format with embedded JSON
        elif "BotError('" in error_str and "')" in error_str:
            json_part = error_str.split("BotError('", 1)[1].rsplit("')", 1)[0]
            try:
                error_data = json.loads(json_part)
                error_message = error_data.get("text", error_str)
                error_type = "poe_api_error"
            except json.JSONDecodeError:
                pass

        # Extract error_id if available
        if "error_id:" in error_message:
            try:
                error_id = error_message.split("error_id:", 1)[1].strip().rstrip(")")
            except Exception:
                pass

        # Determine error type based on message content
        if isinstance(error, ValueError) and "Model" in error_str:
            error_type = "model_not_found"
        elif "Internal server error" in error_message:
            error_type = "poe_server_error"

    except json.JSONDecodeError:
        pass
    except Exception:
        # If any unexpected error occurs during parsing, use the original error message
        pass

    return error_message, error_data, error_type, error_id


def count_tokens(text: str, model: str = None) -> int:
    """Count the number of tokens in a string using the tiktoken library

    Uses cl100k_base tokenizer for all models for consistency and simplicity.
    """
    try:
        # Use cl100k_base tokenizer for all models (used by OpenAI and compatible with Claude)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        # Return an approximation if tiktoken fails
        return len(text) // 4


def count_message_tokens(
    messages: List[fp.ProtocolMessage], model: str = None
) -> Dict[str, int]:
    """Count tokens in a list of messages and return prompt and completion token counts

    Uses a consistent approach for all models.
    """
    prompt_tokens = 0
    completion_tokens = 0

    for msg in messages:
        # Count each message based on its role
        msg_content = msg.content if hasattr(msg, "content") else ""
        token_count = count_tokens(msg_content)

        if msg.role == "bot" or msg.role == "assistant":
            completion_tokens += token_count
        else:
            prompt_tokens += token_count

    # Add a small overhead for formatting (consistent with OpenAI's approach)
    prompt_tokens += 3

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
@app.post("//v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, api_key: str = Depends(get_api_key)
):
    request_id = os.urandom(4).hex()
    logger.info(
        f"[{request_id}] Processing chat completion request for model: {request.model}"
    )

    try:
        # Prepare messages for the API call
        messages = []
        for msg in request.messages:
            role = normalize_role(msg.role)

            content = msg.content or ""

            if isinstance(content, list):
                # Join list elements into a single string
                parts = []
                for comp in content:
                    if isinstance(comp, dict):
                        if comp.get("type") == "text" and "text" in comp:
                            parts.append(comp["text"])
                        elif comp.get("type") == "image":
                            parts.append(f"[Image: {comp.get('image_url', '')}]")
                content = " ".join(parts)

            messages.append(fp.ProtocolMessage(role=role, content=content))

        # If streaming is requested, use StreamingResponse
        if request.stream:
            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
                "X-Accel-Buffering": "no",
            }
            return StreamingResponse(
                stream_openai_format(request.model, messages, api_key),
                headers=headers,
                media_type="text/event-stream",
            )

        # For non-streaming, accumulate the full response
        response = await generate_poe_bot_response(request.model, messages, api_key)

        # Calculate token counts
        token_counts = count_message_tokens(messages)
        response_tokens = count_tokens(response.get("content", ""))
        token_counts["completion_tokens"] = response_tokens
        token_counts["total_tokens"] = token_counts["prompt_tokens"] + response_tokens

        # Set finish reason to stop
        finish_reason = "stop"

        completion_response = {
            "id": "chatcmpl-" + os.urandom(12).hex(),
            "object": "chat.completion",
            "system_fingerprint": "fp_" + os.urandom(12).hex(),
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.get("content"),
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": token_counts,
        }

        return completion_response

    except Exception as e:
        logger.exception(f"Error in chat_completions: {str(e)}")

        # Default error values
        status_code = 500
        error_message = str(e)
        error_type = "server_error"

        # Handle different error types
        if isinstance(e, HTTPException):
            raise e
        elif isinstance(e, PoeAPIError):
            # Use the structured error data from Poe
            error_message = e.message
            error_type = "poe_api_error"
            status_code = e.status_code
            # Include the original error data if available
            if e.error_data:
                error_detail = {
                    "error": {
                        "message": error_message,
                        "type": error_type,
                        "poe_error": e.error_data,
                    }
                }
                # Add error_id if available
                if e.error_id:
                    error_detail["error"]["error_id"] = e.error_id
                raise HTTPException(status_code=status_code, detail=error_detail)
        elif isinstance(e, ValueError):
            if "Model" in str(e):
                status_code = 404
                error_type = "invalid_request_error"
                error_message = str(e)
            else:
                status_code = 400
                error_type = "invalid_request_error"
        else:
            # Use the helper function to parse error information
            error_message, error_data, error_type, error_id = parse_poe_error(e)

            if error_data:
                error_detail = {
                    "error": {
                        "message": error_message,
                        "type": error_type,
                        "poe_error": error_data,
                    }
                }

                if error_id:
                    error_detail["error"]["error_id"] = error_id

                raise HTTPException(status_code=status_code, detail=error_detail)

        # Default error response
        raise HTTPException(
            status_code=status_code,
            detail={"error": {"message": error_message, "type": error_type}},
        )


@app.get("/models")
@app.get("/v1/models")
@app.get("//v1/models")  # Handle double slash case like other endpoints
async def list_models_openai():
    combined_models = list(models) + list(models_mapping.keys())

    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "poe",
                "permission": [
                    {
                        "id": "modelperm-" + os.urandom(12).hex(),
                        "object": "model_permission",
                        "created": int(time.time()),
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    }
                ],
                "root": model,
                "parent": None,
            }
            for model in combined_models
        ],
    }


async def create_stream_chunk(
    message_text: str, model: str, format_type: str, is_first_chunk: bool = False
):
    """Common function to create streaming response chunks"""
    chunk_id = os.urandom(12).hex()
    timestamp = int(time.time())

    if format_type == "completion":
        return {
            "id": f"cmpl-{chunk_id}",
            "object": "text_completion",
            "created": timestamp,
            "model": model,
            "choices": [
                {
                    "text": message_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
    elif format_type == "chat":
        return {
            "id": f"chatcmpl-{chunk_id}",
            "system_fingerprint": "fp_" + os.urandom(12).hex(),
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        **({"role": "assistant"} if is_first_chunk else {}),
                        **({"content": message_text} if message_text else {}),
                    },
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
        }
    else:  # poe format
        return {"response": message_text, "done": False}


async def create_final_chunk(
    model: str, format_type: str, token_counts: Optional[Dict[str, int]] = None
):
    """Common function to create final streaming chunks"""
    chunk_id = os.urandom(12).hex()
    timestamp = int(time.time())

    if format_type == "completion":
        result = {
            "id": f"cmpl-{chunk_id}",
            "object": "text_completion",
            "created": timestamp,
            "model": model,
            "choices": [
                {"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}
            ],
        }
        if token_counts:
            result["usage"] = token_counts
        return result
    elif format_type == "chat":
        result = {
            "id": f"chatcmpl-{chunk_id}",
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": model,
            "choices": [
                {"index": 0, "logprobs": None, "delta": {}, "finish_reason": "stop"}
            ],
        }
        if token_counts:
            result["usage"] = token_counts
        return result
    else:  # poe format
        result = {"response": "", "done": True}
        if token_counts:
            result["usage"] = token_counts
        return result


@functools.lru_cache()
def get_bot_query_base_url() -> str:
    """
    Get the base URL for bot queries from environment variables.
    Falls back to default if not specified.
    """
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Get the base URL from environment or use default
    return os.getenv("BOT_QUERY_API_BASE_URL", "https://api.poe.com/bot/")


async def stream_response(
    model: str, messages: list[fp.ProtocolMessage], api_key: str, format_type: str
):
    """Common streaming function for all response types"""
    model = normalize_model(model)
    first_chunk = True
    accumulated_response = ""

    # Calculate prompt tokens before starting stream
    token_counts = count_message_tokens(messages)

    try:
        async for message in get_bot_response(
            messages=messages,
            bot_name=model,
            api_key=api_key,
            skip_system_prompt=True,
            base_url=get_bot_query_base_url(),
        ):
            chunk = await create_stream_chunk(
                message.text, model, format_type, first_chunk
            )
            accumulated_response += message.text  # Accumulate the full response text
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            first_chunk = False
            await asyncio.sleep(0)  # Allow event loop to process

        # Calculate completion tokens from accumulated response
        completion_tokens = count_tokens(accumulated_response)
        token_counts["completion_tokens"] = completion_tokens
        token_counts["total_tokens"] = token_counts["prompt_tokens"] + completion_tokens

        # Send final message with token counts
        final_chunk = await create_final_chunk(model, format_type, token_counts)
        yield f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8")

        if format_type in ["completion", "chat"]:
            yield b"data: [DONE]\n\n"

    except Exception as e:
        logger.exception(f"Stream error: {str(e)}")

        # Use the helper function to parse error information
        error_message, error_data, error_type, error_id = parse_poe_error(e)

        # Add token counts to error response if available
        if accumulated_response:
            # Calculate completion tokens from accumulated response
            completion_tokens = count_tokens(accumulated_response)
            token_counts["completion_tokens"] = completion_tokens
            token_counts["total_tokens"] = (
                token_counts["prompt_tokens"] + completion_tokens
            )

        error_data = {
            "error": {"message": error_message, "type": error_type, "code": error_type}
        }

        # Add error_id if available
        if error_id:
            error_data["error"]["error_id"] = error_id

        # Add token counts if available and we had some response before the error
        if accumulated_response:
            error_data["usage"] = token_counts

        yield f"data: {json.dumps(error_data)}\n\n".encode("utf-8")
        if format_type in ["completion", "chat"]:
            yield b"data: [DONE]\n\n"


async def stream_completions_format(
    model: str, messages: list[fp.ProtocolMessage], api_key: str
):
    async for chunk in stream_response(model, messages, api_key, "completion"):
        yield chunk


@app.post("/completions")
@app.post("/v1/completions")
@app.post("//v1/completions")
async def completions(request: Request):
    body = await request.json()

    messages = [fp.ProtocolMessage(role="user", content=body.get("prompt", ""))]
    model = body.get("model")
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(
            stream_completions_format(model, messages, await get_api_key()),
            media_type="text/event-stream",
        )

    # For non-streaming requests, accumulate the full response
    response = await generate_poe_bot_response(model, messages, await get_api_key())

    # Calculate token counts
    prompt_tokens = count_tokens(body.get("prompt", ""))
    completion_tokens = count_tokens(response.get("content", ""))
    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    return {
        "id": "cmpl-" + os.urandom(12).hex(),
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "text": response.get("content", ""),
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": token_usage,
    }


@app.get("/")
async def root():
    return {
        "message": "Poe API OpenAI-compatible proxy server",
        "version": "1.0.0",
        "endpoints": {
            "OpenAI-compatible": [
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/models",
            ],
            "Poe-compatible": [],
        },
    }


@app.get("/api/auth/check")
async def check_auth(api_key: str = Depends(get_api_key)):
    """Endpoint to check if authentication is working"""
    return {"status": "authenticated", "timestamp": datetime.now().isoformat()}


async def generate_poe_bot_response(
    model, messages: list[fp.ProtocolMessage], api_key: str
):
    model = normalize_model(model)
    request_id = os.urandom(4).hex()
    logger.info(f"[{request_id}] Processing request for model {model}")
    accumulated_text = ""

    try:
        response = {"role": "assistant", "content": ""}

        async for message in get_bot_response(
            messages=messages,
            bot_name=model,
            api_key=api_key,
            skip_system_prompt=True,
            base_url=get_bot_query_base_url(),
        ):
            accumulated_text += message.text  # Accumulate the text
            response["content"] = accumulated_text

        # Simple success log with response length only
        logger.info(f"[{request_id}] Response received: {len(accumulated_text)} chars")

    except Exception as e:
        logger.exception(f"[{request_id}] Error: {str(e)}")
        # Use the helper function to parse error information
        error_message, error_data, error_type, error_id = parse_poe_error(e)

        if error_data:
            raise PoeAPIError(
                f"Poe API Error: {error_message}",
                error_data=error_data,
                error_id=error_id,
            )

        # If we couldn't parse a structured error, just raise the original
        raise

    return response


async def stream_openai_format(
    model: str, messages: list[fp.ProtocolMessage], api_key: str
):
    async for chunk in stream_response(model, messages, api_key, "chat"):
        yield chunk


@app.get("/openapi.json")
async def get_openapi_json():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Poe-API OpenAI Proxy",
        version="1.0.0",
        description="A proxy server for Poe API that provides OpenAI-compatible endpoints",
        routes=app.routes,
    )

    # Customize the schema as needed
    openapi_schema["info"]["x-logo"] = {"url": "https://poe.com/favicon.ico"}

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Add an exception handler for better logging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(
        f"Exception in {request.method} {request.url.path}: {type(exc).__name__}"
    )

    # For HTTPExceptions, return their predefined responses
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=(
                {"error": exc.detail}
                if not isinstance(exc.detail, dict)
                else exc.detail
            ),
        )

    # For other exceptions, return a 500 error
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": f"An unexpected error occurred: {str(exc)}",
                "type": "server_error",
            }
        },
    )
