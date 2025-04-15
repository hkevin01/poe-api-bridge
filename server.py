# Standard library imports
import asyncio
import functools
import json
import logging
import os
import time
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

# Third-party imports
import fastapi_poe as fp
import tiktoken
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi_poe.client import get_bot_response
from pydantic import BaseModel, Field
from logtail import LogtailHandler

# Configure logging
# Get log level from environment - matches uvicorn's --log-level
log_level = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create the FastAPI app
app = FastAPI()

# Create a logger dependency
class LoggerFactory:
    _instance = None
    _initialized = False
    _logger = None
    _source_token = None
    _source_id = None
    _host = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LoggerFactory()
        return cls._instance
    
    def initialize(self):
        """Initializes the logger with Better Stack configuration."""
        if self._initialized:
            return
        
        # Create a logger instance
        logger = logging.getLogger("poe-openai-proxy")
        
        try:
            # Using dictionary-style access (os.environ[]) to ensure it throws exception when not present
            self._source_token = os.environ["LOGTAIL_SOURCE_TOKEN"]
            self._source_id = os.environ["LOGTAIL_SOURCE_ID"]
            
            # LOGTAIL_HOST is optional with a default value
            self._host = os.environ.get("LOGTAIL_HOST", "s1275096.eu-nbg-2.betterstackdata.com")
            
            # Remove any http:// or https:// prefix if present (Better Stack expects host without protocol)
            if self._host.startswith(("http://", "https://")):
                parsed_url = urlparse(self._host)
                self._host = parsed_url.netloc
            
            # Initialize Better Stack handler
            logtail_handler = LogtailHandler(
                source_token=self._source_token,
                host=self._host,
            )
            
            # Set formatter to include same format as basic logging
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            logtail_handler.setFormatter(formatter)
            
            # Add the handler to the logger
            logger.addHandler(logtail_handler)
            
            # Minimal startup logging
            context = {"source_id": self._source_id}
            logger.info("Poe API Bridge starting", extra={"context": context})
            
            self._initialized = True
            self._logger = logger
            
        except KeyError as e:
            # This will provide a clear error message about which environment variable is missing
            raise ValueError(f"Required environment variable not set: {e}")
    
    def get_logger(self):
        """Returns the initialized logger."""
        if not self._initialized:
            self.initialize()
        return self._logger
    
    def log_context(self, **kwargs):
        """Creates a context dictionary for structured logging, including source_id."""
        if not self._initialized:
            self.initialize()
            
        context = dict(kwargs)
        context["source_id"] = self._source_id
        return {"context": context}
    
    @property
    def source_id(self):
        """Return the source ID."""
        if not self._initialized:
            self.initialize()
        return self._source_id
    
    @property
    def host(self):
        """Return the host."""
        if not self._initialized:
            self.initialize()
        return self._host
    
    @property
    def source_token(self):
        """Return the source token."""
        if not self._initialized:
            self.initialize()
        return self._source_token


# Create a dependency to get the logger
def get_logger():
    factory = LoggerFactory.get_instance()
    return factory.get_logger()

# Create a dependency to get the context
def get_log_context(**kwargs):
    factory = LoggerFactory.get_instance()
    return factory.log_context(**kwargs)

# Create a dependency to get a configured logger with request ID
async def get_request_logger(request: Request = None):
    logger = get_logger()
    request_id = os.urandom(4).hex()
    context = {}
    
    if request:
        context = {
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "user_agent": request.headers.get("user-agent", "unknown"),
            "client_ip": request.client.host if request.client else "unknown",
        }
    
    # Return both the logger and the request_id
    return {
        "logger": logger,
        "request_id": request_id,
        "context": context
    }

# Add request logging middleware with simplified logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    path = request.url.path
    method = request.method
    request_id = os.urandom(4).hex()

    # Get the logger instance
    logger = get_logger()
    
    # Create contextual information for Better Stack
    context = {
        "request_id": request_id,
        "path": path,
        "method": method,
        "user_agent": request.headers.get("user-agent", "unknown"),
        "client_ip": request.client.host if request.client else "unknown",
    }

    # No info log for regular requests to reduce verbosity

    try:
        response = await call_next(request)

        process_time = (time.time() - start_time) * 1000
        formatted_process_time = f"{process_time:.2f}"
        status_code = response.status_code

        # Add response info to context
        context.update({
            "status_code": status_code,
            "process_time_ms": process_time,
        })

        # Only log errors (4xx and 5xx)
        if status_code >= 400:
            logger.warning(
                f"[{request_id}] Error: {method} {path} -> {status_code} (took {formatted_process_time} ms)",
                extra=get_log_context(**context)
            )
        # No info log for successful responses to reduce verbosity

        return response
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        formatted_process_time = f"{process_time:.2f}"

        # Add error info to context
        context.update({
            "error": str(e),
            "error_type": type(e).__name__,
            "process_time_ms": process_time,
        })

        logger.exception(
            f"[{request_id}] Exception: {method} {path} - {str(e)} (took {formatted_process_time} ms)",
            extra=get_log_context(**context)
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
    messages: list[ChatCompletionMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    seed: Optional[int] = None
    response_format: Optional[Dict[str, str]] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, list[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[int, float]] = None
    user: Optional[str] = None


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, list[str]]
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None


class ModerationRequest(BaseModel):
    input: Union[str, list[str]]
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
    messages: list[fp.ProtocolMessage], model: str = None
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
    request: ChatCompletionRequest, 
    api_key: str = Depends(get_api_key),
    logger_data: dict = Depends(get_request_logger)
):
    logger = logger_data["logger"]
    request_id = logger_data["request_id"]
    
    logger.info(
        f"[{request_id}] Processing chat completion request for model: {request.model}",
        extra=get_log_context(request_id=request_id, model=request.model)
    )

    # Log request parameters at debug level
    logger.debug(
        f"[{request_id}] Request parameters:\n"
        f"    model={request.model}\n"
        f"    temperature={request.temperature}\n"
        f"    top_p={request.top_p}\n"
        f"    n={request.n}\n"
        f"    stream={request.stream}\n"
        f"    max_tokens={request.max_tokens}\n"
        f"    presence_penalty={request.presence_penalty}\n"
        f"    frequency_penalty={request.frequency_penalty}\n"
        f"    messages_count={len(request.messages)}"
    )

    try:
        # Validate model first
        model = normalize_model(request.model)

        # Validate messages
        if not request.messages:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Messages array cannot be empty",
                        "type": "invalid_request_error",
                        "param": "messages",
                    }
                },
            )

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
                stream_openai_format(request.model, messages, api_key, request_id),
                headers=headers,
                media_type="text/event-stream",
            )

        # For non-streaming, accumulate the full response
        response = await generate_poe_bot_response(request.model, messages, api_key, request_id)

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
        logger.exception(
            f"Error in chat_completions: {str(e)}", 
            extra=get_log_context(request_id=request_id, error=str(e), error_type=type(e).__name__)
        )

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


async def stream_response(
    model: str, messages: list[fp.ProtocolMessage], api_key: str, format_type: str, request_id: str = None
):
    """Common streaming function for all response types"""
    model = normalize_model(model)
    first_chunk = True
    accumulated_response = ""
    if not request_id:
        request_id = os.urandom(4).hex()
    logger = get_logger()

    # Calculate prompt tokens before starting stream
    token_counts = count_message_tokens(messages)

    try:
        async for message in get_bot_response(
            messages=messages, bot_name=model, api_key=api_key, skip_system_prompt=True
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
        logger.exception(
            f"Stream error: {str(e)}", 
            extra=get_log_context(
                request_id=request_id, 
                model=model, 
                error=str(e), 
                error_type=type(e).__name__
            )
        )

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
    model: str, messages: list[fp.ProtocolMessage], api_key: str, format_type: str, request_id: str = None
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


async def stream_openai_format(
    model: str, messages: list[fp.ProtocolMessage], api_key: str, request_id: str = None
):
    async for chunk in stream_response(model, messages, api_key, "chat", request_id):
        yield chunk


async def stream_completions_format(
    model: str, messages: list[fp.ProtocolMessage], api_key: str, request_id: str = None
):
    async for chunk in stream_response(model, messages, api_key, "completion", request_id):
        yield chunk


@app.get("/")
async def root(request: Request, logger_data: dict = Depends(get_request_logger)):
    logger = logger_data["logger"]
    
    # Get host information
    host = request.headers.get("host", "unknown")
    origin = request.headers.get("origin", "unknown")
    referer = request.headers.get("referer", "unknown")
    
    # No logging on root endpoint to reduce verbosity
    
    return {
        "message": "Poe API OpenAI-compatible proxy server",
        "version": "1.0.0",
        "endpoints": {
            "OpenAI-compatible": [
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/models",
            ],
            "Utility": [
                "/api/auth/check - Verify authentication",
                "/api/logs/test - Test Better Stack logging",
            ],
        },
        "server_time": datetime.now().isoformat(),
        "hosting_domain": host,
        "logging": {
            "status": "enabled",
            "provider": "Better Stack",
            "endpoint": logger.host,
            "source_id": logger.source_id,
            "source_token": logger.source_token[:4] + "***" if logger.source_token else "not_set",
        },
    }


async def generate_poe_bot_response(
    model, messages: list[fp.ProtocolMessage], api_key: str, request_id: str = None
):
    model = normalize_model(model)
    if not request_id:
        request_id = os.urandom(4).hex()
    logger = get_logger()
    
    # No info log for request processing to reduce verbosity
    accumulated_text = ""

    try:
        response = {"role": "assistant", "content": ""}

        async for message in get_bot_response(
            messages=messages, bot_name=model, api_key=api_key, skip_system_prompt=True
        ):
            accumulated_text += message.text  # Accumulate the text
            response["content"] = accumulated_text

        # No info log for successful response to reduce verbosity

    except Exception as e:
        logger.exception(
            f"[{request_id}] Model error: {str(e)}", 
            extra=get_log_context(
                request_id=request_id, 
                model=model, 
                error=str(e), 
                error_type=type(e).__name__
            )
        )
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


# Add an exception handler for better logging
@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, 
    exc: Exception
):
    request_id = os.urandom(4).hex()
    logger = get_logger()
    
    context = {
        "request_id": request_id,
        "path": request.url.path,
        "method": request.method,
        "error": str(exc),
        "error_type": type(exc).__name__,
    }
    
    logger.exception(
        f"Exception in {request.method} {request.url.path}: {type(exc).__name__}",
        extra=get_log_context(**context)
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
                "request_id": request_id,
            }
        },
    )


@app.get("/api/logs/test")
async def test_logging(
    request: Request, 
    message: str = Query("Test log message", description="Custom log message to send"),
    level: str = Query("info", description="Log level (debug, info, warning, error, critical)"),
    logger_data: dict = Depends(get_request_logger)
):
    """Endpoint to test Better Stack logging"""
    logger = logger_data["logger"]
    request_id = logger_data["request_id"]
    current_time = datetime.now().isoformat()
    
    # Get host information
    host = request.headers.get("host", "unknown")
    
    # Create log context
    context = {
        "request_id": request_id,
        "test": True,
        "timestamp": current_time,
        "host": host,
        "domain": request.base_url.netloc,
        "user_agent": request.headers.get("user-agent", "unknown"),
        "client_ip": request.client.host if request.client else "unknown",
        "message": message,
    }
    
    # Log with the specified level
    log_message = f"[{request_id}] TEST LOG: {message}"
    
    if level.lower() == "debug":
        logger.debug(log_message, extra=get_log_context(**context))
    elif level.lower() == "warning":
        logger.warning(log_message, extra=get_log_context(**context))
    elif level.lower() == "error":
        logger.error(log_message, extra=get_log_context(**context))
    elif level.lower() == "critical":
        logger.critical(log_message, extra=get_log_context(**context))
    else:
        # Default to info
        logger.info(log_message, extra=get_log_context(**context))
    
    return {
        "status": "success",
        "message": f"Test log sent to Better Stack with level: {level}",
        "timestamp": current_time,
        "request_id": request_id,
        "log_details": {
            "level": level,
            "content": message,
            "endpoint": logger.host,
            "source_id": logger.source_id,
        }
    }


@app.get("/models")
@app.get("/v1/models")
@app.get("//v1/models")  # Handle double slash case like other endpoints
async def list_models_openai(logger_data: dict = Depends(get_request_logger)):
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


@app.post("/completions")
@app.post("/v1/completions")
@app.post("//v1/completions")
async def completions(
    request: Request, 
    logger_data: dict = Depends(get_request_logger)
):
    logger = logger_data["logger"]
    request_id = logger_data["request_id"]
    body = await request.json()

    messages = [fp.ProtocolMessage(role="user", content=body.get("prompt", ""))]
    model = body.get("model")
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(
            stream_completions_format(model, messages, await get_api_key(), request_id),
            media_type="text/event-stream",
        )

    # For non-streaming requests, accumulate the full response
    response = await generate_poe_bot_response(model, messages, await get_api_key(), request_id)

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


@app.get("/api/auth/check")
async def check_auth(
    request: Request, 
    api_key: str = Depends(get_api_key),
    logger_data: dict = Depends(get_request_logger)
):
    """Endpoint to check if authentication is working"""
    # No logging for auth check to reduce verbosity
    
    return {
        "status": "authenticated", 
        "timestamp": datetime.now().isoformat(),
        "server_info": {
            "domain": request.base_url.netloc,
            "scheme": request.base_url.scheme,
        }
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
async def get_openapi_json(logger_data: dict = Depends(get_request_logger)):
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
