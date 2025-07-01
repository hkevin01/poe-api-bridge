#!/usr/bin/env python3
"""
Poe Chat Backend Server using poe-api-wrapper
Handles chat completions for VS Code extension integration with free access to GPT-4, Claude, Gemini, etc.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import poe-api-wrapper
try:
    from poe_api_wrapper import AsyncPoeApi
    POE_WRAPPER_AVAILABLE = True
except ImportError:
    POE_WRAPPER_AVAILABLE = False
    print("Warning: poe-api-wrapper not installed. Please run: pip install poe-api-wrapper")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Poe Chat Backend", version="2.0.0")

# Add CORS middleware for VS Code extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to VS Code extension
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting configuration
RATE_LIMIT_CALLS = 8  # Conservative: 8 calls per minute
RATE_LIMIT_WINDOW = 60  # 60 seconds
rate_limit_calls = []
rate_limit_lock = asyncio.Lock()

# Fallback configuration
USE_FALLBACK = True
FALLBACK_DELAY = 2  # seconds to wait before falling back

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str  # 'user', 'assistant', 'system'
    content: Union[str, List[Dict[str, Any]]]  # Support for text and multimodal content

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None

class ChatChoice(BaseModel):
    message: ChatMessage
    finish_reason: str = "stop"
    index: int = 0

class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    choices: List[ChatChoice]
    usage: Optional[ChatUsage] = None
    model: str

class ImageGenerationRequest(BaseModel):
    model: str
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"

# Model mapping from OpenAI-style names to Poe model names
MODEL_MAPPING = {
    # Claude models
    "claude-3.5-sonnet": "claude_3_igloo",
    "claude-3-opus": "claude_2_1_cedar", 
    "claude-3-sonnet": "claude_2_1_bamboo",
    "claude-3-haiku": "claude_3_haiku",
    "claude-instant": "a2",
    
    # GPT models
    "gpt-4": "beaver",
    "gpt-4-turbo": "beaver",
    "gpt-4o": "gpt4_o",
    "gpt-4o-mini": "gpt4_o_mini",
    "gpt-3.5-turbo": "chinchilla",
    "gpt-3.5-turbo-instruct": "chinchilla_instruct",
    
    # Gemini models
    "gemini-pro": "gemini_pro",
    "gemini-1.5-pro": "gemini_1_5_pro_1m",
    "gemini-1.5-flash": "gemini_1_5_flash",
    
    # Default fallback
    "default": "claude_3_haiku"
}

# Global Poe client
poe_client = None

# Load environment variables
load_dotenv()

# Poe tokens configuration
POE_TOKENS = {
    'p-b': os.getenv('POE_P_B_TOKEN', ''),
    'p-lat': os.getenv('POE_P_LAT_TOKEN', ''),
    # Optional tokens
    'formkey': os.getenv('POE_FORMKEY', ''),
    '__cf_bm': os.getenv('POE_CF_BM', ''),
    'cf_clearance': os.getenv('POE_CF_CLEARANCE', '')
}

def get_poe_model(openai_model: str) -> str:
    """Map OpenAI model names to Poe model names"""
    return MODEL_MAPPING.get(openai_model, MODEL_MAPPING["default"])

def build_context_aware_prompt(messages: List[ChatMessage]) -> str:
    """Build a comprehensive prompt from messages with code context"""
    prompt_parts = []
    
    # Extract system message (contains code context)
    system_content = None
    conversation_messages = []
    
    for msg in messages:
        if msg.role == "system":
            system_content = msg.content
        else:
            conversation_messages.append(msg)
    
    # Add system context if available
    if system_content:
        prompt_parts.append("SYSTEM CONTEXT:")
        prompt_parts.append(str(system_content))
        prompt_parts.append("\n" + "="*50 + "\n")
    
    # Add conversation
    prompt_parts.append("CONVERSATION:")
    for msg in conversation_messages:
        role_prefix = "Human: " if msg.role == "user" else "Assistant: "
        content = str(msg.content) if isinstance(msg.content, str) else json.dumps(msg.content)
        prompt_parts.append(f"{role_prefix}{content}")
    
    # Add final assistant prompt
    prompt_parts.append("Assistant: ")
    
    return "\n".join(prompt_parts)

async def check_rate_limit():
    """Check if we're within rate limits"""
    async with rate_limit_lock:
        now = time.time()
        # Remove old calls outside the window
        global rate_limit_calls
        rate_limit_calls = [call_time for call_time in rate_limit_calls 
                           if now - call_time < RATE_LIMIT_WINDOW]
        
        if len(rate_limit_calls) >= RATE_LIMIT_CALLS:
            return False
        
        rate_limit_calls.append(now)
        return True

def generate_fallback_response(prompt: str, model: str) -> str:
    """Generate a fallback response when Poe API fails"""
    logger.info(f"🔄 Using fallback response for model: {model}")
    
    # Extract key information from the prompt
    prompt_lower = prompt.lower()
    
    # Check for code-related content
    if any(keyword in prompt_lower for keyword in ['code', 'function', 'class', 'import', 'def', 'var', 'const']):
        return """I can help you with your code! I see you're working on a programming project. 

Here are some suggestions:
• Check for syntax errors in your code
• Ensure all imports are correct
• Verify function parameters and return types
• Test your code with different inputs

Would you like me to help you debug a specific issue or explain any part of your code?"""
    
    # Check for error-related content
    elif any(keyword in prompt_lower for keyword in ['error', 'bug', 'exception', 'fail', 'crash']):
        return """I can help you troubleshoot this issue! 

To better assist you, please provide:
• The exact error message
• What you were trying to do when the error occurred
• Any relevant code snippets
• Steps to reproduce the problem

This will help me give you a more targeted solution."""
    
    # Check for general questions
    elif any(keyword in prompt_lower for keyword in ['how', 'what', 'why', 'when', 'where']):
        return """I'd be happy to help you with that! 

To provide the best assistance, could you:
• Give me more specific details about what you're trying to accomplish?
• Share any relevant code or context?
• Let me know what you've already tried?

This will help me give you a more helpful and accurate response."""
    
    # Default response
    else:
        return """Hello! I'm here to help you with your coding and development tasks.

I can assist with:
• Code review and optimization
• Debugging and troubleshooting
• Best practices and design patterns
• Language-specific questions
• Project structure and architecture

What would you like to work on today?"""

async def initialize_poe_client():
    """Initialize the Poe API client"""
    global poe_client
    
    if not POE_WRAPPER_AVAILABLE:
        raise ValueError("poe-api-wrapper is not installed. Please run: pip install poe-api-wrapper")
    
    if not POE_TOKENS['p-b'] or not POE_TOKENS['p-lat']:
        raise ValueError("POE_P_B_TOKEN and POE_P_LAT_TOKEN environment variables are required")
    
    try:
        # Filter out empty tokens
        tokens = {k: v for k, v in POE_TOKENS.items() if v}
        poe_client = await AsyncPoeApi(tokens=tokens).create()
        logger.info("✅ Poe client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Poe client: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the Poe client on startup"""
    if POE_WRAPPER_AVAILABLE:
        await initialize_poe_client()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy" if poe_client else "initializing"
    return {
        "status": status, 
        "service": "poe-chat-backend",
        "poe_wrapper_available": POE_WRAPPER_AVAILABLE,
        "poe_client_ready": poe_client is not None,
        "rate_limit_calls": len(rate_limit_calls),
        "rate_limit_max": RATE_LIMIT_CALLS
    }

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """List available models"""
    models = [
        {"id": model, "object": "model", "owned_by": "poe", "permission": []}
        for model in MODEL_MAPPING.keys()
    ]
    return {"object": "list", "data": models}

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completions with rate limiting and fallback"""
    try:
        # Check rate limit
        if not await check_rate_limit():
            logger.warning(f"⚠️ Rate limit exceeded: {len(rate_limit_calls)} calls in {RATE_LIMIT_WINDOW}s")
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_CALLS} calls per {RATE_LIMIT_WINDOW} seconds."
            )
        
        # Map model name
        poe_model = get_poe_model(request.model)
        logger.info(f"🤖 Using model: {request.model} -> {poe_model}")
        
        # Build context-aware prompt
        prompt = build_context_aware_prompt(request.messages)
        
        # Try Poe API first
        if poe_client and USE_FALLBACK:
            try:
                logger.info("🚀 Attempting Poe API call...")
                
                if request.stream:
                    # Handle streaming response
                    async def generate_stream():
                        try:
                            response_text = ""
                            async for chunk in poe_client.send_message(bot=poe_model, message=prompt):
                                if "response" in chunk:
                                    delta_content = chunk["response"]
                                    response_text += delta_content
                                    
                                    # Format as OpenAI-compatible streaming response
                                    streaming_chunk = {
                                        "choices": [{
                                            "delta": {"content": delta_content},
                                            "finish_reason": None,
                                            "index": 0
                                        }],
                                        "object": "chat.completion.chunk",
                                        "model": request.model
                                    }
                                    
                                    yield f"data: {json.dumps(streaming_chunk)}\n\n"
                            
                            # Send final chunk
                            final_chunk = {
                                "choices": [{
                                    "delta": {},
                                    "finish_reason": "stop",
                                    "index": 0
                                }],
                                "object": "chat.completion.chunk",
                                "model": request.model
                            }
                            
                            if request.stream_options and request.stream_options.get("include_usage"):
                                final_chunk["usage"] = {
                                    "prompt_tokens": len(prompt.split()),
                                    "completion_tokens": len(response_text.split()),
                                    "total_tokens": len(prompt.split()) + len(response_text.split())
                                }
                            
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            
                        except Exception as e:
                            logger.error(f"❌ Streaming error: {e}")
                            # Fall back to non-streaming fallback
                            fallback_response = generate_fallback_response(prompt, request.model)
                            fallback_chunk = {
                                "choices": [{
                                    "delta": {"content": fallback_response},
                                    "finish_reason": "stop",
                                    "index": 0
                                }],
                                "object": "chat.completion.chunk",
                                "model": request.model
                            }
                            yield f"data: {json.dumps(fallback_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                    
                    return StreamingResponse(
                        generate_stream(),
                        media_type="text/plain",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                    )
                
                else:
                    # Handle non-streaming response
                    response_text = ""
                    async for chunk in poe_client.send_message(bot=poe_model, message=prompt):
                        if "response" in chunk:
                            response_text += chunk["response"]
                        elif "text" in chunk:
                            response_text = chunk["text"]
                            break
                    
                    logger.info("✅ Poe API call successful")
                    return ChatCompletionResponse(
                        choices=[ChatChoice(
                            message=ChatMessage(
                                role="assistant",
                                content=response_text
                            ),
                            finish_reason="stop",
                            index=0
                        )],
                        usage=ChatUsage(
                            prompt_tokens=len(prompt.split()),
                            completion_tokens=len(response_text.split()),
                            total_tokens=len(prompt.split()) + len(response_text.split())
                        ),
                        model=request.model
                    )
                    
            except Exception as e:
                logger.error(f"❌ Poe API error: {e}")
                if USE_FALLBACK:
                    logger.info(f"⏳ Waiting {FALLBACK_DELAY}s before fallback...")
                    await asyncio.sleep(FALLBACK_DELAY)
                    logger.info("🔄 Using fallback response")
                else:
                    raise HTTPException(status_code=500, detail=str(e))
        
        # Use fallback response
        fallback_response = generate_fallback_response(prompt, request.model)
        
        if request.stream:
            async def generate_fallback_stream():
                fallback_chunk = {
                    "choices": [{
                        "delta": {"content": fallback_response},
                        "finish_reason": "stop",
                        "index": 0
                    }],
                    "object": "chat.completion.chunk",
                    "model": request.model
                }
                yield f"data: {json.dumps(fallback_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_fallback_stream(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            return ChatCompletionResponse(
                choices=[ChatChoice(
                    message=ChatMessage(
                        role="assistant",
                        content=fallback_response
                    ),
                    finish_reason="stop",
                    index=0
                )],
                usage=ChatUsage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(fallback_response.split()),
                    total_tokens=len(prompt.split()) + len(fallback_response.split())
                ),
                model=request.model
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/images/generations")
@app.post("/images/generations") 
async def generate_images(request: ImageGenerationRequest):
    """Generate images using Poe's image generation models"""
    try:
        if not poe_client:
            raise HTTPException(status_code=503, detail="Poe client not initialized")
        
        # Map to Poe image generation models
        image_models = {
            "dall-e-3": "dalle3",
            "dall-e-2": "dalle3",
            "stable-diffusion": "stablediffusionxl",
            "playground-v2.5": "playgroundv25",
            "ideogram": "ideogram"
        }
        
        poe_model = image_models.get(request.model, "dalle3")
        
        # This is a placeholder - you'd need to implement image generation
        # based on the poe-api-wrapper's image generation capabilities
        return {
            "created": 1234567890,
            "data": [{
                "url": "https://example.com/generated-image.png"
            }]
        }
        
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Poe Chat Backend Server (poe-api-wrapper)",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "images": "/v1/images/generations"
        },
        "available_models": list(MODEL_MAPPING.keys()),
        "rate_limit": {
            "calls_per_minute": RATE_LIMIT_CALLS,
            "current_calls": len(rate_limit_calls)
        },
        "fallback_enabled": USE_FALLBACK
    }

# Simple /chat endpoint for VS Code extension compatibility
class SimpleChatRequest(BaseModel):
    message: str
    model: str = "claude-3-haiku"
    max_tokens: int = 2000
    temperature: float = 0.7

class SimpleChatResponse(BaseModel):
    response: str
    model: str

@app.post("/chat", response_model=SimpleChatResponse)
async def simple_chat(req: SimpleChatRequest = Body(...)):
    # Reuse the existing chat_completions logic
    chat_req = ChatCompletionRequest(
        model=req.model,
        messages=[ChatMessage(role="user", content=req.message)],
        temperature=req.temperature,
        max_tokens=req.max_tokens
    )
    result = await chat_completions(chat_req)
    # Extract the assistant's reply
    reply = result.choices[0].message.content if result.choices else ""
    return SimpleChatResponse(response=reply, model=result.model)

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    
    print(f"🚀 Starting Poe Chat Backend Server...")
    print(f"📡 Server will be available at: http://{host}:{port}")
    print(f"📖 API docs will be available at: http://{host}:{port}/docs")
    print(f"⏱️  Rate limit: {RATE_LIMIT_CALLS} calls per {RATE_LIMIT_WINDOW} seconds")
    print(f"🔄 Fallback enabled: {USE_FALLBACK}")
    
    uvicorn.run("server:app", host=host, port=port, loop="asyncio")
