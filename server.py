#!/usr/bin/env python3
"""
Poe Chat Backend Server
Handles chat completions for VS Code extension integration
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Poe Chat Backend", version="1.0.0")

# Add CORS middleware for VS Code extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to VS Code extension
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000

class ChatChoice(BaseModel):
    message: ChatMessage
    finish_reason: str = "stop"

class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    choices: List[ChatChoice]
    usage: Optional[ChatUsage] = None

load_dotenv()

POE_API_KEY = os.getenv("POE_API_KEY")
POE_API_BASE_URL = os.getenv("POE_API_BASE_URL", "https://api.poe.com/v1")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "poe-chat-backend"}

@app.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """
    Handle chat completions from VS Code extension
    """
    try:
        logger.info(f"Received chat request for model: {request.model}")
        logger.info(f"Number of messages: {len(request.messages)}")
        
        # Extract system message (contains code context)
        system_message = None
        user_messages = []
        
        for msg in request.messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                user_messages.append(msg)
        
        # Build prompt for LLM
        if system_message:
            # Combine system context with conversation
            full_prompt = f"{system_message}\n\nConversation:\n"
            for msg in user_messages:
                role_prefix = "Human: " if msg.role == "user" else "Assistant: "
                full_prompt += f"{role_prefix}{msg.content}\n"
            full_prompt += "Assistant: "
        else:
            # Fallback for simple conversations
            full_prompt = "\n".join([
                f"{'Human: ' if msg.role == 'user' else 'Assistant: '}{msg.content}" 
                for msg in user_messages
            ]) + "\nAssistant: "
        
        logger.info(f"Built prompt with {len(full_prompt)} characters")
        
        # Call LLM API (placeholder - replace with your actual LLM integration)
        response_text = await call_llm_api(full_prompt, request.model)
        
        # Calculate token usage (rough estimation)
        prompt_tokens = len(full_prompt.split())
        completion_tokens = len(response_text.split())
        total_tokens = prompt_tokens + completion_tokens
        
        # Format response
        return ChatResponse(
            choices=[
                ChatChoice(
                    message=ChatMessage(
                        role="assistant",
                        content=response_text
                    )
                )
            ],
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def call_llm_api(prompt: str, model: str) -> str:
    """
    Call the Poe.com OpenAI-compatible API using the selected model.
    """
    try:
        # Compose messages for Poe API (system + user)
        # For now, treat the first line as system, rest as user
        lines = prompt.split("\n")
        system_message = lines[0] if lines else "You are a helpful assistant."
        user_content = "\n".join(lines[1:]) if len(lines) > 1 else ""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        headers = {
            "Authorization": f"Bearer {POE_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7
        }
        response = requests.post(f"{POE_API_BASE_URL}/chat/completions", headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Poe API error: {e}")
        return f"I'm sorry, but I encountered an error while processing your request: {str(e)}"

def generate_mock_response(prompt: str) -> str:
    """
    Generate a mock response based on the prompt content
    """
    if "error" in prompt.lower() or "bug" in prompt.lower():
        return "I can see you're dealing with an error. Let me help you debug this. Could you share more details about the specific error message you're seeing?"
    
    elif "function" in prompt.lower() or "method" in prompt.lower():
        return "I can help you with functions and methods. I see the code you've shared. What specific aspect would you like me to explain or help you improve?"
    
    elif "test" in prompt.lower():
        return "Testing is crucial for code quality. I can help you write tests or improve existing ones. What type of testing are you working on?"
    
    elif "performance" in prompt.lower() or "optimize" in prompt.lower():
        return "Performance optimization is important. I can analyze your code for potential improvements. What specific performance concerns do you have?"
    
    else:
        return "I can see your code and project structure. I'm here to help you with any questions about your codebase, debugging, refactoring, or general development tasks. What would you like to know?"

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Poe Chat Backend Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat/completions"
        }
    }

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Poe Chat Backend Server on {host}:{port}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
