"""
FastAPI Backend for Ollama Chatbot
Connects frontend to local Ollama instance running Qwen2.5:3b-instruct
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from typing import List, Optional
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ollama Chatbot API",
    description="Backend API for chatbot using Ollama Qwen2.5:3b-instruct",
    version="1.0.0"
)

# CORS Configuration - allows frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_NAME='qwen3:4b'
OLLAMA_URL='http://localhost:11434'

# Pydantic Models for request/response validation
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    response: str
    model: str

class HealthResponse(BaseModel):
    status: str
    ollama: str
    model: str

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Ollama Chatbot API is running",
        "model": MODEL_NAME,
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Verifies that Ollama is running and accessible
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check if Ollama is running
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            
            if response.status_code == 200:
                models = response.json()
                model_names = [m["name"] for m in models.get("models", [])]
                
                # Check if our specific model is available
                model_available = any(MODEL_NAME in name for name in model_names)
                
                return HealthResponse(
                    status="healthy",
                    ollama="connected",
                    model="available" if model_available else "not found"
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Ollama is running but returned an error"
                )
                
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=503,
            detail="Ollama connection timeout. Is Ollama running?"
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Make sure Ollama is running on port 11434"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Ollama health check failed: {str(e)}"
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    Receives user message and conversation history, returns AI response
    """
    logger.info(f"Received message: {request.message[:50]}...")
    
    try:
        # Build message array for Ollama
        messages = []
        
        # Add conversation history
        for msg in request.history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": request.message
        })
        
        logger.info(f"Sending {len(messages)} messages to Ollama")
        
        # Call Ollama API
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                }
            )
            
            # Handle errors
            if response.status_code != 200:
                logger.error(f"Ollama returned status {response.status_code}: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Ollama API error: {response.text}"
                )
            
            # Parse response
            result = response.json()
            assistant_message = result.get("message", {}).get("content", "")
            
            if not assistant_message:
                raise HTTPException(
                    status_code=500,
                    detail="Received empty response from Ollama"
                )
            
            logger.info(f"Received response: {assistant_message[:50]}...")
            
            return ChatResponse(
                response=assistant_message,
                model=MODEL_NAME
            )
            
    except httpx.TimeoutException:
        logger.error("Request to Ollama timed out")
        raise HTTPException(
            status_code=504,
            detail="Request to Ollama timed out. The model might be processing a complex request."
        )
    except httpx.ConnectError:
        logger.error("Failed to connect to Ollama")
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Make sure Ollama is running on port 11434"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Ollama URL: {OLLAMA_URL}")
        logger.error(f"Model name: {MODEL_NAME}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/models")
async def list_models():
    """List all available models in Ollama"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to fetch models from Ollama"
                )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to Ollama: {str(e)}"
        )

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Starting Ollama Chatbot Backend")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Server will run on: http://localhost:8000")
    print(f"API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )