"""
Main FastAPI application for serving fine-tuned LLM models
"""
import os
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextGenerationPipeline
)
from pathlib import Path
import logging
import uvicorn
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security utilities
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class User(BaseModel):
    username: str
    email: Optional[str] = None
    hashed_password: str
    disabled: Optional[bool] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

class GenerationResponse(BaseModel):
    generated_text: str
    generation_time: float
    model: str

# Initialize FastAPI
app = FastAPI(
    title="LLM Deployment API",
    description="API for serving fine-tuned LLM models",
    version="1.0.0"
)

# Model and tokenizer will be loaded at startup
model = None
tokenizer = None
pipeline = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    # In a real app, you would get the user from a database
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# This would be a database in a real app
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "hashed_password": get_password_hash("testpassword"),
        "disabled": False,
    }
}

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return User(**user_dict)

@app.on_event("startup")
async def startup_event():
    """Initialize model and tokenizer on startup"""
    global model, tokenizer, pipeline
    
    model_path = os.getenv("MODEL_PATH", "/path/to/your/fine-tuned/model")
    
    try:
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.post("/token")
async def login_for_access_token():
    """Generate a JWT token for authentication"""
    # In a real app, verify the username and password from a database
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": "testuser"},  # Replace with actual user
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate text using the fine-tuned model"""
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        start_time = datetime.now()
        
        # Generate text
        output = pipeline(
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "generated_text": output[0]["generated_text"],
            "generation_time": generation_time,
            "model": os.getenv("MODEL_NAME", "fine-tuned-model")
        }
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating text: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        workers=1
    )
