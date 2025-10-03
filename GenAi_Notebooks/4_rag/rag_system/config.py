from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "gpt-3.5-turbo-instruct"  # Using OpenAI's model as an example

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K_RESULTS = 4

# API Keys (should be loaded from environment variables in production)
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # Replace with your actual key
