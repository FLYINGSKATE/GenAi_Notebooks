# RAG System with LangChain and ChromaDB

This is a complete Retrieval-Augmented Generation (RAG) system built with LangChain and ChromaDB. It allows you to process documents, create vector embeddings, and answer questions based on the document content.

## Features

- **Document Processing**: Supports multiple document formats (PDF, TXT, DOCX, MD)
- **Vector Storage**: Uses ChromaDB for efficient vector similarity search
- **Question Answering**: Leverages language models for accurate responses
- **Interactive CLI**: Easy-to-use command-line interface
- **Configurable**: Customize chunking, embedding models, and more

## Prerequisites

- Python 3.8+
- pip

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag_system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   # For OpenAI models
   export OPENAI_API_KEY="your-openai-api-key"
   
   # For Hugging Face models (optional)
   export HUGGINGFACEHUB_API_TOKEN="your-huggingface-token"
   ```

## Usage

### 1. Process Documents

To process documents and create the vector store:

```bash
python cli.py process /path/to/your/documents \
    --chunk-size 1000 \
    --chunk-overlap 200
```

### 2. Query the System

#### Single Query:
```bash
python cli.py query "Your question here" --show-sources
```

#### Interactive Mode:
```bash
python cli.py interactive
```

In interactive mode, you can:
- Type your questions to get answers
- Toggle source display with `/sources on` or `/sources off`
- Exit with `exit` or `quit`

## Configuration

Edit `config.py` to customize:
- Embedding model
- Chunk size and overlap
- Vector store location
- Number of results to return

## Project Structure

```
rag_system/
├── data/                   # Directory for storing documents
├── vector_store/           # Vector store database
├── src/
│   ├── document_processor.py  # Document loading and processing
│   ├── vector_store_manager.py # Vector store management
│   └── rag_system.py       # Main RAG system implementation
├── cli.py                  # Command-line interface
├── config.py              # Configuration settings
└── README.md              # This file
```

## Customization

### Using Different Embedding Models

Edit `config.py` to change the embedding model:

```python
# For OpenAI embeddings (requires API key)
EMBEDDING_MODEL = "text-embedding-ada-002"

# For local models
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

### Using Different LLMs

Modify the `_initialize_llm` method in `rag_system.py` to use different language models.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Vector Store Errors**:
   - Delete the `vector_store` directory and reprocess documents
   - Ensure you have write permissions in the project directory

3. **API Key Issues**:
   - Verify your API keys are set in environment variables
   - Check for typos in the keys

## License

This project is licensed under the MIT License.
