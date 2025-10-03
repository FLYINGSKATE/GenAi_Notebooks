from pathlib import Path
from typing import List, Union
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles loading and processing of various document types."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".md": UnstructuredMarkdownLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".doc": UnstructuredWordDocumentLoader,
        }
    
    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a single document from file path."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        try:
            loader = self.supported_formats[file_ext](str(file_path))
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def load_documents(self, directory: Union[str, Path]) -> List[Document]:
        """Load all supported documents from a directory."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")
            
        all_docs = []
        for ext in self.supported_formats:
            for file_path in directory.glob(f"*{ext}"):
                try:
                    docs = self.load_document(file_path)
                    all_docs.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {file_path}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {str(e)}")
        
        return all_docs
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for processing."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    processor = DocumentProcessor()
    
    # Load a single document
    try:
        docs = processor.load_document("example.pdf")
        print(f"Loaded {len(docs)} pages from example.pdf")
        
        # Chunk the documents
        chunks = processor.chunk_documents(docs)
        print(f"Split into {len(chunks)} chunks")
        
    except Exception as e:
        print(f"Error: {str(e)}")
