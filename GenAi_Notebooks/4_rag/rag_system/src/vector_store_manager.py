from typing import List, Optional
from pathlib import Path
import logging
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from .document_processor import DocumentProcessor
from config import VECTOR_STORE_DIR, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages the vector store for document storage and retrieval."""
    
    def __init__(self, persist_directory: str = str(VECTOR_STORE_DIR)):
        """Initialize the vector store manager.
        
        Args:
            persist_directory: Directory to persist the vector store
        """
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None
        
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create a new vector store from documents.
        
        Args:
            documents: List of documents to add to the vector store
            
        Returns:
            Chroma: The created vector store
        """
        logger.info(f"Creating new vector store with {len(documents)} documents")
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vector_store.persist()
        return self.vector_store
    
    def load_vector_store(self) -> Optional[Chroma]:
        """Load an existing vector store from disk.
        
        Returns:
            Optional[Chroma]: The loaded vector store, or None if not found
        """
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            # Verify the vector store is accessible
            _ = self.vector_store._collection.count()
            logger.info("Successfully loaded existing vector store")
            return self.vector_store
        except Exception as e:
            logger.warning(f"Could not load existing vector store: {str(e)}")
            return None
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        if self.vector_store is None:
            return self.create_vector_store(documents).add_documents(documents)
        
        return self.vector_store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: The query string
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        if self.vector_store is None:
            self.vector_store = self.load_vector_store()
            if self.vector_store is None:
                return []
                
        return self.vector_store.similarity_search(query, k=k)
    
    def delete_all(self) -> None:
        """Delete all vectors from the store."""
        if self.vector_store is not None:
            self.vector_store.delete_collection()
            self.vector_store = None


def process_and_store_documents(directory: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> VectorStoreManager:
    """Helper function to process documents and create a vector store.
    
    Args:
        directory: Directory containing documents
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        VectorStoreManager: The configured vector store manager
    """
    # Initialize components
    doc_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vector_store_manager = VectorStoreManager()
    
    # Load and process documents
    documents = doc_processor.load_documents(directory)
    chunks = doc_processor.chunk_documents(documents)
    
    # Create and populate vector store
    vector_store_manager.create_vector_store(chunks)
    
    return vector_store_manager


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    try:
        # Process documents and create vector store
        vs_manager = process_and_store_documents("data")
        
        # Perform a search
        query = "What is the main topic?"
        results = vs_manager.similarity_search(query)
        
        print(f"\nResults for query: '{query}'")
        for i, doc in enumerate(results, 1):
            print(f"\nDocument {i}:")
            print("-" * 50)
            print(doc.page_content[:500] + "...")
            
    except Exception as e:
        print(f"Error: {str(e)}")
