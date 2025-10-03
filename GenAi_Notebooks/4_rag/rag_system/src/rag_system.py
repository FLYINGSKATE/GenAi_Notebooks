import logging
from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from .vector_store_manager import VectorStoreManager
from config import LLM_MODEL, TOP_K_RESULTS

logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system class that integrates document processing, vector storage, and question answering."""
    
    def __init__(self, vector_store_manager: Optional[VectorStoreManager] = None):
        """Initialize the RAG system.
        
        Args:
            vector_store_manager: Optional pre-configured vector store manager
        """
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        self.llm = self._initialize_llm()
        self.qa_chain = None
        
    def _initialize_llm(self, temperature: float = 0.0):
        """Initialize the language model.
        
        Args:
            temperature: Controls randomness in the model's output
            
        Returns:
            The initialized language model
        """
        try:
            return OpenAI(
                model_name=LLM_MODEL,
                temperature=temperature,
                max_tokens=1000
            )
        except Exception as e:
            logger.error(f"Failed to initialize language model: {str(e)}")
            raise
    
    def _create_qa_chain(self):
        """Create the question-answering chain with a custom prompt."""
        if self.vector_store_manager.vector_store is None:
            self.vector_store_manager.vector_store = self.vector_store_manager.load_vector_store()
            
        if self.vector_store_manager.vector_store is None:
            raise ValueError("No vector store available. Please load or create one first.")
        
        # Define the prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}
        
        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store_manager.vector_store.as_retriever(
                search_kwargs={"k": TOP_K_RESULTS}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return self.qa_chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing the answer and source documents
        """
        if self.qa_chain is None:
            self.qa_chain = self._create_qa_chain()
            
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"].strip(),
                "sources": result["source_documents"]
            }
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            return {
                "answer": "Sorry, I encountered an error while processing your question.",
                "sources": []
            }
    
    def get_similar_documents(self, query: str, k: int = None) -> List[Document]:
        """Get documents similar to the query.
        
        Args:
            query: The query string
            k: Number of results to return (defaults to config value)
            
        Returns:
            List of similar documents
        """
        return self.vector_store_manager.similarity_search(
            query, 
            k=k or TOP_K_RESULTS
        )
    
    def clear(self) -> None:
        """Clear the vector store and reset the QA chain."""
        self.vector_store_manager.delete_all()
        self.qa_chain = None


def initialize_rag_system() -> RAGSystem:
    """Initialize and return a configured RAG system."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the RAG system
    rag_system = RAGSystem()
    
    # Try to load existing vector store
    rag_system.vector_store_manager.load_vector_store()
    
    return rag_system


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize the RAG system
        print("Initializing RAG system...")
        rag = initialize_rag_system()
        
        # Example query
        question = "What is the main topic of the documents?"
        print(f"\nQuestion: {question}")
        
        # Get answer
        result = rag.query(question)
        print(f"\nAnswer: {result['answer']}")
        
        # Show sources
        if result["sources"]:
            print("\nSources:")
            for i, doc in enumerate(result["sources"], 1):
                print(f"\nDocument {i}:")
                print("-" * 50)
                print(doc.page_content[:300] + "...")
        
    except Exception as e:
        print(f"Error: {str(e)}")
