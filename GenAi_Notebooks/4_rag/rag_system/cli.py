#!/usr/bin/env python3
"""Command-line interface for the RAG system."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_system import initialize_rag_system, RAGSystem
from src.vector_store_manager import VectorStoreManager, process_and_store_documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def process_documents(args):
    """Process documents and create/update the vector store."""
    try:
        print(f"Processing documents from: {args.directory}")
        process_and_store_documents(
            directory=args.directory,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        print("Documents processed successfully!")
    except Exception as e:
        print(f"Error processing documents: {str(e)}", file=sys.stderr)
        sys.exit(1)

def query_rag(args, rag_system: RAGSystem):
    """Query the RAG system."""
    try:
        result = rag_system.query(args.query)
        
        print("\n" + "=" * 80)
        print(f"QUESTION: {args.query}")
        print("=" * 80)
        print("\nANSWER:")
        print(result["answer"])
        
        if result["sources"] and args.show_sources:
            print("\n" + "-" * 80)
            print("SOURCES:")
            for i, doc in enumerate(result["sources"], 1):
                print(f"\n[{i}] Source: {doc.metadata.get('source', 'Unknown')}")
                print("-" * 40)
                print(doc.page_content)
        
        print("\n" + "=" * 80 + "\n")
        
    except Exception as e:
        print(f"Error querying RAG system: {str(e)}", file=sys.stderr)
        sys.exit(1)

def interactive_mode(rag_system: RAGSystem):
    """Run the RAG system in interactive mode."""
    print("\n" + "=" * 60)
    print("RAG System - Interactive Mode")
    print("Type 'exit' or 'quit' to end the session")
    print("Type '/sources on' to show sources, '/sources off' to hide them")
    print("=" * 60 + "\n")
    
    show_sources = True
    
    while True:
        try:
            # Get user input
            user_input = input("\nYour question: ").strip()
            
            # Check for commands
            if user_input.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
            elif user_input.lower() == '/sources on':
                show_sources = True
                print("Source display enabled")
                continue
            elif user_input.lower() == '/sources off':
                show_sources = False
                print("Source display disabled")
                continue
            elif not user_input:
                continue
                
            # Process the query
            args = argparse.Namespace(query=user_input, show_sources=show_sources)
            query_rag(args, rag_system)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="RAG System Command Line Interface")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process documents')
    process_parser.add_argument('directory', help='Directory containing documents to process')
    process_parser.add_argument('--chunk-size', type=int, default=1000, help='Size of text chunks')
    process_parser.add_argument('--chunk-overlap', type=int, default=200, help='Overlap between chunks')
    process_parser.set_defaults(func=process_documents)
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('query', help='The question to ask')
    query_parser.add_argument('--show-sources', action='store_true', help='Show source documents')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Run in interactive mode')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize RAG system for query/interactive modes
    rag_system = None
    if hasattr(args, 'func'):
        # Process command - no need to initialize RAG system
        args.func(args)
    else:
        # Query or interactive mode - initialize RAG system
        try:
            rag_system = initialize_rag_system()
            
            if args.command == 'query':
                query_rag(args, rag_system)
            elif args.command == 'interactive':
                interactive_mode(rag_system)
            else:
                parser.print_help()
                
        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
