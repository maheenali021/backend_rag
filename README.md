# AI Robotics Textbook RAG Agent

This is a Retrieval-Augmented Generation (RAG) agent that provides answers based on the AI Robotics textbook content. The system uses OpenRouter API for language modeling and Qdrant for vector storage and semantic search.

## About

This RAG system allows users to ask questions about the AI Robotics textbook and receive accurate answers based on the book's content with proper source attribution.

## Features

- Semantic search through textbook content
- Context-aware question answering
- Source attribution for all responses
- Session management for conversations
- Hallucination prevention through grounding

## API Endpoints

-  - Main chat endpoint for asking questions
-  - Create a new conversation session
-  - Get conversation history
-  - Clear conversation history
-  - Health check endpoint

## Environment Variables

This application requires the following environment variables to be set:

-  - Your OpenRouter API key (required)
-  - Qdrant vector database URL (required)
-  - Qdrant API key (required) 
-  - Name of the Qdrant collection (default: book_content)
-  - Cohere API key for embeddings (required)

## Usage

Send a POST request to  with a JSON body containing a  field:

```json
{
  "query": "What is this book about?"
}
```

## Example Response

```json
{
  "response": "The book is about Physical AI & Humanoid Robotics...", 
  "query": "What is this book about?",
  "retrieved_chunks": [...],
  "source_attribution": [...],
  "confidence_score": 0.46,
  "has_sufficient_context": true
}
```

## Deployment

This application is designed for deployment on Hugging Face Spaces using Docker.

## License

This project is licensed under the terms specified in the original repository.
