# RAG Pipeline API Documentation

## Overview
The RAG Pipeline provides programmatic access to content extraction, chunking, embedding generation, and semantic search capabilities for the AI Robotics textbook.

## Core Services

### ContentExtractor
Handles extraction of content from the deployed website.

#### Methods
- `extract_content_from_url(url: str) -> BookContent`
  - Extracts content from a single URL
  - Parameters: `url` - URL to extract content from
  - Returns: `BookContent` object with extracted content

- `extract_content_from_website(base_url: str = None) -> List[BookContent]`
  - Extracts content from the entire website by finding all relevant URLs
  - Parameters: `base_url` - Base URL of the website (optional, defaults to config)
  - Returns: List of `BookContent` objects

- `validate_extraction(book_content: BookContent) -> bool`
  - Validates the extracted content
  - Parameters: `book_content` - BookContent object to validate
  - Returns: True if content is valid, False otherwise

### ContentChunker
Handles chunking of content into semantically meaningful pieces.

#### Methods
- `chunk_content(book_content: BookContent, chunk_size: int = None, overlap_size: int = None) -> List[ContentChunk]`
  - Chunks the content of a BookContent object
  - Parameters:
    - `book_content` - BookContent object to chunk
    - `chunk_size` - Size of each chunk (optional, defaults to config)
    - `overlap_size` - Size of overlap between chunks (optional, defaults to config)
  - Returns: List of `ContentChunk` objects

- `validate_chunks(content_chunks: List[ContentChunk], max_chunk_size: int = None) -> bool`
  - Validates that chunks meet size requirements
  - Parameters:
    - `content_chunks` - List of ContentChunk objects to validate
    - `max_chunk_size` - Maximum allowed chunk size (optional, defaults to config)
  - Returns: True if all chunks are valid, False otherwise

### EmbeddingGenerator
Generates vector embeddings using Cohere API.

#### Methods
- `generate_embeddings(content_chunks: List[ContentChunk], batch_size: int = 96) -> List[EmbeddingVector]`
  - Generate embeddings for a list of content chunks
  - Parameters:
    - `content_chunks` - List of ContentChunk objects to generate embeddings for
    - `batch_size` - Number of chunks to process in each batch
  - Returns: List of `EmbeddingVector` objects

- `validate_embeddings(embedding_vectors: List[EmbeddingVector]) -> bool`
  - Validate the generated embeddings
  - Parameters: `embedding_vectors` - List of EmbeddingVector objects to validate
  - Returns: True if all embeddings are valid, False otherwise

### VectorStorage
Handles storage and retrieval of embedding vectors in Qdrant.

#### Methods
- `store_embeddings(embedding_vectors: List[EmbeddingVector], metadata_list: List[Dict] = None, batch_size: int = 200) -> bool`
  - Store embedding vectors in Qdrant
  - Parameters:
    - `embedding_vectors` - List of EmbeddingVector objects to store
    - `metadata_list` - Optional list of metadata for each vector
    - `batch_size` - Number of vectors to store in each batch
  - Returns: True if storage was successful, False otherwise

- `search_similar(query_vector: List[float], top_k: int = 10) -> List[VectorRecord]`
  - Search for similar vectors in Qdrant
  - Parameters:
    - `query_vector` - Vector to search for similarity
    - `top_k` - Number of similar vectors to return
  - Returns: List of `VectorRecord` objects with similarity scores

### SemanticSearch
Provides semantic search capabilities using vector embeddings.

#### Methods
- `search(query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[VectorRecord]`
  - Perform semantic search for a query
  - Parameters:
    - `query` - Search query string
    - `top_k` - Number of results to return
    - `min_similarity` - Minimum similarity threshold
  - Returns: List of `VectorRecord` objects with similarity scores

- `search_with_context(query: str, top_k: int = 5, context_window: int = 2) -> List[Dict]`
  - Perform semantic search and return results with additional context
  - Parameters:
    - `query` - Search query string
    - `top_k` - Number of results to return
    - `context_window` - Number of sentences to include as context around matches
  - Returns: List of dictionaries with search results and context

## Main Pipeline Classes

### ExtractionPipeline
Orchestrates the content extraction and chunking process.

#### Methods
- `run_extraction_pipeline(url: str = None) -> List[ContentChunk]`
  - Run the complete extraction and chunking pipeline
  - Parameters: `url` - URL to extract content from (optional, defaults to config)
  - Returns: List of `ContentChunk` objects

- `validate_pipeline_output(book_contents: List[BookContent], content_chunks: List[ContentChunk]) -> bool`
  - Validate the output of the extraction pipeline
  - Parameters:
    - `book_contents` - List of BookContent objects
    - `content_chunks` - List of ContentChunk objects
  - Returns: True if output is valid, False otherwise

### RAGPipeline
Main orchestration class for the complete RAG pipeline.

#### Methods
- `run_complete_pipeline(website_url: str = None, validate: bool = True) -> Dict`
  - Run the complete RAG pipeline from content extraction to vector storage
  - Parameters:
    - `website_url` - URL of the website to process (optional, defaults to config)
    - `validate` - Whether to perform validation steps
  - Returns: Dictionary with pipeline results and metrics

- `query(search_query: str, top_k: int = 5) -> List[VectorRecord]`
  - Query the RAG pipeline for relevant content
  - Parameters:
    - `search_query` - Query string
    - `top_k` - Number of results to return
  - Returns: List of `VectorRecord` objects with relevant content

## Configuration

### Environment Variables
The pipeline uses the following environment variables:

- `COHERE_API_KEY`: API key for Cohere embedding service
- `QDRANT_URL`: URL for Qdrant Cloud instance
- `QDRANT_API_KEY`: API key for Qdrant Cloud
- `QDRANT_COLLECTION_NAME`: Name of the Qdrant collection (default: book_content)
- `WEBSITE_URL`: URL of the deployed book website (default: GitHub Pages URL)
- `CHUNK_SIZE`: Target size of each chunk (default: 512)
- `CHUNK_OVERLAP`: Size of overlap between chunks (default: 64)
- `COHERE_RATE_LIMIT_DELAY`: Delay between Cohere API calls (default: 1.0)

### Configuration Validation
Use the `ConfigValidator` class to validate your configuration before running the pipeline:

```python
from rag_pipeline.utils.config_validator import ConfigValidator

validator = ConfigValidator()
is_valid, errors = validator.validate_config()
if not is_valid:
    print(f"Configuration errors: {errors}")
```

## Usage Examples

### Running the Complete Pipeline
```python
from rag_pipeline.pipelines.rag_pipeline import RAGPipeline

# Initialize the pipeline
rag_pipeline = RAGPipeline()

# Run the complete pipeline
results = rag_pipeline.run_complete_pipeline()
print(f"Pipeline completed with status: {results['status']}")
```

### Performing a Semantic Search
```python
from rag_pipeline.services.semantic_search import SemanticSearch

# Initialize semantic search
semantic_search = SemanticSearch()

# Perform a search
results = semantic_search.search("What is ROS 2?", top_k=5)
for result in results:
    print(f"Similarity: {result.similarity_score}")
    print(f"Content: {result.content[:100]}...")
```

### Extracting and Chunking Content
```python
from rag_pipeline.pipelines.extraction_pipeline import ExtractionPipeline

# Initialize the extraction pipeline
extraction_pipeline = ExtractionPipeline()

# Extract and chunk content from a URL
content_chunks = extraction_pipeline.run_extraction_pipeline("https://your-book-site.github.io")
print(f"Extracted {len(content_chunks)} content chunks")
```

## Error Handling

The RAG Pipeline includes comprehensive error handling:

- `RAGPipelineError`: Base exception class for RAG Pipeline errors
- `ContentExtractionError`: Raised when content extraction fails
- `ContentChunkingError`: Raised when content chunking fails
- `EmbeddingGenerationError`: Raised when embedding generation fails
- `VectorStorageError`: Raised when vector storage operations fail
- `ConfigurationError`: Raised when configuration is invalid or missing
- `APIError`: Raised when API calls fail
- `ValidationError`: Raised when validation fails

## Logging

The pipeline uses structured logging with different levels:
- `DEBUG`: Detailed information for debugging
- `INFO`: General information about pipeline execution
- `WARNING`: Issues that don't stop execution
- `ERROR`: Errors that may affect results
- `CRITICAL`: Serious errors that stop execution

Logs are written to both console and file (`rag_pipeline.log` by default).