# RAG Pipeline User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Running the Pipeline](#running-the-pipeline)
5. [Using the Semantic Search](#using-the-semantic-search)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Introduction

The RAG (Retrieval-Augmented Generation) Pipeline is designed to extract content from the AI Robotics textbook website, chunk it into semantic pieces, generate vector embeddings using Cohere, and store them in Qdrant for semantic search capabilities.

The pipeline enables users to ask questions about the book content and receive contextually relevant answers based on semantic similarity.

## Prerequisites

Before running the RAG Pipeline, you need:

1. **Python 3.8+**: The pipeline is written in Python
2. **Cohere API Key**: Required for embedding generation
3. **Qdrant Cloud Account**: Required for vector storage
4. **Website Access**: The book website must be publicly accessible

### Required Python Packages
The pipeline requires the following Python packages:
- requests>=2.31.0
- beautifulsoup4>=4.12.2
- cohere>=4.9.0
- qdrant-client>=1.9.0
- python-dotenv>=1.0.0
- pydantic>=2.5.0
- tqdm>=4.66.1
- lxml>=4.9.3

## Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies
```bash
cd backend/rag_pipeline
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the `backend/rag_pipeline/` directory:

```bash
cp .env.example .env
```

Then edit the `.env` file with your specific configuration:

```env
# Cohere API Configuration
COHERE_API_KEY=your_cohere_api_key_here
COHERE_MODEL=embed-multilingual-v3.0

# Qdrant Configuration
QDRANT_URL=your_qdrant_cluster_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=book_content

# Website URL for content extraction
WEBSITE_URL=https://your-book-site.github.io

# Chunking parameters
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# Rate limiting parameters
COHERE_RATE_LIMIT_DELAY=1.0
```

### 4. Validate Configuration
Before running the pipeline, validate your configuration:

```python
from rag_pipeline.utils.config_validator import ConfigValidator

validator = ConfigValidator()
is_valid, errors = validator.validate_config()
if is_valid:
    print("Configuration is valid!")
else:
    print(f"Configuration errors: {errors}")
```

## Running the Pipeline

### Complete Pipeline Execution

To run the complete RAG pipeline:

```bash
python run_rag_pipeline.py
```

This will:
1. Extract content from the configured website
2. Chunk the content semantically
3. Generate embeddings using Cohere
4. Store embeddings in Qdrant
5. Validate the results

### Custom Parameters

You can specify custom parameters:

```bash
# Run with a specific URL
python run_rag_pipeline.py --url https://custom-book-site.com

# Run without validation (faster)
python run_rag_pipeline.py --validate

# Run incremental pipeline (for updates)
python run_rag_pipeline.py --incremental
```

### Programmatic Usage

You can also run the pipeline programmatically:

```python
from rag_pipeline.pipelines.rag_pipeline import RAGPipeline

# Initialize the pipeline
rag_pipeline = RAGPipeline()

# Run the complete pipeline
results = rag_pipeline.run_complete_pipeline()
print(f"Pipeline completed with status: {results['status']}")
print(f"Content chunks processed: {results['content_chunks_count']}")
print(f"Embeddings generated: {results['embeddings_count']}")
```

## Using the Semantic Search

Once the pipeline has been run and embeddings are stored, you can perform semantic searches:

### Command Line Search

```bash
python -c "
from rag_pipeline.pipelines.rag_pipeline import RAGPipeline
pipeline = RAGPipeline()
results = pipeline.query('What is ROS 2?', top_k=3)
for i, result in enumerate(results):
    print(f'{i+1}. {result.content[:200]}... (Score: {result.similarity_score:.3f})')
"
```

### Programmatic Search

```python
from rag_pipeline.services.semantic_search import SemanticSearch

# Initialize semantic search
semantic_search = SemanticSearch()

# Perform a search
results = semantic_search.search("What is ROS 2?", top_k=5)

# Display results
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f"  Similarity: {result.similarity_score:.3f}")
    print(f"  Content: {result.content[:150]}...")
    print(f"  Source: {result.source_url}")
    print()
```

### Search with Context

For more detailed results with context:

```python
from rag_pipeline.services.semantic_search import SemanticSearch

semantic_search = SemanticSearch()
results = semantic_search.search_with_context(
    query="Explain robot navigation",
    top_k=3,
    context_window=2
)

for result in results:
    print(f"Content: {result['content'][:100]}...")
    print(f"Context: {result['context']}")
    print(f"Similarity: {result['similarity_score']:.3f}")
    print()
```

## Configuration

### Chunking Parameters

- `CHUNK_SIZE`: Size of each content chunk in characters (default: 512)
  - Larger chunks provide more context but may mix unrelated topics
  - Smaller chunks are more precise but may lose context

- `CHUNK_OVERLAP`: Overlap between consecutive chunks (default: 64)
  - Helps preserve context at chunk boundaries
  - Higher overlap means more redundancy but better context preservation

### API Rate Limits

- `COHERE_RATE_LIMIT_DELAY`: Delay between Cohere API calls in seconds (default: 1.0)
  - Adjust based on your Cohere API plan's rate limits
  - Increase if you encounter rate limit errors

### Collection Configuration

- `QDRANT_COLLECTION_NAME`: Name of the Qdrant collection (default: book_content)
  - Should be unique for different content sources
  - Can be changed to separate different book versions

## Troubleshooting

### Common Issues

#### 1. API Key Errors
**Problem**: "Invalid API key" or authentication errors
**Solution**:
- Verify your Cohere and Qdrant API keys are correct
- Check that your `.env` file is properly formatted
- Ensure your API keys have the necessary permissions

#### 2. Website Access Issues
**Problem**: "Failed to extract content" or network errors
**Solution**:
- Verify the website URL is accessible
- Check if the website has anti-bot measures
- Ensure your IP isn't blocked by the website

#### 3. Rate Limit Errors
**Problem**: "Rate limit exceeded" errors from Cohere
**Solution**:
- Increase the `COHERE_RATE_LIMIT_DELAY` value
- Consider upgrading your Cohere API plan for higher limits
- Run the pipeline during off-peak hours

#### 4. Memory Issues
**Problem**: "Out of memory" errors during processing
**Solution**:
- Reduce the batch size in the embedding generation
- Process smaller sections of the website at a time
- Increase system memory if possible

### Debugging Tips

1. **Enable Debug Logging**:
   ```python
   import logging
   from rag_pipeline.utils.logging_config import setup_logging

   # Set debug level
   logger = setup_logging(log_level=logging.DEBUG)
   ```

2. **Validate Configuration First**:
   Always validate your configuration before running the full pipeline:
   ```python
   from rag_pipeline.utils.config_validator import ConfigValidator
   validator = ConfigValidator()
   is_valid, errors = validator.validate_config()
   ```

3. **Test Individual Components**:
   Test each pipeline component separately before running the full pipeline.

## Best Practices

### 1. Chunking Strategy
- Use semantic boundaries (headings, paragraphs) rather than arbitrary character counts
- Balance chunk size: large enough to maintain context, small enough to be specific
- Test different chunk sizes to find the optimal balance for your content

### 2. API Usage
- Monitor your API usage to stay within rate limits
- Implement retry logic for transient failures
- Use batch processing where possible to optimize API calls

### 3. Content Quality
- Clean and normalize content before embedding
- Remove boilerplate content (navigation, headers, footers)
- Preserve important structural information (chapter, section)

### 4. Error Handling
- Implement graceful degradation when parts of the pipeline fail
- Log detailed information for debugging
- Monitor pipeline metrics for performance optimization

### 5. Validation
- Validate content extraction quality
- Monitor embedding generation success rates
- Test search result relevance regularly

## Monitoring and Maintenance

### Pipeline Metrics
The pipeline provides various metrics:
- Content extraction completeness
- Chunking quality and semantic coherence
- Embedding generation success rate
- Storage reliability
- Search performance and accuracy

### Regular Maintenance
- Periodically re-run the pipeline for content updates
- Monitor API usage and costs
- Update configuration as content or requirements change
- Review search result quality and adjust parameters as needed

## Support

For additional support:
- Check the API documentation for detailed method information
- Review the test files for usage examples
- Consult the source code for implementation details
- Reach out to the development team for complex issues