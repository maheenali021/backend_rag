"""
Test script to verify Qdrant connection and vector data
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from rag_pipeline.config import Config

# Load environment variables
load_dotenv()

def test_qdrant_connection():
    """Test Qdrant connection and check collections"""
    print("Testing Qdrant connection...")

    # Check if environment variables are set
    print(f"QDRANT_URL: {'SET' if os.getenv('QDRANT_URL') else 'NOT SET'}")
    print(f"QDRANT_API_KEY: {'SET' if os.getenv('QDRANT_API_KEY') else 'NOT SET'}")
    print(f"QDRANT_COLLECTION_NAME: {os.getenv('QDRANT_COLLECTION_NAME', 'NOT SET')}")

    if not os.getenv('QDRANT_URL') or not os.getenv('QDRANT_API_KEY'):
        print("ERROR: Qdrant URL or API key not set in environment variables")
        return False

    try:
        # Initialize Qdrant client
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            prefer_grpc=False
        )

        # Test connection by listing collections
        collections = client.get_collections()
        print(f"Connected successfully! Available collections: {[coll.name for coll in collections.collections]}")

        # Check if the expected collection exists
        collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'book_content')
        collection_exists = any(coll.name == collection_name for coll in collections.collections)

        if collection_exists:
            print(f"OK Collection '{collection_name}' exists")

            # Get collection info
            collection_info = client.get_collection(collection_name)
            print(f"  Points in collection: {collection_info.points_count}")
            print(f"  Config: {collection_info.config}")
        else:
            print(f"X Collection '{collection_name}' does not exist")
            print(f"  Available collections: {[coll.name for coll in collections.collections]}")

        return True

    except Exception as e:
        print(f"ERROR connecting to Qdrant: {str(e)}")
        return False

def test_cohere_connection():
    """Test Cohere connection"""
    print("\nTesting Cohere connection...")

    print(f"COHERE_API_KEY: {'SET' if os.getenv('COHERE_API_KEY') else 'NOT SET'}")

    if not os.getenv('COHERE_API_KEY'):
        print("ERROR: Cohere API key not set in environment variables")
        return False

    try:
        import cohere
        co = cohere.Client(os.getenv('COHERE_API_KEY'))

        # Test with a simple embed call
        response = co.embed(
            texts=["test"],
            model="embed-multilingual-v3.0",
            input_type="search_query"
        )

        print("OK Cohere connection successful!")
        print(f"  Embedding dimension: {len(response.embeddings[0])}")
        return True

    except Exception as e:
        print(f"ERROR connecting to Cohere: {str(e)}")
        return False

if __name__ == "__main__":
    print("Vector Database and Embedding Connection Test")
    print("="*50)

    qdrant_ok = test_qdrant_connection()
    cohere_ok = test_cohere_connection()

    print("\n" + "="*50)
    if qdrant_ok and cohere_ok:
        print("OK All connections are working properly!")
    else:
        print("X Some connections failed. Please check your configuration.")