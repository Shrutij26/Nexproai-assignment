import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Using small embeddings for cost efficiency
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Chunking Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # LanceDB Configuration
    LANCEDB_URI = os.getenv("LANCEDB_URI", "./data/lancedb")
    TABLE_NAME = os.getenv("TABLE_NAME", "document_chunks")
    
    # Data directory
    DATA_DIR = os.getenv("DATA_DIR", "./data/raw")
