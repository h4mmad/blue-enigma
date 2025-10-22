"""
Configuration module for Hybrid AI Travel Assistant.
Loads credentials and settings from environment variables (.env file).
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env.local file
load_dotenv(".env.local")

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "vietnam-travel")
PINECONE_VECTOR_DIM = int(os.getenv("PINECONE_VECTOR_DIM", "1536"))

# Cache Configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_STATS_LOGGING = os.getenv("CACHE_STATS_LOGGING", "true").lower() == "true"

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Embedding Cache Configuration (exact match, for storing embeddings)
EMBEDDING_CACHE_ENABLED = os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
EMBEDDING_CACHE_TTL = os.getenv("EMBEDDING_CACHE_TTL")  # Optional TTL in seconds
if EMBEDDING_CACHE_TTL:
    EMBEDDING_CACHE_TTL = int(EMBEDDING_CACHE_TTL)

# Semantic Cache Configuration (similarity-based, for LLM responses)
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.1"))
SEMANTIC_CACHE_TTL = os.getenv("SEMANTIC_CACHE_TTL")  # Optional TTL in seconds
if SEMANTIC_CACHE_TTL:
    SEMANTIC_CACHE_TTL = int(SEMANTIC_CACHE_TTL)

# Validation: Check that required API keys are set
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it in .env file.")
if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD not found in environment variables. Please set it in .env file.")
