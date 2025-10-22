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

# Redis Configuration (for embedding cache)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_CACHE_TTL = os.getenv("REDIS_CACHE_TTL")  # Optional TTL in seconds (None = no expiration)
if REDIS_CACHE_TTL:
    REDIS_CACHE_TTL = int(REDIS_CACHE_TTL)

# Validation: Check that required API keys are set
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it in .env file.")
if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD not found in environment variables. Please set it in .env file.")
