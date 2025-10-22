"""
Redis Cache Module
Provides simple Redis-based caching for embeddings with exact text matching.
Designed for scalable, persistent embedding storage to avoid redundant OpenAI API calls.
"""
import redis
import hashlib
import json
import time
import logging
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Cache:
    """
    Simple Redis-based cache for embedding vectors with exact text matching.

    Features:
    - Persistent storage across application restarts
    - Exact text matching (normalized for consistency)
    - TTL (Time To Live) support for automatic expiration
    - Cache hit/miss statistics tracking
    - JSON serialization for embedding vectors
    - Scalable Redis backend
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        default_ttl: Optional[int] = None,
        key_prefix: str = "embedding"
    ):
        """
        Initialize the Redis cache.

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number (0-15)
            default_ttl: Default time-to-live in seconds (None = no expiration)
            key_prefix: Prefix for all cache keys
        """
        self.host = host
        self.port = port
        self.db = db
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix

        # Statistics tracking (in-memory, per session)
        self._stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0,
            'api_calls_saved': 0,
            'sets': 0
        }

        # Initialize Redis connection
        self._connect()

        logger.info(f"Redis Cache initialized: {host}:{port}/{db}, prefix={key_prefix}, ttl={default_ttl}")

    def _connect(self) -> None:
        """
        Establish connection to Redis server.

        Raises:
            redis.ConnectionError: If unable to connect to Redis
        """
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False,  # Handle binary data for embeddings
                socket_connect_timeout=5,
                socket_timeout=5
            )

            # Test connection
            self.client.ping()
            logger.info(f"Successfully connected to Redis at {self.host}:{self.port}")

        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            raise

    def is_connected(self) -> bool:
        """
        Check if Redis connection is active.

        Returns:
            True if connected, False otherwise
        """
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent cache keys.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text (lowercase, stripped, collapsed whitespace)
        """
        # Convert to lowercase
        normalized = text.lower()

        # Strip leading/trailing whitespace
        normalized = normalized.strip()

        # Collapse multiple spaces into single space
        normalized = ' '.join(normalized.split())

        return normalized

    def _generate_cache_key(self, text: str) -> str:
        """
        Generate a deterministic cache key from text.

        Args:
            text: Input text

        Returns:
            Full cache key with prefix (e.g., "embedding:abc123...")
        """
        # Normalize text first
        normalized = self._normalize_text(text)

        # Generate SHA256 hash
        hash_object = hashlib.sha256(normalized.encode('utf-8'))
        hash_key = hash_object.hexdigest()

        # Add prefix
        cache_key = f"{self.key_prefix}:{hash_key}"

        return cache_key

    def get(self, text: str) -> Optional[List[float]]:
        """
        Retrieve embedding from cache if available.

        Args:
            text: Query text to look up (will be normalized)

        Returns:
            Cached embedding vector if found, None otherwise
        """
        self._stats['total_requests'] += 1

        # Generate cache key
        cache_key = self._generate_cache_key(text)

        try:
            # Check if key exists
            value_bytes = self.client.get(cache_key)

            if value_bytes is not None:
                # Cache hit
                self._stats['hits'] += 1
                self._stats['api_calls_saved'] += 1

                # Deserialize JSON
                cached_data = json.loads(value_bytes.decode('utf-8'))

                logger.debug(f"Cache HIT for text: '{text[:50]}...'")

                # Return the embedding vector
                return cached_data.get('embedding')
            else:
                # Cache miss
                self._stats['misses'] += 1

                logger.debug(f"Cache MISS for text: '{text[:50]}...'")

                return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize cached data: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def set(
        self,
        text: str,
        embedding: List[float],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store embedding in cache.

        Args:
            text: Original query text (will be normalized for key generation)
            embedding: Embedding vector to cache
            ttl: Time-to-live in seconds (None = use default_ttl)

        Returns:
            True if successful, False otherwise
        """
        # Generate cache key
        cache_key = self._generate_cache_key(text)

        # Use default TTL if not specified
        if ttl is None:
            ttl = self.default_ttl

        try:
            # Create cache entry with metadata
            cache_entry = {
                'embedding': embedding,
                'text': text[:200],  # Store preview of original text
                'normalized': self._normalize_text(text)[:200],
                'timestamp': time.time(),
                'dimension': len(embedding)
            }

            # Serialize to JSON
            value_json = json.dumps(cache_entry)
            value_bytes = value_json.encode('utf-8')

            # Store in Redis with optional TTL
            if ttl is not None:
                self.client.setex(cache_key, ttl, value_bytes)
            else:
                self.client.set(cache_key, value_bytes)

            self._stats['sets'] += 1

            logger.debug(f"Cache SET for text: '{text[:50]}...' (TTL: {ttl})")

            return True

        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize embedding: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    # Backward compatibility aliases
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Retrieve embedding from cache (alias for get()).

        Args:
            text: Query text to look up

        Returns:
            Cached embedding vector if found, None otherwise
        """
        return self.get(text)

    def set_embedding(
        self,
        text: str,
        embedding: List[float],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store embedding in cache (alias for set()).

        Args:
            text: Original query text
            embedding: Embedding vector to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful, False otherwise
        """
        return self.set(text, embedding, ttl=ttl)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current cache statistics.

        Returns:
            Dictionary with cache statistics including Redis info
        """
        stats = self._stats.copy()

        # Calculate derived metrics
        total_requests = stats['total_requests']
        if total_requests > 0:
            stats['hit_rate'] = (stats['hits'] / total_requests) * 100
            stats['miss_rate'] = (stats['misses'] / total_requests) * 100
        else:
            stats['hit_rate'] = 0.0
            stats['miss_rate'] = 0.0

        # Get Redis info
        try:
            # Count keys with our prefix
            pattern = f"{self.key_prefix}:*"
            cursor = 0
            key_count = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                key_count += len(keys)
                if cursor == 0:
                    break
            stats['redis_keys_count'] = key_count

            # Get Redis memory usage info
            info = self.client.info('memory')
            stats['redis_memory_used_mb'] = info.get('used_memory', 0) / (1024 * 1024)

        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            stats['redis_keys_count'] = 0
            stats['redis_memory_used_mb'] = 0

        # Estimate cost savings (OpenAI text-embedding-3-small: $0.00002 per 1K tokens)
        # Assume average query is ~20 tokens
        avg_tokens_per_query = 20
        cost_per_1k_tokens = 0.00002
        estimated_savings = (stats['api_calls_saved'] * avg_tokens_per_query * cost_per_1k_tokens) / 1000
        stats['estimated_cost_savings_usd'] = estimated_savings

        return stats

    def print_stats(self) -> None:
        """
        Print formatted cache statistics to console.
        """
        stats = self.get_stats()

        print("\n" + "="*60)
        print("REDIS EMBEDDING CACHE STATISTICS")
        print("="*60)
        print(f"Total Requests:        {stats['total_requests']}")
        print(f"Cache Hits:            {stats['hits']} ({stats['hit_rate']:.1f}%)")
        print(f"Cache Misses:          {stats['misses']} ({stats['miss_rate']:.1f}%)")
        print(f"API Calls Saved:       {stats['api_calls_saved']}")
        print(f"Cache Sets:            {stats['sets']}")
        print(f"Redis Keys Count:      {stats['redis_keys_count']}")
        print(f"Redis Memory Used:     {stats['redis_memory_used_mb']:.2f} MB")
        print(f"Est. Cost Savings:     ${stats['estimated_cost_savings_usd']:.6f}")
        print("="*60 + "\n")

    def clear(self, reset_stats: bool = True) -> int:
        """
        Clear all cache entries with the current prefix.

        Args:
            reset_stats: Whether to reset statistics

        Returns:
            Number of keys deleted
        """
        try:
            # Find all keys with our prefix using SCAN (safer for production)
            pattern = f"{self.key_prefix}:*"
            cursor = 0
            deleted_count = 0

            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted_count += self.client.delete(*keys)
                if cursor == 0:
                    break

            # Reset stats if requested
            if reset_stats:
                self._stats = {
                    'hits': 0,
                    'misses': 0,
                    'total_requests': 0,
                    'api_calls_saved': 0,
                    'sets': 0
                }

            logger.info(f"Cache cleared: {deleted_count} keys deleted")

            return deleted_count

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    def delete(self, text: str) -> bool:
        """
        Delete a specific cache entry.

        Args:
            text: Text whose embedding should be deleted

        Returns:
            True if deleted, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(text)
            result = self.client.delete(cache_key)
            return result > 0

        except Exception as e:
            logger.error(f"Error deleting cache entry: {e}")
            return False

    def exists(self, text: str) -> bool:
        """
        Check if a cache entry exists for given text.

        Args:
            text: Text to check

        Returns:
            True if exists, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(text)
            return self.client.exists(cache_key) > 0

        except Exception as e:
            logger.error(f"Error checking cache existence: {e}")
            return False

    def get_ttl(self, text: str) -> Optional[int]:
        """
        Get the remaining TTL for a cache entry.

        Args:
            text: Text to check TTL for

        Returns:
            Remaining TTL in seconds, -1 if no expiry, None if key doesn't exist
        """
        try:
            cache_key = self._generate_cache_key(text)
            ttl = self.client.ttl(cache_key)

            if ttl == -2:  # Key doesn't exist
                return None
            elif ttl == -1:  # Key exists but has no expiry
                return -1
            else:
                return ttl

        except Exception as e:
            logger.error(f"Error getting TTL: {e}")
            return None

    def close(self) -> None:
        """
        Close the Redis connection.
        """
        try:
            self.client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __len__(self) -> int:
        """
        Return current number of cache keys.
        """
        try:
            pattern = f"{self.key_prefix}:*"
            cursor = 0
            key_count = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                key_count += len(keys)
                if cursor == 0:
                    break
            return key_count
        except Exception:
            return 0

    def __bool__(self) -> bool:
        """Cache object is always truthy."""
        return True

    def __repr__(self) -> str:
        """String representation of cache."""
        try:
            key_count = len(self)
            return (f"Cache(host={self.host}:{self.port}, db={self.db}, "
                    f"keys={key_count}, hits={self._stats['hits']}, "
                    f"misses={self._stats['misses']})")
        except Exception:
            return f"Cache(host={self.host}:{self.port}, db={self.db})"


# Singleton instance (optional)
_global_cache: Optional[Cache] = None


def get_global_cache(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    default_ttl: Optional[int] = None,
    key_prefix: str = "embedding"
) -> Cache:
    """
    Get or create the global cache instance (singleton pattern).

    Args:
        host: Redis server host (only used on first call)
        port: Redis server port (only used on first call)
        db: Redis database number (only used on first call)
        default_ttl: Default TTL in seconds (only used on first call)
        key_prefix: Key prefix (only used on first call)

    Returns:
        Global Cache instance
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = Cache(
            host=host,
            port=port,
            db=db,
            default_ttl=default_ttl,
            key_prefix=key_prefix
        )

    return _global_cache
