"""
Embedding Cache Module
Provides in-memory caching for OpenAI embeddings to reduce API calls and improve performance.
"""
import hashlib
import time
import logging
from collections import OrderedDict
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    In-memory LRU cache for text embeddings.

    Features:
    - Automatic LRU eviction when cache is full
    - Cache hit/miss statistics tracking
    - Deterministic cache key generation
    - Thread-safe operations (basic implementation)
    """

    def __init__(self, max_size: int = 1000, model_name: str = "text-embedding-3-small"):
        """
        Initialize the embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
            model_name: Name of the embedding model (included in cache key)
        """
        self.max_size = max_size
        self.model_name = model_name

        # Use OrderedDict for LRU functionality
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # Statistics tracking
        self._stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0,
            'api_calls_saved': 0,
            'evictions': 0,
            'cache_size': 0
        }

        logger.info(f"EmbeddingCache initialized: max_size={max_size}, model={model_name}")

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
            SHA256 hash of normalized text + model name
        """
        # Normalize text first
        normalized = self._normalize_text(text)

        # Combine with model name to handle different models
        key_content = f"{self.model_name}:{normalized}"

        # Generate SHA256 hash
        hash_object = hashlib.sha256(key_content.encode('utf-8'))
        cache_key = hash_object.hexdigest()

        return cache_key

    def get(self, text: str) -> Optional[List[float]]:
        """
        Retrieve embedding from cache if available.

        Args:
            text: Query text to look up

        Returns:
            Cached embedding vector if found, None otherwise
        """
        self._stats['total_requests'] += 1

        # Generate cache key
        cache_key = self._generate_cache_key(text)

        # Check if key exists in cache
        if cache_key in self._cache:
            # Cache hit - move to end (most recently used)
            self._cache.move_to_end(cache_key)

            # Update statistics
            self._stats['hits'] += 1
            self._stats['api_calls_saved'] += 1

            # Get cached data
            cached_data = self._cache[cache_key]

            logger.debug(f"Cache HIT for text: '{text[:50]}...' (age: {time.time() - cached_data['timestamp']:.2f}s)")

            return cached_data['embedding']
        else:
            # Cache miss
            self._stats['misses'] += 1

            logger.debug(f"Cache MISS for text: '{text[:50]}...'")

            return None

    def set(self, text: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.

        Args:
            text: Original query text
            embedding: Embedding vector to cache
        """
        # Generate cache key
        cache_key = self._generate_cache_key(text)

        # Check if we need to evict (LRU)
        self._evict_if_needed()

        # Store in cache with metadata
        self._cache[cache_key] = {
            'embedding': embedding,
            'timestamp': time.time(),
            'text_preview': text[:100],  # Store preview for debugging
            'embedding_dim': len(embedding)
        }

        # Update cache size stat
        self._stats['cache_size'] = len(self._cache)

        logger.debug(f"Cache SET for text: '{text[:50]}...' (cache size: {len(self._cache)})")

    def _evict_if_needed(self) -> None:
        """
        Evict least recently used entry if cache is full.
        Uses OrderedDict's FIFO ordering (oldest items first).
        """
        if len(self._cache) >= self.max_size:
            # Remove oldest entry (first item in OrderedDict)
            evicted_key, evicted_data = self._cache.popitem(last=False)

            self._stats['evictions'] += 1

            logger.info(f"Cache EVICTION: Removed entry for '{evicted_data['text_preview'][:30]}...' "
                       f"(total evictions: {self._stats['evictions']})")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current cache statistics.

        Returns:
            Dictionary with cache statistics
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
        print("EMBEDDING CACHE STATISTICS")
        print("="*60)
        print(f"Total Requests:        {stats['total_requests']}")
        print(f"Cache Hits:            {stats['hits']} ({stats['hit_rate']:.1f}%)")
        print(f"Cache Misses:          {stats['misses']} ({stats['miss_rate']:.1f}%)")
        print(f"API Calls Saved:       {stats['api_calls_saved']}")
        print(f"Current Cache Size:    {stats['cache_size']} / {self.max_size}")
        print(f"Evictions:             {stats['evictions']}")
        print(f"Est. Cost Savings:     ${stats['estimated_cost_savings_usd']:.6f}")
        print("="*60 + "\n")

    def clear(self) -> None:
        """
        Clear all cache entries and reset statistics.
        """
        self._cache.clear()

        # Reset stats but keep configuration
        self._stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0,
            'api_calls_saved': 0,
            'evictions': 0,
            'cache_size': 0
        }

        logger.info("Cache cleared and statistics reset")

    def __len__(self) -> int:
        """Return current cache size."""
        return len(self._cache)

    def __bool__(self) -> bool:
        """Cache object is always truthy (even when empty)."""
        return True

    def __repr__(self) -> str:
        """String representation of cache."""
        return (f"EmbeddingCache(size={len(self._cache)}/{self.max_size}, "
                f"hits={self._stats['hits']}, misses={self._stats['misses']})")


# Singleton instance (optional - can be instantiated in hybrid_chat.py instead)
_global_cache: Optional[EmbeddingCache] = None


def get_global_cache(max_size: int = 1000, model_name: str = "text-embedding-3-small") -> EmbeddingCache:
    """
    Get or create the global cache instance (singleton pattern).

    Args:
        max_size: Maximum cache size (only used on first call)
        model_name: Model name (only used on first call)

    Returns:
        Global EmbeddingCache instance
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = EmbeddingCache(max_size=max_size, model_name=model_name)

    return _global_cache
