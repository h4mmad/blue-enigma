"""
Semantic Cache Module using RedisVL
Provides semantic similarity-based caching for LLM responses to save entire query pipeline.
"""
import logging
from typing import List, Optional, Dict, Any
from redisvl.extensions.cache.llm import SemanticCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMSemanticCache:
    """
    Semantic cache for LLM responses using RedisVL.

    Caches complete LLM responses based on semantic similarity of user queries.
    When a similar query is found, returns cached response without running the full pipeline:
    - Skips embedding API call
    - Skips Pinecone query
    - Skips Neo4j query
    - Skips GPT-4 call
    """

    def __init__(
        self,
        name: str = "llm_response_cache",
        redis_url: str = "redis://localhost:6379",
        distance_threshold: float = 0.1,
        default_ttl: Optional[int] = None,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the semantic cache for LLM responses.

        Args:
            name: Cache index name
            redis_url: Redis connection URL
            distance_threshold: Semantic similarity threshold (0.0-1.0)
                - Lower = stricter matching (e.g., 0.05 = very similar queries only)
                - Higher = more lenient (e.g., 0.2 = broader semantic matches)
                Default 0.1 is a good balance
            default_ttl: Default time-to-live in seconds (None = no expiration)
            openai_api_key: OpenAI API key for automatic embedding generation
            embedding_model: OpenAI embedding model to use (default: text-embedding-3-small)
        """
        self.name = name
        self.distance_threshold = distance_threshold
        self.default_ttl = default_ttl
        self.embedding_model = embedding_model

        # Statistics tracking (in-memory, per session)
        self._stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0,
            'pipeline_calls_saved': 0,
            'stores': 0
        }

        # Initialize RedisVL SemanticCache with OpenAI vectorizer
        try:
            from redisvl.utils.vectorize import OpenAITextVectorizer

            # Create OpenAI vectorizer - RedisVL will auto-generate embeddings
            vectorizer = OpenAITextVectorizer(
                model=embedding_model,
                api_config={"api_key": openai_api_key}
            )

            self._cache = SemanticCache(
                name=name,
                distance_threshold=distance_threshold,
                ttl=default_ttl,
                vectorizer=vectorizer,
                redis_url=redis_url,
                overwrite=False
            )
            logger.info(f"Semantic LLM cache initialized: {redis_url}, "
                       f"model={embedding_model}, threshold={distance_threshold}, ttl={default_ttl}")
        except Exception as e:
            logger.error(f"Failed to initialize semantic cache: {e}")
            raise

    def check(
        self,
        query: str
    ) -> tuple[Optional[str], Optional[List[float]]]:
        """
        Check if a semantically similar query exists in cache.
        RedisVL will automatically generate embeddings using OpenAI.

        Args:
            query: User query text

        Returns:
            Tuple of (cached_response, query_embedding):
            - If cache hit: (response, embedding)
            - If cache miss: (None, embedding) - embedding can be reused!
        """
        self._stats['total_requests'] += 1

        try:
            # Generate embedding for the query
            # We need to do this explicitly so we can return it
            embedding = self._cache._vectorizer.embed(query)

            # Search for semantically similar queries using the embedding
            results = self._cache.check(
                prompt=query,
                vector=embedding,  # Pass pre-computed embedding
                num_results=1,
                return_fields=["response", "prompt", "vector_distance"]
            )

            if results and len(results) > 0:
                # Cache hit
                self._stats['hits'] += 1
                self._stats['pipeline_calls_saved'] += 1

                result = results[0]
                distance = result.get('vector_distance', 'N/A')
                cached_query = result.get('prompt', '')[:50]

                logger.info(f"✓ SEMANTIC CACHE HIT!")
                logger.info(f"  Query: '{query[:60]}...'")
                logger.info(f"  Matched: '{cached_query}...' (distance: {distance})")

                # Return cached response AND the embedding (for potential reuse)
                return result.get('response'), embedding
            else:
                # Cache miss - return None but also the embedding so it can be reused!
                self._stats['misses'] += 1
                logger.debug(f"Cache MISS for query: '{query[:60]}...'")
                return None, embedding

        except Exception as e:
            logger.error(f"Error checking semantic cache: {e}")
            self._stats['misses'] += 1
            return None, None

    def store(
        self,
        query: str,
        llm_response: str,
        query_embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store LLM response in semantic cache.

        Args:
            query: User query text
            llm_response: The LLM's response to cache
            query_embedding: Pre-computed embedding vector (avoids regeneration!)
                If None, RedisVL will generate it (costs extra API call)
            metadata: Optional metadata (e.g., model, tokens, context)
            ttl: Time-to-live in seconds (None = use default_ttl)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Store in semantic cache
            # If query_embedding is provided, RedisVL won't regenerate it!
            self._cache.store(
                prompt=query,
                response=llm_response,
                vector=query_embedding,  # Pass embedding to avoid regeneration
                metadata=metadata,
                ttl=ttl if ttl is not None else self.default_ttl
            )

            self._stats['stores'] += 1

            if query_embedding is not None:
                logger.debug(f"Cached LLM response (reused embedding) for query: '{query[:60]}...'")
            else:
                logger.debug(f"Cached LLM response (generated new embedding) for query: '{query[:60]}...'")

            return True

        except Exception as e:
            logger.error(f"Error storing in semantic cache: {e}")
            return False

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

        # Estimate cost savings
        # Full pipeline cost per query (rough estimate):
        # - Embedding: $0.00002 per 1K tokens * ~20 tokens = $0.0000004
        # - GPT-4o-mini: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
        # - Average query: ~500 input tokens, ~300 output tokens
        # - Total: ~$0.00025 per query
        avg_cost_per_query = 0.00025
        estimated_savings = stats['pipeline_calls_saved'] * avg_cost_per_query
        stats['estimated_cost_savings_usd'] = estimated_savings

        return stats

    def print_stats(self) -> None:
        """
        Print formatted cache statistics to console.
        """
        stats = self.get_stats()

        print("\n" + "="*60)
        print("SEMANTIC LLM CACHE STATISTICS")
        print("="*60)
        print(f"Cache Name:            {self.name}")
        print(f"Distance Threshold:    {self.distance_threshold}")
        print(f"Total Requests:        {stats['total_requests']}")
        print(f"Cache Hits:            {stats['hits']} ({stats['hit_rate']:.1f}%)")
        print(f"Cache Misses:          {stats['misses']} ({stats['miss_rate']:.1f}%)")
        print(f"Pipeline Calls Saved:  {stats['pipeline_calls_saved']}")
        print(f"Responses Stored:      {stats['stores']}")
        print(f"Est. Cost Savings:     ${stats['estimated_cost_savings_usd']:.6f}")
        print("="*60 + "\n")

    def clear(self, reset_stats: bool = True) -> bool:
        """
        Clear all cache entries.

        Args:
            reset_stats: Whether to reset statistics

        Returns:
            True if successful
        """
        try:
            self._cache.clear()

            # Reset stats if requested
            if reset_stats:
                self._stats = {
                    'hits': 0,
                    'misses': 0,
                    'total_requests': 0,
                    'pipeline_calls_saved': 0,
                    'stores': 0
                }

            logger.info("Semantic cache cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Error clearing semantic cache: {e}")
            return False

    def set_threshold(self, threshold: float) -> None:
        """
        Update the semantic similarity threshold.

        Args:
            threshold: New distance threshold (0.0-1.0)
        """
        self._cache.set_threshold(threshold)
        self.distance_threshold = threshold
        logger.info(f"Semantic cache threshold updated to {threshold}")

    def close(self) -> None:
        """
        Close the Redis connection.
        """
        try:
            self._cache.disconnect()
            logger.info("Semantic cache connection closed")
        except Exception as e:
            logger.error(f"Error closing semantic cache connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation of cache."""
        return (f"LLMSemanticCache(name={self.name}, threshold={self.distance_threshold}, "
                f"hits={self._stats['hits']}, misses={self._stats['misses']})")


# Singleton instance (optional)
_global_semantic_cache: Optional[LLMSemanticCache] = None


def get_global_semantic_cache(
    name: str = "llm_response_cache",
    redis_url: str = "redis://localhost:6379",
    distance_threshold: float = 0.1,
    default_ttl: Optional[int] = None,
    openai_api_key: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small"
) -> LLMSemanticCache:
    """
    Get or create the global semantic cache instance (singleton pattern).

    Args:
        name: Cache index name (only used on first call)
        redis_url: Redis URL (only used on first call)
        distance_threshold: Semantic threshold (only used on first call)
        default_ttl: Default TTL in seconds (only used on first call)
        openai_api_key: OpenAI API key (only used on first call)
        embedding_model: Embedding model name (only used on first call)

    Returns:
        Global LLMSemanticCache instance
    """
    global _global_semantic_cache

    if _global_semantic_cache is None:
        _global_semantic_cache = LLMSemanticCache(
            name=name,
            redis_url=redis_url,
            distance_threshold=distance_threshold,
            default_ttl=default_ttl,
            openai_api_key=openai_api_key,
            embedding_model=embedding_model
        )

    return _global_semantic_cache
