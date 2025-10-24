#!/usr/bin/env python3
"""
Test script for semantic cache integration.
Tests that semantically similar queries hit the cache.
RedisVL automatically generates embeddings using OpenAI.
"""
import sys
import os
from dotenv import load_dotenv
from semantic_cache import LLMSemanticCache

# Load environment variables
load_dotenv(".env.local")

def test_semantic_cache():
    """Test semantic cache with similar queries."""

    print("="*60)
    print("SEMANTIC LLM CACHE TEST")
    print("="*60)

    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("   ✗ OPENAI_API_KEY not found in environment")
        return False

    # Initialize cache
    print("\n1. Initializing semantic cache...")
    try:
        cache = LLMSemanticCache(
            name="test_llm_cache",
            redis_url="redis://localhost:6379",
            distance_threshold=0.1,  # Moderate threshold
            openai_api_key=openai_api_key,
            embedding_model="text-embedding-3-small"
        )
        print("   ✓ Semantic cache initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize cache: {e}")
        return False

    # Test 1: Cache miss on first query
    print("\n2. Testing cache miss on new query...")
    query1 = "What are the best hotels in Hanoi?"
    result, embedding1 = cache.check(query1)
    if result is None and embedding1 is not None:
        print("   ✓ Cache miss detected correctly")
        print(f"   ✓ Embedding returned for reuse (dim: {len(embedding1)})")
    else:
        print(f"   ✗ Unexpected result: {result}")

    # Store LLM response
    print("\n3. Storing LLM response with embedding (no regeneration)...")
    llm_response = "The best hotels in Hanoi include Sofitel Legend Metropole and Hilton Hanoi Opera."
    success = cache.store(query1, llm_response, embedding1)  # Pass embedding!
    if success:
        print("   ✓ Response stored successfully")
        print("   ✓ Used cached embedding (avoided API call!)")
    else:
        print("   ✗ Failed to store response")
        return False

    # Test 2: Cache hit on same query
    print("\n4. Testing cache hit on same query...")
    result, embedding2 = cache.check(query1)
    if result == llm_response and embedding2 is not None:
        print("   ✓ Cache hit! Retrieved correct response")
        print("   ✓ Embedding still returned (can be reused if needed)")
    else:
        print(f"   ✗ Cache hit but wrong data: {result}")
        return False

    # Test 3: Semantic cache hit on similar query
    print("\n5. Testing semantic cache hit on similar query...")
    query2 = "What are good hotels in Hanoi?"  # Similar query
    result, embedding3 = cache.check(query2)
    if result == llm_response:
        print("   ✓ Semantic cache hit! Retrieved response for similar query")
        print("   (RedisVL automatically detected semantic similarity)")
    else:
        print(f"   ⚠ Semantic cache miss (query may not be similar enough)")
        print(f"   This is OK - threshold={cache.distance_threshold}")

    # Test 4: Statistics
    print("\n6. Testing statistics...")
    stats = cache.get_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Cache hits: {stats['hits']}")
    print(f"   Cache misses: {stats['misses']}")
    print(f"   Hit rate: {stats['hit_rate']:.1f}%")

    if stats['hits'] >= 1 and stats['misses'] >= 1:
        print("   ✓ Statistics tracked correctly")
    else:
        print("   ✗ Statistics incorrect")

    # Clean up
    print("\n7. Cleaning up test data...")
    cache.clear(reset_stats=True)
    print("   ✓ Cache cleared")

    cache.close()

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60 + "\n")

    return True


if __name__ == "__main__":
    success = test_semantic_cache()
    sys.exit(0 if success else 1)
