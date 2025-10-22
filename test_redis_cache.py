#!/usr/bin/env python3
"""
Test script for Redis cache integration.
Tests basic cache operations and verifies it works correctly.
"""
import sys
from redis_cache import Cache

def test_redis_cache():
    """Test Redis cache basic operations."""

    print("="*60)
    print("REDIS CACHE TEST")
    print("="*60)

    # Initialize cache
    print("\n1. Initializing Redis cache...")
    try:
        cache = Cache(
            host="localhost",
            port=6379,
            db=0,
            key_prefix="test_embedding"
        )
        print("   ✓ Cache initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize cache: {e}")
        return False

    # Test connection
    print("\n2. Testing connection...")
    if cache.is_connected():
        print("   ✓ Connected to Redis")
    else:
        print("   ✗ Not connected to Redis")
        return False

    # Test cache miss
    print("\n3. Testing cache miss...")
    test_text = "What are the best hotels in Hanoi?"
    result = cache.get(test_text)
    if result is None:
        print("   ✓ Cache miss detected correctly")
    else:
        print(f"   ✗ Unexpected cache hit: {result}")

    # Test cache set
    print("\n4. Testing cache set...")
    test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dim vector
    success = cache.set(test_text, test_embedding)
    if success:
        print("   ✓ Embedding stored successfully")
    else:
        print("   ✗ Failed to store embedding")
        return False

    # Test cache hit
    print("\n5. Testing cache hit...")
    result = cache.get(test_text)
    if result == test_embedding:
        print("   ✓ Cache hit! Retrieved correct embedding")
    else:
        print(f"   ✗ Cache hit but wrong data")
        return False

    # Test normalization (case-insensitive)
    print("\n6. Testing text normalization...")
    result = cache.get("WHAT ARE THE BEST HOTELS IN HANOI?")
    if result == test_embedding:
        print("   ✓ Normalization works (case-insensitive)")
    else:
        print("   ✗ Normalization failed")
        return False

    # Test normalization (whitespace)
    result = cache.get("  what   are  the   best  hotels  in  hanoi?  ")
    if result == test_embedding:
        print("   ✓ Normalization works (whitespace handling)")
    else:
        print("   ✗ Normalization failed")
        return False

    # Test statistics
    print("\n7. Testing statistics...")
    stats = cache.get_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Cache hits: {stats['hits']}")
    print(f"   Cache misses: {stats['misses']}")
    print(f"   Hit rate: {stats['hit_rate']:.1f}%")

    if stats['hits'] >= 2 and stats['misses'] >= 1:
        print("   ✓ Statistics tracked correctly")
    else:
        print("   ✗ Statistics incorrect")

    # Test exists
    print("\n8. Testing exists check...")
    if cache.exists(test_text):
        print("   ✓ Exists check works")
    else:
        print("   ✗ Exists check failed")

    # Test delete
    print("\n9. Testing delete...")
    if cache.delete(test_text):
        print("   ✓ Entry deleted successfully")
    else:
        print("   ✗ Delete failed")

    # Verify deletion
    result = cache.get(test_text)
    if result is None:
        print("   ✓ Verified deletion successful")
    else:
        print("   ✗ Entry still exists after delete")

    # Clean up
    print("\n10. Cleaning up test data...")
    deleted = cache.clear(reset_stats=True)
    print(f"   ✓ Cleared {deleted} keys")

    cache.close()

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60 + "\n")

    return True


if __name__ == "__main__":
    success = test_redis_cache()
    sys.exit(0 if success else 1)
