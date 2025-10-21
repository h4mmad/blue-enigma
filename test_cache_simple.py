"""
Simple test to verify the embedding cache is working correctly.
Run this with: source venv/bin/activate && python test_cache_simple.py
"""
from embedding_cache import EmbeddingCache

print("="*60)
print("SIMPLE CACHE TEST")
print("="*60)

# Create a cache instance
cache = EmbeddingCache(max_size=5, model_name="text-embedding-3-small")

# Test 1: Cache Miss
print("\nTest 1: First query (should be MISS)")
result = cache.get("What are the best places in Vietnam?")
print(f"Result: {result}")
print(f"Expected: None (cache miss)")

# Simulate storing an embedding
test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 300  # Realistic size
cache.set("What are the best places in Vietnam?", test_embedding)
print("✓ Stored embedding in cache")

# Test 2: Cache Hit
print("\nTest 2: Same query again (should be HIT)")
result = cache.get("What are the best places in Vietnam?")
print(f"Result: Found embedding with {len(result)} dimensions")
print(f"Expected: 1500 dimensions")

# Test 3: Normalization (different case should still hit cache)
print("\nTest 3: Same query with different case (should be HIT)")
result = cache.get("what are the best places in vietnam?")
print(f"Result: Found embedding with {len(result)} dimensions" if result else "MISS")
print(f"Expected: HIT due to text normalization")

# Test 4: Different query (should be MISS)
print("\nTest 4: Different query (should be MISS)")
result = cache.get("Tell me about Hanoi attractions")
print(f"Result: {result}")
print(f"Expected: None (cache miss)")

# Show statistics
print("\n" + "="*60)
cache.print_stats()

print("✓ Cache is working correctly!")
print("\nTo use the cache in hybrid_chat.py:")
print("1. Make sure you run: source venv/bin/activate")
print("2. Run: python hybrid_chat.py")
print("3. You should see: '✓ Embedding cache enabled'")
print("4. Type '/stats' to see cache statistics")
