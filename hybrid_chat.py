# hybrid_chat.py
import json
from typing import List
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config
from semantic_cache import LLMSemanticCache
import atexit

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5

INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(api_key=config.OPENAI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Connect to Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east1-gcp")
    )

index = pc.Index(INDEX_NAME)

# Connect to Neo4j
driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# Initialize semantic cache (RedisVL-based, similarity matching for LLM responses)
semantic_cache = None
if config.CACHE_ENABLED and config.SEMANTIC_CACHE_ENABLED:
    try:
        redis_url = f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}"
        semantic_cache = LLMSemanticCache(
            name="llm_response_cache",
            redis_url=redis_url,
            distance_threshold=config.SEMANTIC_CACHE_THRESHOLD,
            default_ttl=config.SEMANTIC_CACHE_TTL,
            openai_api_key=config.OPENAI_API_KEY,
            embedding_model=EMBED_MODEL
        )
        print(f"✓ Semantic cache enabled (threshold: {config.SEMANTIC_CACHE_THRESHOLD})")

        # Register cleanup handler to print stats on exit
        def print_cache_stats_on_exit():
            if semantic_cache and config.CACHE_STATS_LOGGING:
                semantic_cache.print_stats()

        atexit.register(print_cache_stats_on_exit)
    except Exception as e:
        print(f"✗ Failed to initialize semantic cache: {e}")
        print("  Please ensure Redis is running: docker start redis")
        semantic_cache = None
else:
    print("✗ Semantic cache disabled")

# -----------------------------
# Helper functions
# -----------------------------
def embed_text(text: str) -> List[float]:
    """
    Get embedding for a text string.
    Always calls OpenAI API (semantic cache handles full response caching).
    """
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    embedding = resp.data[0].embedding
    return embedding

def pinecone_query(query_text: str, top_k=TOP_K, min_score=0.5):  # Threshold set to 0.5 for better semantic matching
    """
    Query Pinecone index using embedding.
    Filters results by minimum similarity score threshold.

    Args:
        query_text: The search query
        top_k: Number of results to retrieve from Pinecone
        min_score: Minimum similarity score threshold (default: 0.5)

    Returns:
        List of matches that meet the threshold, or empty list if none qualify
    """
    vec = embed_text(query_text)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )

    # Filter by threshold
    all_matches = res["matches"]
    strong_matches = [m for m in all_matches if m.get("score", 0) >= min_score]

    print(f"DEBUG: Pinecone returned {len(all_matches)} results, {len(strong_matches)} above threshold {min_score}")
    if strong_matches:
        print(f"  Top score: {strong_matches[0].get('score', 0):.3f}")
    elif all_matches:
        print(f"  Highest score (below threshold): {all_matches[0].get('score', 0):.3f}")

    return strong_matches

def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    """Fetch neighboring nodes from Neo4j in ONE batch query."""
    if not node_ids:
        return []

    facts = []
    with driver.session() as session:
        # Single batch query - queries all nodes at once instead of looping
        q = """
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE n.id IN $node_ids
        RETURN n.id AS source, type(r) AS rel,
               labels(m) AS labels, m.id AS id,
               m.name AS name, m.type AS type,
               m.description AS description
        ORDER BY n.id, m.id
        LIMIT 50
        """

        recs = session.run(q, node_ids=node_ids)
        for r in recs:
            facts.append({
                "source": r["source"],
                "rel": r["rel"],
                "target_id": r["id"],
                "target_name": r["name"],
                "target_desc": (r["description"] or "")[:400],
                "labels": r["labels"]
            })

    print(f"DEBUG: Graph facts: {len(facts)}")
    return facts

def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a chat prompt combining vector DB matches and graph facts."""
    system = (
        "You are a helpful travel assistant specializing in Vietnam travel. Use the provided semantic search results "
        "and graph facts to answer the user's query briefly and concisely. "
        "Cite node ids when referencing specific places or attractions."
    )

    vec_context = []
    for m in pinecone_matches:
        meta = m["metadata"]
        score = m.get("score", None)
        snippet = f"- id: {m['id']}, name: {meta.get('name','')}, type: {meta.get('type','')}, score: {score}"
        if meta.get("city"):
            snippet += f", city: {meta.get('city')}"
        vec_context.append(snippet)

    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts
    ]

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"User query: {user_query}\n\n"
         "Top semantic matches (from vector DB):\n" + "\n".join(vec_context[:10]) + "\n\n"
         "Graph facts (neighboring relations):\n" + "\n".join(graph_context[:20]) + "\n\n"
         "Based on the above, answer the user's question. If helpful, suggest 2–3 concrete itinerary steps or tips and mention node ids for references."}
    ]
    print(prompt) # for debugging
    return prompt

def call_chat(prompt_messages):
    """Call OpenAI ChatCompletion."""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=prompt_messages,
        max_tokens=600,
        temperature=0.0,  # Changed temperature to 0 for more deterministic response
        seed=42  # Added seed for more determinism
    )
    return resp.choices[0].message.content

# -----------------------------
# Interactive chat
# -----------------------------
def interactive_chat():
    global semantic_cache  # Declare global to access the module-level variable

    print("="*60)
    print("HYBRID TRAVEL ASSISTANT (with Semantic Cache)")
    print("="*60)
    print("Commands:")
    print("  - Type your travel question to get an answer")
    print("  - Type '/stats' to view cache statistics")
    print("  - Type '/clear' to clear the cache")
    print("  - Type 'exit' or 'quit' to quit")
    print("="*60)

    while True:
        query = input("\nEnter your travel question: ").strip()
        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            break

        # Special commands
        if query.lower() == "/stats":
            if semantic_cache:
                semantic_cache.print_stats()
            else:
                print("Cache is disabled.")
            continue

        if query.lower() == "/clear":
            if semantic_cache:
                semantic_cache.clear()
                print("✓ Semantic cache cleared successfully.")
            else:
                print("Cache is disabled.")
            continue

        # Step 1: Check semantic cache for similar query
        # This generates the embedding ONCE and returns it for reuse
        cached_response = None
        query_embedding = None

        if semantic_cache:
            print("⟳ Checking semantic cache and generating embedding...")
            cached_response, query_embedding = semantic_cache.check(query)

        if cached_response:
            # Cache hit! Return cached response immediately
            # Skipped: Pinecone query, Neo4j query, GPT call
            # (embedding was already generated for cache check)
            print("\n=== Assistant Answer (from cache) ===\n")
            print(cached_response)
            print("\n=== End ===\n")
            continue

        # Step 2: Cache miss - run full pipeline
        print("⟳ Cache miss - running full pipeline...")

        # Generate embedding if not already done by semantic cache
        if query_embedding is None:
            print("⟳ Generating embedding...")
            query_embedding = embed_text(query)

        # Query Pinecone using the embedding we already have
        vec = query_embedding  # Reuse embedding from cache check!
        res = index.query(
            vector=vec,
            top_k=TOP_K,
            include_metadata=True,
            include_values=False
        )

        # Filter by threshold
        all_matches = res["matches"]
        matches = [m for m in all_matches if m.get("score", 0) >= 0.5]

        print(f"DEBUG: Pinecone returned {len(all_matches)} results, {len(matches)} above threshold 0.5")
        if matches:
            print(f"  Top score: {matches[0].get('score', 0):.3f}")
        elif all_matches:
            print(f"  Highest score (below threshold): {all_matches[0].get('score', 0):.3f}")

        # Check if we have strong matches above threshold
        if not matches:
            fallback_response = (
                "I don't have specific information about that in my Vietnam travel database.\n\n"
                "You could try:\n"
                "  - Rephrasing your question\n"
                "  - Asking about Vietnam destinations, cities, or attractions\n"
                "  - Asking about Vietnamese food, culture, or activities"
            )
            print("\n=== Assistant Response ===")
            print(fallback_response)
            print("\n=== End ===\n")

            # Cache the fallback response too
            if semantic_cache:
                semantic_cache.store(query, fallback_response, query_embedding)
            continue

        # Query Neo4j for graph context
        match_ids = [m["id"] for m in matches]
        graph_facts = fetch_graph_context(match_ids)

        # Build prompt and call LLM
        prompt = build_prompt(query, matches, graph_facts)
        answer = call_chat(prompt)

        # Display answer
        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")

        # Step 3: Store response in semantic cache
        if semantic_cache:
            metadata = {
                'num_matches': len(matches),
                'num_graph_facts': len(graph_facts),
                'model': CHAT_MODEL
            }
            # Pass the embedding we already have to avoid regeneration!
            semantic_cache.store(query, answer, query_embedding, metadata=metadata)

if __name__ == "__main__":
    interactive_chat()