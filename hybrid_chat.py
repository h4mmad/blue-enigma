# hybrid_chat.py
import json
from typing import List
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config
from embedding_cache import EmbeddingCache
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

# Initialize embedding cache
embedding_cache = None
if config.CACHE_ENABLED:
    embedding_cache = EmbeddingCache(
        max_size=config.CACHE_MAX_SIZE,
        model_name=EMBED_MODEL
    )
    print(f"✓ Embedding cache enabled (max size: {config.CACHE_MAX_SIZE})")

    # Register cleanup handler to print stats on exit
    def print_cache_stats_on_exit():
        if embedding_cache and config.CACHE_STATS_LOGGING:
            embedding_cache.print_stats()

    atexit.register(print_cache_stats_on_exit)
else:
    print("✗ Embedding cache disabled")

# -----------------------------
# Helper functions
# -----------------------------
def embed_text(text: str) -> List[float]:
    """
    Get embedding for a text string.
    Uses cache if enabled to avoid redundant API calls.
    """
    # Try to get from cache first
    if embedding_cache:
        cached_embedding = embedding_cache.get(text)
        if cached_embedding is not None:
            print("✓ Found in cache - using cached embedding")
            return cached_embedding

    # Cache miss or cache disabled - call OpenAI API
    print("⟳ Cache miss - calling OpenAI API for embedding...")
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    embedding = resp.data[0].embedding

    # Store in cache for future use
    if embedding_cache:
        embedding_cache.set(text, embedding)
        print("✓ Stored embedding in cache")

    return embedding

def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding."""
    vec = embed_text(query_text)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    print("DEBUG: Pinecone top 5 results:")
    print(len(res["matches"]))
    return res["matches"]

def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    """Fetch neighboring nodes from Neo4j."""
    facts = []
    with driver.session() as session:
        for nid in node_ids:
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.type AS type, m.description AS description "
                "ORDER BY m.id "  # Added ORDER BY for deterministic results
                "LIMIT 10"
            )
            
            recs = session.run(q, nid=nid)
            for r in recs:
                facts.append({
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r["description"] or "")[:400],
                    "labels": r["labels"]
                })
    print("DEBUG: Graph facts:")
    print(len(facts))
    return facts

def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a chat prompt combining vector DB matches and graph facts."""
    system = (
        "You are a helpful travel assistant that is consistent with your answers. Use the provided semantic search results "
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
    global embedding_cache  # Declare global to access the module-level variable

    print("="*60)
    print("HYBRID TRAVEL ASSISTANT")
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
            if embedding_cache:
                embedding_cache.print_stats()
            else:
                print("Cache is disabled.")
            continue

        if query.lower() == "/clear":
            if embedding_cache:
                embedding_cache.clear()
                print("✓ Cache cleared successfully.")
            else:
                print("Cache is disabled.")
            continue

        # Process normal query
        matches = pinecone_query(query, top_k=TOP_K)
        match_ids = [m["id"] for m in matches]
        graph_facts = fetch_graph_context(match_ids)
        prompt = build_prompt(query, matches, graph_facts)
        answer = call_chat(prompt)
        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")

if __name__ == "__main__":
    interactive_chat()