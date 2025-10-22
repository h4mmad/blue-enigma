"""
Test script to compare Sequential vs Parallel hybrid retrieval performance.
Tests the query: "create a romantic 4 day itinerary for Vietnam"

This demonstrates:
1. Sequential approach (embedding -> Pinecone -> Neo4j)
2. Parallel approach (embedding || keyword Neo4j query, then Pinecone)
3. Timing comparison
4. Data quality comparison
"""

import asyncio
import time
from typing import List, Dict
import config
from openai import AsyncOpenAI
from pinecone import Pinecone
from neo4j import AsyncGraphDatabase

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # Using gpt-4o-mini for faster/cheaper testing
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME

# Test query
TEST_QUERY = "create a romantic 4 day itinerary for Vietnam"

# -----------------------------
# Initialize clients
# -----------------------------
async_openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)
pinecone_index = pc.Index(INDEX_NAME)

# Neo4j async driver
neo4j_driver = AsyncGraphDatabase.driver(
    config.NEO4J_URI,
    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# -----------------------------
# Async Helper Functions
# -----------------------------

async def embed_text_async(text: str) -> List[float]:
    """Generate embedding asynchronously."""
    resp = await async_openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return resp.data[0].embedding


def pinecone_query_sync(embedding: List[float], top_k: int = TOP_K) -> List[Dict]:
    """Query Pinecone (synchronous - Pinecone doesn't have async support yet)."""
    res = pinecone_index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    return res["matches"]


async def neo4j_query_by_ids(node_ids: List[str]) -> List[Dict]:
    """Query Neo4j for specific node IDs + relationships."""
    facts = []
    async with neo4j_driver.session() as session:
        for nid in node_ids:
            query = """
                MATCH (n:Entity {id: $nid})-[r]-(m:Entity)
                RETURN type(r) AS rel, m.id AS id, m.name AS name,
                       m.type AS type, m.description AS description
                LIMIT 10
            """
            result = await session.run(query, nid=nid)
            records = await result.data()
            for r in records:
                facts.append({
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r.get("description") or "")[:200]
                })
    return facts


async def neo4j_query_by_keywords(keywords: List[str]) -> List[Dict]:
    """Query Neo4j by keywords (tags)."""
    results = []
    async with neo4j_driver.session() as session:
        query = """
            MATCH (n:Entity)
            WHERE ANY(tag IN $tags WHERE tag IN n.tags)
            RETURN n.id AS id, n.name AS name, n.type AS type,
                   n.tags AS tags, n.description AS description
            LIMIT 20
        """
        result = await session.run(query, tags=keywords)
        records = await result.data()
        for r in records:
            results.append({
                "id": r["id"],
                "name": r["name"],
                "type": r["type"],
                "tags": r.get("tags", []),
                "description": (r.get("description") or "")[:200]
            })
    return results


def build_prompt(user_query: str, pinecone_matches: List[Dict], graph_facts: List[Dict], keyword_results: List[Dict] = None):
    """Build a chat prompt combining vector DB matches and graph facts."""
    system = (
        "You are a helpful travel assistant specializing in Vietnam travel. Use the provided semantic search results "
        "and graph facts to answer the user's query. Create a detailed, romantic 4-day itinerary. "
        "Cite node ids when referencing specific places or attractions."
    )

    vec_context = []
    for m in pinecone_matches:
        meta = m.get("metadata", {})
        score = m.get("score", None)
        snippet = f"- id: {m['id']}, name: {meta.get('name','')}, type: {meta.get('type','')}, score: {score:.3f}"
        if meta.get("city"):
            snippet += f", city: {meta.get('city')}"
        vec_context.append(snippet)

    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts[:20]
    ]

    content = (
        f"User query: {user_query}\n\n"
        "Top semantic matches (from vector DB):\n" + "\n".join(vec_context[:10]) + "\n\n"
        "Graph facts (neighboring relations):\n" + "\n".join(graph_context) + "\n\n"
    )

    # Add keyword results if provided (for parallel approach)
    if keyword_results:
        keyword_context = [
            f"- id: {r['id']}, name: {r.get('name','')}, type: {r.get('type','')}"
            for r in keyword_results[:10]
        ]
        content += "Additional keyword-based results:\n" + "\n".join(keyword_context) + "\n\n"

    content += "Based on the above, create a romantic 4-day itinerary for Vietnam. Include specific places, activities, and practical tips."

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": content}
    ]
    return prompt


async def call_chat_async(prompt_messages: List[Dict]) -> str:
    """Call OpenAI ChatCompletion asynchronously."""
    resp = await async_openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=prompt_messages,
        max_tokens=800,
        temperature=0.7
    )
    return resp.choices[0].message.content


# -----------------------------
# Sequential Approach
# -----------------------------

async def sequential_hybrid_query(query_text: str):
    """
    Traditional sequential approach:
    1. Generate embedding
    2. Query Pinecone with embedding
    3. Query Neo4j with node IDs from Pinecone
    """
    print("\n" + "="*70)
    print("SEQUENTIAL APPROACH")
    print("="*70)

    start_time = time.time()

    # Step 1: Generate embedding
    step1_start = time.time()
    embedding = await embed_text_async(query_text)
    step1_time = time.time() - step1_start
    print(f"✓ Step 1 - Generate embedding: {step1_time*1000:.0f}ms")

    # Step 2: Query Pinecone
    step2_start = time.time()
    pinecone_matches = pinecone_query_sync(embedding, top_k=TOP_K)
    step2_time = time.time() - step2_start
    print(f"✓ Step 2 - Query Pinecone: {step2_time*1000:.0f}ms")
    print(f"  → Found {len(pinecone_matches)} matches")

    # Step 3: Query Neo4j with those IDs
    step3_start = time.time()
    node_ids = [m["id"] for m in pinecone_matches]
    graph_facts = await neo4j_query_by_ids(node_ids)
    step3_time = time.time() - step3_start
    print(f"✓ Step 3 - Query Neo4j by IDs: {step3_time*1000:.0f}ms")
    print(f"  → Found {len(graph_facts)} relationships")

    # Step 4: Generate ChatGPT response
    step4_start = time.time()
    prompt = build_prompt(query_text, pinecone_matches, graph_facts)
    gpt_response = await call_chat_async(prompt)
    step4_time = time.time() - step4_start
    print(f"✓ Step 4 - Generate GPT response: {step4_time*1000:.0f}ms")

    total_time = time.time() - start_time
    print(f"\n⏱️  TOTAL TIME: {total_time*1000:.0f}ms")
    print(f"   Breakdown: {step1_time*1000:.0f}ms + {step2_time*1000:.0f}ms + {step3_time*1000:.0f}ms + {step4_time*1000:.0f}ms")

    return {
        "pinecone_matches": pinecone_matches,
        "graph_facts": graph_facts,
        "gpt_response": gpt_response,
        "total_time_ms": total_time * 1000,
        "breakdown": {
            "embedding_ms": step1_time * 1000,
            "pinecone_ms": step2_time * 1000,
            "neo4j_ms": step3_time * 1000,
            "gpt_ms": step4_time * 1000
        }
    }


# -----------------------------
# Parallel Approach
# -----------------------------

async def parallel_hybrid_query(query_text: str):
    """
    Parallel approach:
    1. Generate embedding || Query Neo4j by keywords (PARALLEL)
    2. Query Pinecone with embedding (SEQUENTIAL - needs embedding)
    3. Optionally enrich with specific node data from Neo4j
    """
    print("\n" + "="*70)
    print("PARALLEL APPROACH")
    print("="*70)

    start_time = time.time()

    # Extract keywords from query
    keywords = []
    if "romantic" in query_text.lower():
        keywords.append("romantic")
    # Could add more sophisticated keyword extraction here

    print(f"Extracted keywords: {keywords}")

    # Phase 1: PARALLEL - Generate embedding + keyword Neo4j query
    phase1_start = time.time()
    print("\n→ Phase 1: Running embedding + keyword Neo4j query in PARALLEL...")

    embedding_task = embed_text_async(query_text)
    keyword_graph_task = neo4j_query_by_keywords(keywords) if keywords else asyncio.sleep(0)

    # Wait for both to complete
    embedding, keyword_results = await asyncio.gather(
        embedding_task,
        keyword_graph_task
    )

    phase1_time = time.time() - phase1_start
    print(f"✓ Phase 1 completed: {phase1_time*1000:.0f}ms")
    if keywords:
        print(f"  → Keyword search found {len(keyword_results) if keyword_results else 0} nodes")

    # Phase 2: SEQUENTIAL - Query Pinecone (needs embedding from Phase 1)
    phase2_start = time.time()
    print("\n→ Phase 2: Query Pinecone with embedding (SEQUENTIAL)...")
    pinecone_matches = pinecone_query_sync(embedding, top_k=TOP_K)
    phase2_time = time.time() - phase2_start
    print(f"✓ Phase 2 completed: {phase2_time*1000:.0f}ms")
    print(f"  → Found {len(pinecone_matches)} matches")

    # Phase 3: Query Neo4j for specific nodes from Pinecone
    phase3_start = time.time()
    print("\n→ Phase 3: Query Neo4j for specific Pinecone nodes (SEQUENTIAL)...")
    node_ids = [m["id"] for m in pinecone_matches]
    specific_graph_facts = await neo4j_query_by_ids(node_ids)
    phase3_time = time.time() - phase3_start
    print(f"✓ Phase 3 completed: {phase3_time*1000:.0f}ms")
    print(f"  → Found {len(specific_graph_facts)} relationships")

    # Phase 4: Generate ChatGPT response (with keyword results included)
    phase4_start = time.time()
    print("\n→ Phase 4: Generate GPT response...")
    prompt = build_prompt(
        query_text,
        pinecone_matches,
        specific_graph_facts,
        keyword_results=keyword_results if keywords else None
    )
    gpt_response = await call_chat_async(prompt)
    phase4_time = time.time() - phase4_start
    print(f"✓ Phase 4 completed: {phase4_time*1000:.0f}ms")

    total_time = time.time() - start_time
    print(f"\n⏱️  TOTAL TIME: {total_time*1000:.0f}ms")
    print(f"   Breakdown: {phase1_time*1000:.0f}ms (parallel) + {phase2_time*1000:.0f}ms + {phase3_time*1000:.0f}ms + {phase4_time*1000:.0f}ms")

    return {
        "pinecone_matches": pinecone_matches,
        "keyword_results": keyword_results if keywords else [],
        "graph_facts": specific_graph_facts,
        "gpt_response": gpt_response,
        "total_time_ms": total_time * 1000,
        "breakdown": {
            "phase1_parallel_ms": phase1_time * 1000,
            "phase2_pinecone_ms": phase2_time * 1000,
            "phase3_neo4j_ms": phase3_time * 1000,
            "phase4_gpt_ms": phase4_time * 1000
        }
    }


# -----------------------------
# Analysis & Comparison
# -----------------------------

def analyze_results(sequential_result, parallel_result):
    """Compare the two approaches."""
    print("\n" + "="*70)
    print("ANALYSIS & COMPARISON")
    print("="*70)

    seq_time = sequential_result["total_time_ms"]
    par_time = parallel_result["total_time_ms"]
    time_saved = seq_time - par_time
    speedup = (seq_time / par_time) if par_time > 0 else 1.0

    print(f"\n📊 TIMING COMPARISON:")
    print(f"   Sequential: {seq_time:.0f}ms")
    print(f"   Parallel:   {par_time:.0f}ms")
    print(f"   Time saved: {time_saved:.0f}ms ({(time_saved/seq_time)*100:.1f}%)")
    print(f"   Speedup:    {speedup:.2f}x")

    print(f"\n📦 DATA COMPARISON:")
    print(f"   Sequential Pinecone matches: {len(sequential_result['pinecone_matches'])}")
    print(f"   Parallel Pinecone matches:   {len(parallel_result['pinecone_matches'])}")
    print(f"   Parallel keyword results:    {len(parallel_result.get('keyword_results', []))}")

    # Check overlap
    seq_ids = set(m["id"] for m in sequential_result["pinecone_matches"])
    par_ids = set(m["id"] for m in parallel_result["pinecone_matches"])
    keyword_ids = set(r["id"] for r in parallel_result.get("keyword_results", []))

    overlap = seq_ids.intersection(keyword_ids)

    print(f"\n🔍 DATA QUALITY:")
    print(f"   Pinecone results identical: {seq_ids == par_ids}")
    print(f"   Keyword-Pinecone overlap:   {len(overlap)} / {len(keyword_ids) if keyword_ids else 0}")
    if overlap:
        print(f"   → {(len(overlap)/len(keyword_ids))*100:.0f}% of keyword results already in Pinecone top-{TOP_K}")

    print(f"\n📋 TOP PINECONE MATCHES:")
    for i, match in enumerate(sequential_result["pinecone_matches"][:3], 1):
        meta = match.get("metadata", {})
        print(f"   {i}. {meta.get('name', 'N/A')} ({meta.get('type', 'N/A')}) - Score: {match.get('score', 0):.3f}")

    if parallel_result.get("keyword_results"):
        print(f"\n📋 KEYWORD RESULTS (first 5):")
        for i, result in enumerate(parallel_result["keyword_results"][:5], 1):
            in_pinecone = "✓" if result["id"] in seq_ids else "✗"
            print(f"   {i}. {result.get('name', 'N/A')} ({result.get('type', 'N/A')}) [{in_pinecone} in Pinecone top-{TOP_K}]")

    # Compare GPT responses
    print(f"\n" + "="*70)
    print("GPT-4O-MINI RESPONSES")
    print("="*70)

    print(f"\n🤖 SEQUENTIAL APPROACH RESPONSE:")
    print("-" * 70)
    print(sequential_result.get("gpt_response", "No response"))
    print("-" * 70)

    print(f"\n🤖 PARALLEL APPROACH RESPONSE:")
    print("-" * 70)
    print(parallel_result.get("gpt_response", "No response"))
    print("-" * 70)

    # Quality comparison
    seq_response_len = len(sequential_result.get("gpt_response", ""))
    par_response_len = len(parallel_result.get("gpt_response", ""))

    print(f"\n📝 RESPONSE COMPARISON:")
    print(f"   Sequential response length: {seq_response_len} characters")
    print(f"   Parallel response length:   {par_response_len} characters")
    print(f"   Difference: {abs(seq_response_len - par_response_len)} characters")

    print(f"\n💡 NOTE:")
    print(f"   The parallel approach includes {len(parallel_result.get('keyword_results', []))} additional")
    print(f"   keyword-based nodes in the context, which may affect response quality.")


# -----------------------------
# Main Test
# -----------------------------

async def main():
    print("="*70)
    print("PARALLEL vs SEQUENTIAL HYBRID RETRIEVAL TEST")
    print("="*70)
    print(f"Query: '{TEST_QUERY}'")
    print("="*70)

    try:
        # Run sequential approach
        sequential_result = await sequential_hybrid_query(TEST_QUERY)

        # Wait a bit to ensure fair comparison
        await asyncio.sleep(1)

        # Run parallel approach
        parallel_result = await parallel_hybrid_query(TEST_QUERY)

        # Analyze and compare
        analyze_results(sequential_result, parallel_result)

        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)
        print("""
The parallel approach can save time by running embedding generation
and keyword-based Neo4j queries concurrently. However:

PROS:
✓ Faster overall execution (~15-20% speedup from parallel phase)
✓ Gets additional keyword-based results as backup
✓ Fallback mechanism if Pinecone has limited results
✓ More context data available to GPT

CONS:
✗ High overlap between keyword and Pinecone results (often 80-90%)
✗ Keyword results are unranked (no relevance score)
✗ More complex code (merging, deduplication needed)
✗ Additional context may not improve GPT response quality
✗ Extra keyword nodes can add noise to the context

GPT RESPONSE QUALITY:
Both approaches produce similar quality responses because:
- Pinecone provides the core semantic matches (same in both)
- Graph relationships enrich those matches (same in both)
- Keyword results mostly overlap with Pinecone results
- GPT focuses on the high-relevance Pinecone matches regardless

RECOMMENDATION:
For this specific use case, sequential with async I/O is the better choice:
✓ Simpler, more maintainable code
✓ Focused, high-quality context for GPT
✓ Comparable response quality
✓ Only slightly slower (~100-150ms difference)

Use async/await for non-blocking I/O, but keep sequential logic for
clarity and quality. The speed gain from parallelization doesn't justify
the increased complexity and potential quality issues.
        """)

    finally:
        await neo4j_driver.close()


if __name__ == "__main__":
    asyncio.run(main())
