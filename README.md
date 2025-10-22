## Instructions to run the hybrid chat

Run redis docker:

```bash
docker run -d --name redis -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

### System Flowchart with Semantic Cache

```mermaid
flowchart TD
    A[User Query: Best hotels in Hanoi?] --> B[Call OpenAI Embeddings API]

    B --> C[text-embedding-3-small<br/>Returns 1536-dim vector]

    C --> D{Check RedisVL Semantic Cache<br/>using the embedding vector}

    D -->|Cache HIT<br/>Cosine Distance < 0.1| E[Return Cached LLM Response]
    E --> F[Display Response to User]
    F --> END[End]

    D -->|Cache MISS<br/>No similar query| G[Query Pinecone Vector DB<br/>Using same embedding vector]

    G --> H[Pinecone returns Top K=5<br/>with Cosine Similarity Scores]

    H --> I{Filter: Scores >= 0.3?}

    I -->|All Below 0.3| J[Fallback Response:<br/>No specific information found]
    J --> K[Cache Fallback in RedisVL<br/>Store with same embedding vector]
    K --> F

    I -->|At Least One >= 0.3| L[Extract Node IDs from Matches]

    L --> M[Query Neo4j Graph DB]
    M --> N[Fetch Relationships<br/>Up to 10 neighbors per node]

    N --> O[Build Context Prompt:<br/>System + Pinecone Matches + Graph Facts]

    O --> P[Send to OpenAI Chat API]
    P --> Q[gpt-4o-mini generates response]
    Q --> R[Cache Response in RedisVL<br/>Store with same embedding vector]

    R --> F

    style A fill:#e1f5ff,color:#000
    style B fill:#FFA500,color:#000
    style C fill:#FFA500,color:#000
    style E fill:#90EE90,color:#000
    style J fill:#ffe1e1,color:#000
    style F fill:#e1ffe1,color:#000
    style END fill:#f0f0f0,color:#000
    style D fill:#FFE4B5,color:#000
```

### Cache Strategy

**Semantic Cache (RedisVL)**:

- Caches complete LLM responses based on semantic similarity
- Uses OpenAI embeddings + cosine distance (threshold: 0.1)
- Saves entire pipeline on cache hit: Embedding + Pinecone + Neo4j + GPT

**Optimization**:

- Single embedding API call per request (reused for cache + Pinecone)
- Cache hit skips all downstream operations
- Persistent across restarts
