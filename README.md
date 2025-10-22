## Instructions to run the hybrid chat

Run redis docker:
```bash
docker run -d --name redis -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

### System Flowchart with Semantic Cache

```mermaid
flowchart TD
    A[User Query: Best hotels in Hanoi?] --> B{Check RedisVL Semantic Cache}

    B -->|Generate Embedding| C[Call OpenAI Embeddings API]
    C --> D[text-embedding-3-small returns 1536-dim vector]

    D --> E{Search Cache by Vector Similarity<br/>Cosine Distance Threshold: 0.1}

    E -->|Cache HIT<br/>Similar query found| F[Return Cached LLM Response]
    F --> G[Display Response to User]
    G --> END[End]

    E -->|Cache MISS<br/>No similar query| H[Query Pinecone Vector DB<br/>Reuse embedding from step D]

    H --> I[Pinecone returns Top K=5<br/>with Cosine Similarity Scores]

    I --> J{Filter: Scores >= 0.3?}

    J -->|All Below 0.3| K[Fallback Response:<br/>No specific information found]
    K --> L[Cache Fallback Response<br/>with embedding]
    L --> G

    J -->|At Least One >= 0.3| M[Extract Node IDs from Matches]

    M --> N[Query Neo4j Graph DB]
    N --> O[Fetch Relationships<br/>Up to 10 neighbors per node]

    O --> P[Build Context Prompt:<br/>System + Pinecone Matches + Graph Facts]

    P --> Q[Send to OpenAI Chat API]
    Q --> R[gpt-4o-mini generates response]
    R --> S[Cache Response in RedisVL<br/>Reuse embedding from step D]

    S --> G

    style A fill:#e1f5ff,color:#000
    style F fill:#90EE90,color:#000
    style K fill:#ffe1e1,color:#000
    style G fill:#e1ffe1,color:#000
    style END fill:#f0f0f0
    style B fill:#FFE4B5,color:#000
    style E fill:#FFE4B5,color:#000
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
