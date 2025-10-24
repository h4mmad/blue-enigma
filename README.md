## Architecture overview

<img width="937" height="736" alt="image" src="https://github.com/user-attachments/assets/c174385c-1e5d-491b-92ff-a1bcb29361a5" />

## System Flowchart with Semantic Cache

```mermaid
flowchart TD
    A[User Query: reate a romantic 4 day itinerary for
Vietnam] --> B[Call OpenAI Embeddings API]

    B --> C[text-embedding-3-small<br/>Returns 1536-dim vector]

    C --> D{Check RedisVL Semantic Cache<br/>using the embedding vector}

    D -->|Cache HIT<br/>Cosine Distance < 0.1| E[Return Cached LLM Response]
    E --> F[Display Response to User]
    F --> END[End]

    D -->|Cache MISS<br/>No similar query| G[Query Pinecone Vector DB<br/>Using same embedding vector]

    G --> H[Pinecone returns Top K=5<br/>with Cosine Similarity Scores]

    H --> I{Filter: Scores >= 0.5?}

    I -->|All Below 0.5| J[Fallback Response:<br/>No specific information found]
    J --> K[Cache Fallback in RedisVL<br/>Store with same embedding vector]
    K --> F

    I -->|At Least One >= 0.5| L[Extract Node IDs from Matches]

    L --> M[Batch Query Neo4j Graph DB]
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

## Instructions to run the hybrid chat

### Prerequisites

**1. Run Redis container:**

```bash
docker run -d --name redis -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

**2. Run Neo4j container:**

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest
```

**3. Set up environment variables:**

Create a `.env.local` file with:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=vietnam-travel
PINECONE_VECTOR_DIM=1536

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Cache Settings
CACHE_ENABLED=true
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_THRESHOLD=0.1
```

**4. Install dependencies:**

```bash
pip install -r requirements.txt
```

**5. Run the application:**

```bash
python hybrid_chat.py
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
