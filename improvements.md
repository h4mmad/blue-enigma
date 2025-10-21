# Improvements and Fixes

## 1. Environment Variable Management & Security

**Problem:** Original project had hardcoded API keys in `config.py`, creating a security risk.

**Solution:**

- Created `.env` file to store sensitive credentials
- Created `.gitignore` file to prevent API keys from being exposed in version control
- Refactored `config.py` to load environment variables using `python-dotenv`
- Added validation to ensure required API keys are set

C Enhanced security, and prevents accidental credential exposure.

---

## 2. Dependency Updates & Package Management

**Problem:** Outdated Python packages causing import errors and compatibility issues.

**Issues Fixed:**

1. **Pinecone Package Migration**

   - Old: `pinecone-client==2.2.0` (deprecated)
   - New: `pinecone==7.3.0` (official package)
   - Error: `ImportError: cannot import name 'Pinecone' from 'pinecone'`

2. **OpenAI Package Update**
   - Old: `openai==1.0.0`
   - New: `openai==2.3.0`
   - Error: `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`

**Solution:**

- Uninstalled deprecated `pinecone-client` package
- Installed official `pinecone` package (v7.3.0)
- Upgraded `openai` to v2.3.0 for compatibility with httpx

**Commands Used:**

```bash
pip uninstall -y pinecone-client
pip install pinecone
pip install --upgrade openai
```

**Benefits:** Resolved import errors and improved compatibility.

## 3. Pinecone Configuration Updates

**Problem:** Incorrect Pinecone region configuration causing 404 errors.

**Error:** `Resource cloud: gcp region: us-east1-gcp not found`

**Solution:**

- Changed from GCP region to AWS region for free tier compatibility
- Updated [pinecone_upload.py:34-36](pinecone_upload.py#L34-L36):
  - Old: `cloud="gcp", region="us-east1-gcp"`
  - New: `cloud="aws", region="us-east-1"`

**Benefits:** Compatible with Pinecone free tier, enables index creation without paid subscription.

---

## 4. Environment File Configuration

**Problem:** Environment variables not loading from `.env.local` file.

**Solution:**

- Updated [config.py:9](config.py#L9) to explicitly load `.env.local`:
  - Old: `load_dotenv()`
  - New: `load_dotenv(".env.local")`

**Benefits:** Supports custom environment file naming, allows multiple environment configurations (dev, staging, prod).

---

## 5. Neo4j Database Setup & Configuration

**Problem:** Neo4j database was not installed or running, causing connection errors when executing `hybrid_chat.py`.

**Error:**

```
neo4j.exceptions.ServiceUnavailable: Couldn't connect to localhost:7687
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solution:**

### Installation Method: Docker (Recommended)

1. **Installed Neo4j using Docker container:**

   ```bash
   docker run -d \
     --name neo4j-travel \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=none \
     neo4j:latest
   ```

2. **Port Configuration:**

   - Port `7687`: Bolt protocol (database connections)
   - Port `7474`: HTTP/Browser interface for Neo4j Browser UI

3. **Authentication Configuration:**

   - Set `NEO4J_AUTH=none` to disable authentication for development
   - Updated [.env.local](config.py#L12-L14) with:
     ```
     NEO4J_URI=bolt://localhost:7687
     NEO4J_USER=neo4j
     NEO4J_PASSWORD=none
     ```

4. **Data Loading:**
   - Executed `load_to_neo4j.py` to populate the database
   - Successfully created 360 nodes (10 Cities, 100 Hotels, 100 Activities, 150 Attractions)
   - Successfully created 360 relationships connecting entities

**Useful Docker Commands:**

```bash
# Stop Neo4j
docker stop neo4j-travel

# Start Neo4j
docker start neo4j-travel

# View logs
docker logs neo4j-travel

# Remove container
docker stop neo4j-travel && docker rm neo4j-travel

# Access Neo4j Browser
# Open http://localhost:7474 in web browser
```

---

## 6. Deterministic Response Generation

**Problem:** System was generating different responses for identical queries, causing inconsistent recommendations and poor user experience.

**Issue Details:**

- Same user query: "plan a romantic trip to Vietnam"
- Response 1: Suggests "cozy hotel" and specific activities
- Response 2: Suggests "go to a cafe" and different activities
- Unacceptable for a travel company where consistency builds trust

**Root Causes Identified:**

### 6.1 Non-Deterministic Neo4j Query Results

**Problem:** Neo4j queries without `ORDER BY` return results in arbitrary order.

**Original Query** in [hybrid_chat.py:67-72](hybrid_chat.py#L67-L72):

```cypher
MATCH (n:Entity {id:$nid})-[r]-(m:Entity)
RETURN type(r) AS rel, labels(m) AS labels, m.id AS id,
       m.name AS name, m.type AS type, m.description AS description
LIMIT 10
```

**Issue:** Without `ORDER BY`, Neo4j can return the same 10 relationships in different orders due to:

- Internal storage ordering changes
- Query planner optimization differences
- Cache state variations
- Concurrent database operations

**Impact:** Different graph facts in the prompt → Different AI responses

**Solution:** Added `ORDER BY m.id` to ensure consistent ordering:

```cypher
MATCH (n:Entity {id:$nid})-[r]-(m:Entity)
RETURN type(r) AS rel, labels(m) AS labels, m.id AS id,
       m.name AS name, m.type AS type, m.description AS description
ORDER BY m.id  -- Ensures deterministic result ordering
LIMIT 10
```

### 6.2 OpenAI Temperature Parameter

**Problem:** Default or high `temperature` values introduce randomness in text generation.

**Original Setting:** `temperature=0.2` (allows creative variation)

**Why This Was Problematic:**

- Temperature controls randomness in word selection during generation
- Even small values like 0.2 can cause different word choices for identical context
- Not suitable for business applications requiring consistent recommendations

**Temperature Scale:**
| Value | Behavior | Use Case |
|-------|----------|----------|
| 0.0 | Deterministic (greedy decoding) | Business recommendations ✅ |
| 0.2 | Slight creativity | Original setting ❌ |
| 0.7 | Balanced creativity | Creative writing |
| 1.0+ | High randomness | Brainstorming |

**Solution:** Changed to `temperature=0.0` in [hybrid_chat.py:127](hybrid_chat.py#L127):

```python
temperature=0.0  # Changed from 0.2 for deterministic responses
```

**Benefits:**

- Same context always produces similar response
- Consistent recommendations for similar queries
- Reliable metrics for A/B testing
- Builds user trust through predictability

### 6.3 OpenAI Seed Parameter

**Problem:** Even with `temperature=0.0`, minor variations can occur due to infrastructure differences.

**Solution:** Added `seed=42` parameter in [hybrid_chat.py:128](hybrid_chat.py#L128):

```python
seed=42  # Added seed for maximum determinism
```

**What the Seed Does:**

- Controls the random number generator used during text generation
- Makes sampling more reproducible across API calls
- Industry standard: Use consistent seed (42 is conventional) for deterministic behavior

**Note:** OpenAI states that determinism is "not guaranteed" due to model updates and infrastructure changes, but adding a seed significantly improves consistency.

---

## 7. Graph Visualization Fix

**Problem:** `visualize_graph.py` script failed with `TypeError` when attempting to generate graph visualization.

**Error:**

```
TypeError: Network.show() got an unexpected keyword argument 'notebook'
```

**Root Cause:**
The pyvis library API changed in newer versions. The `show()` method no longer accepts the `notebook` parameter - it's only specified during `Network()` initialization.

**Original Code** in [visualize_graph.py:33](visualize_graph.py#L33):

```python
net.show(output_html, notebook=False)  # notebook parameter not supported here
```

**Solution:**

```python
net.show(output_html)  # Removed notebook parameter
```

**Additional Fix:**
Added proper driver cleanup to prevent resource warning:

```python
driver.close()  # Properly close the driver to avoid resource warnings
```

**Result:**

- Successfully generates `neo4j_viz.html`

---
