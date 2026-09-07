# MEMLIB-AI

`memlib` is a lightweight persistent-memory library for AI agents.

It provides long-term memory across conversations and sessions using:

- canonical memories in SQLite;
- semantic retrieval through vector embeddings;
- conversation summaries;
- relationship-aware knowledge-graph retrieval.

The project is inspired by systems such as Mem0 and LangMem, while keeping the architecture intentionally small and understandable.

---

## Memory Architecture (V1)

Memlib currently maintains five persistent data layers.

### 1. Conversation History

Raw user and assistant messages are stored in SQLite and scoped by:

- `user_id`
- `chat_id`

### 2. Conversation Summaries

Each chat maintains a compact persistent summary in SQLite.

Summaries preserve important context such as:

- ongoing tasks;
- decisions;
- explicit preferences;
- constraints;
- unresolved context.

### 3. Global Memories

Durable user information such as facts, preferences, interests, goals, and instructions is stored in SQLite.

Each memory has a unique `memory_id`.

SQLite is the **canonical source of truth** for long-term memory.

### 4. Global Memory Embeddings

Global memories are embedded into a LangChain-compatible vector store for semantic retrieval.

Each vector entry contains:

- `memory_id`
- `user_id`

The same `memory_id` links the vector representation back to the canonical SQLite memory.

### 5. Knowledge Graph

Canonical memories may also be converted into structured relationships and stored in Neo4j.

Example:

```text
"User loves Manchester United."
````

becomes:

```text
User --LOVES--> Manchester United
```

Each graph relationship stores:

* `memory_id`
* `user_id`

The graph is a **derived representation**, not a second memory source.

---

## Core Design

```text
SQLite
  -> canonical memory

Vector Store
  -> semantic similarity retrieval

Knowledge Graph
  -> relationship retrieval
```

All derived representations remain linked through `memory_id`.

---

## Features

* persistent memory across sessions;
* user-scoped retrieval using `user_id`;
* conversation-scoped summaries using `chat_id`;
* LLM-based memory extraction;
* LLM-based `ADD`, `UPDATE`, `DELETE`, and `NOOP` resolution;
* semantic retrieval through vector similarity;
* canonical memory resolution back to SQLite;
* Neo4j relationship storage;
* graph-aware retrieval routing;
* memory provenance through shared `memory_id`;
* LangChain-compatible model support;
* LangChain-compatible vector-store support;
* optional graph backend.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SitanshuA091/memlib-ai.git
cd memlib-ai
```

Sync dependencies:

```bash
uv sync
```

For test dependencies:

```bash
uv sync --group test
```

The project uses a `src` layout and exposes the package as:

```python
import memlib
```

---

## Environment Variables

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key

NEO4J_URI=your_neo4j_aura_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=your_neo4j_database

HF_TOKEN=your_huggingface_token
```

Other provider keys such as OpenAI, Gemini, Anthropic, or Mistral may also be used depending on the selected LangChain model.

Keep `.env` out of version control:

```gitignore
.env
```

---

## Main Interface

`Memory` is the main public API.

A developer provides:

* an `LLMClient`;
* a LangChain `BaseChatModel`;
* a `user_id`;
* a `chat_id`;
* a `MemoryStore`;
* a `MemoryVectorStore`;
* optionally, a `GraphStore`.

Example:

```python
memory = Memory(
    llm=llm,
    chat_model=chat_model,
    user_id=user_id,
    chat_id=chat_id,
    store=store,
    vector_store=vector_store,
    graph_store=graph_store,
)
```

Internally:

```text
LLMClient
  -> extraction
  -> update resolution
  -> summarization

BaseChatModel
  -> retrieval routing
  -> graph extraction
```

---

## `Memory.add(user_message, assistant_response)`

Processes a completed conversation turn.

```text
User + Assistant
        ↓
Conversation History
        ↓
Conversation Summary
        ↓
Memory Extractor
        ↓
Candidate Memories
        ↓
Semantic Retrieval
        ↓
Updater
        ↓
ADD / UPDATE / DELETE / NOOP
        ↓
Canonical SQLite Memory
        ├── Vector Store
        └── Graph Extractor -> Neo4j
```

### ADD

Creates a new memory and synchronizes it to the vector store and graph.

### UPDATE

Updates the canonical memory, replaces its vector representation, removes old graph relationships, and rebuilds them from the updated memory.

### DELETE

Removes the memory from SQLite, the vector store, and Neo4j.

### NOOP

Makes no storage changes and avoids duplicate memories.

---

## `Memory.search(query, limit=5)`

Performs semantic retrieval.

```text
Query
  ↓
Vector Search
  ↓
memory_id
  ↓
Canonical SQLite Memory
```

The vector store acts only as a retrieval index.

Returned memories are resolved back to SQLite so the application always works with canonical memory data.

---

## `Memory.get_context(query, limit=5)`

Builds prompt-ready persistent context.

The system first retrieves relevant memories semantically.

If a graph store is enabled, the router decides whether those memories are sufficient.

```text
Query
  ↓
Vector Retrieval
  ↓
Canonical Memories
  ↓
Router
  ├── sufficient -> return memory context
  └── graph needed -> query Neo4j
                         ↓
                    combined context
```

Graph retrieval is therefore optional and only used when relational context adds useful information.

---

## Knowledge Graph Design

Graph relationships are created only from finalized canonical memories.

Example:

```text
Canonical memory:
"User is learning Gaussian splatting."
```

may produce:

```text
User --LEARNING--> Gaussian splatting
```

Each graph relationship stores provenance through its `memory_id`.

On memory updates, old graph relationships for that memory are removed and re-extracted.

On deletion, corresponding graph relationships are removed as well.

---

## Graph Retrieval

The current Neo4j search is intentionally simple.

Natural-language queries are reduced to useful terms and matched against:

* subject names;
* relationship names;
* object names.

For example:

```text
Which football club do I love?
```

may use terms such as:

```text
football
club
love
```

This is a lightweight V1 strategy and can later be replaced by more advanced graph traversal or entity-based retrieval.

---

## Typical Agent Flow

```text
User message
    ↓
memory.get_context(query)
    ↓
Agent / LLM generates response
    ↓
memory.add(user_message, assistant_response)
```

Only relevant long-term memories are retrieved instead of injecting the entire memory store into the model.

---

## Internal Components

* `memory.py` — public API and orchestration.
* `types.py` — shared memory and graph data structures.
* `extractor.py` — durable memory extraction.
* `updater.py` — `ADD` / `UPDATE` / `DELETE` / `NOOP` resolution.
* `retriever.py` — semantic retrieval and canonical memory lookup.
* `summarizer.py` — conversation summarization.
* `store.py` — SQLite persistence.
* `vector_store.py` — vector-store abstraction.
* `router.py` — decides whether graph retrieval is needed.
* `graph_extractor.py` — converts canonical memories into graph relationships.
* `graph_store.py` — graph-store abstraction and Neo4j implementation.
* `llm.py` — model adapter used by memory components.

---
## Testing

The repository includes both automated pytest tests and an interactive end-to-end memory flow test.

Run all pytest-discoverable tests with:

```bash
uv run pytest -q tests
````

Run only the agent pipeline tests with:

```bash
uv run pytest -q tests/agent_pipeline
```

For more detailed pytest output:

```bash
uv run pytest tests/agent_pipeline -v
```

The interactive memory flow test is run separately:

```bash
uv run python -m tests.test_memory_flow
```

It exercises the full stack with SQLite, vector retrieval, LLM-based memory updates, routing, and Neo4j knowledge-graph synchronization.



## Current Scope

V1 currently supports:

* persistent conversation history;
* conversation summaries;
* canonical global memories;
* semantic vector retrieval;
* LLM-based extraction and update resolution;
* user-scoped retrieval;
* Neo4j relationship storage;
* graph-aware retrieval routing;
* synchronized provenance across SQLite, vectors, and graph relationships.

Possible future extensions include:


* richer entity extraction;
* configurable relationship ontologies;
* more advanced retrieval strategies;
* automated consistency checks across storage layers.

