# API Reference

## `memlib.memory.Memory`

```python
Memory(
    llm: LLMClient,
    chat_model: BaseChatModel,
    user_id: str,
    chat_id: str,
    store: MemoryStore,
    vector_store: MemoryVectorStore,
    graph_store: GraphStore | None = None,
)
```

### `add(user_message: str, assistant_response: str) -> None`
Processes a completed conversation turn: stores it, updates the running summary, extracts candidate memories, resolves each against similar existing memories, and applies the resulting `ADD` / `UPDATE` / `DELETE` / `NOOP`.

### `search(query: str, *, limit: int = 5) -> list[MemoryItem]`
Vector search resolved back to canonical `MemoryItem`s. No graph routing, no formatting.

### `get_context(query: str, *, limit: int = 5, graph_limit: int = 10) -> str`
Prompt-ready string combining retrieved memories and, when the router decides it's useful, graph relationship facts. Returns `""` when nothing relevant is found.

### `delete(memory_id: str) -> None`
Removes the memory from SQLite, the vector store, and (if configured) the graph.

### `clear() -> None`
Removes every memory belonging to `user_id` across all three layers.

---

## `memlib.store.MemoryStore`

SQLite-backed canonical store. Tables: `messages`, `memories`, `summaries`.

| Method | Purpose |
|---|---|
| `add_messages(chat_id, user_id, messages)` | Append a conversation turn |
| `get_recent_messages(chat_id, user_id, limit=2)` | Fetch the most recent turns, oldest first |
| `add_memory(memory_id, user_id, content, memory_type, metadata=None)` | Insert a new canonical memory |
| `update_memory(memory_id, content, metadata=None)` | Update content in place, same ID |
| `delete_memory(memory_id)` | Remove a canonical memory |
| `get_memory(memory_id)` | Fetch one memory as a `MemoryItem` |
| `get_memories(user_id)` | Fetch every memory for a user, oldest first |
| `get_summary(chat_id, user_id)` | Fetch the running conversation summary |
| `save_summary(chat_id, user_id, summary)` | Upsert the conversation summary |

---

## `memlib.vector_store.MemoryVectorStore`

```python
MemoryVectorStore(vector_store: VectorStore)
```

| Method | Purpose |
|---|---|
| `add(memory_id, content, user_id, metadata=None)` | Embed and tag a new memory |
| `update(memory_id, content, user_id, metadata=None)` | Delete + re-add the embedding |
| `delete(memory_id)` | Remove the embedding |
| `search(query, user_id, *, limit=5) -> list[Document]` | User-filtered similarity search |

---

## `memlib.graph_store`

### `GraphStore` (abstract base)
`add_fact`, `delete_memory`, `search`, `clear_user`, `close`.

### `Neo4jGraphStore(uri, username, password, *, database=None)`
Neo4j implementation of `GraphStore`. `search` reduces the query to lowercase non-stopword terms and matches them against entity and relation names with `CONTAINS`.

---

## `memlib.llm.LLMClient`

```python
LLMClient(llm: BaseChatModel)
```

| Method | Purpose |
|---|---|
| `complete(system_prompt, user_prompt) -> str` | System + user message completion, normalized to a plain string |
| `invoke(messages) -> str` | Generic invoke wrapper for advanced usage |

---

## `memlib.types`

| Type | Fields |
|---|---|
| `Message` | `role`, `content` |
| `CandidateMemory` | `content`, `memory_type`, `metadata` |
| `MemoryItem` | `id`, `content`, `metadata` |
| `MemoryUpdate` | `action` (`ADD`/`UPDATE`/`DELETE`/`NOOP`), `memory_id`, `content`, `memory_type`, `metadata` |
| `RetrievalDecision` | `needs_graph`, `reason` |
| `GraphFact` | `subject`, `relation`, `object`, `memory_id`, `user_id`, `metadata` |

---

## Internal components (not part of the public API, but referenced throughout the docs)

`MemoryExtractor`, `MemoryUpdater`, `MemoryRetriever`, `MemoryRouter`, `GraphExtractor`, `ConversationSummarizer` — each is instantiated internally by `Memory.__init__` and not meant to be constructed directly by application code.
