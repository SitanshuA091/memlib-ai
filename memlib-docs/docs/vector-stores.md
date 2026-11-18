# Vector Stores

`MemoryVectorStore` wraps any LangChain `VectorStore` and adds exactly the bookkeeping memlib needs: tagging documents with `memory_id` and `user_id`, and filtering searches by user.

```python
class MemoryVectorStore:
    def __init__(self, vector_store: VectorStore) -> None: ...
    def add(self, memory_id, content, user_id, metadata=None) -> None: ...
    def update(self, memory_id, content, user_id, metadata=None) -> None: ...
    def delete(self, memory_id) -> None: ...
    def search(self, query, user_id, *, limit=5) -> list[Document]: ...
```

- **`add`** embeds `content` as a LangChain `Document`, with `memory_id` and `user_id` merged into its metadata, using the embedding's own ID as the `memory_id` (so vector-store ID and SQLite `memory_id` are always the same value).
- **`update`** is implemented as `delete` followed by `add` — there's no in-place embedding update, since most vector stores don't support one uniformly.
- **`search`** calls `similarity_search(query, k=limit, filter={"user_id": user_id})` — user isolation is enforced at the vector-store filter level, not after the fact in Python.

## Swapping backends

Any LangChain `VectorStore` works. Chroma is used in the README examples because it needs no external service, but FAISS, Pinecone, Weaviate, Qdrant, or anything else LangChain supports plugs in the same way:

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

chroma = Chroma(
    collection_name="agent_memories",
    embedding_function=embeddings,
    persist_directory="./memory_vectors",
)

vector_store = MemoryVectorStore(chroma)
```

Only the embeddings model and the `VectorStore` construction change — `MemoryVectorStore`, `Memory`, and everything downstream are unaffected.

## Why the vector store is never authoritative

`MemoryRetriever.search()` takes the `memory_id` out of each hit's metadata and looks the memory up in SQLite — it never returns the vector store's own copy of the content. If a vector store's stored text ever drifted from SQLite (a partial write, a migration, a backend swap mid-flight), the application would still only ever see the canonical SQLite content. The vector store's only job is telling memlib *which* memories are relevant, not *what* they say.
