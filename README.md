# MEMLIB-AI

`memlib` is a lightweight memory library for AI agents that provides persistent memory across conversations and sessions. It is a simplified implementation inspired by memory systems such as Mem0 and LangMem.

## V1 Memory Architecture

Memlib currently maintains four types of persistent data:

* **Conversation history** — raw user/assistant messages stored in SQLite and associated with a `user_id` and `chat_id`.
* **Conversation summaries** — compact summaries of individual conversations, stored persistently in SQLite.
* **Global memories** — durable user facts, preferences, goals, and instructions stored as individual records in SQLite with unique memory IDs.
* **Global memory embeddings** — embeddings of global memories stored in a provider-adaptive vector store for semantic retrieval.

The same `memory_id` is used between SQLite and the vector store so each embedding maps directly to its corresponding global memory.

## Features

* **Model-adaptive** — use LangChain-compatible chat models from providers such as Groq, OpenAI, Gemini, Anthropic, or Mistral.
* **Vector-store adaptive** — designed around LangChain-compatible vector stores such as Chroma, FAISS, and Pinecone.
* **Persistent memory** — global memories and conversation data survive process restarts.
* **User-scoped memory** — memories and vector retrieval are filtered by `user_id`.
* **Conversation-scoped context** — conversation history and summaries are associated with `chat_id`.
* **LLM-based memory extraction** — extracts durable information from the latest user/assistant turn.
* **LLM-based memory resolution** — decides whether a memory should be `ADD`, `UPDATE`, `DELETE`, or `NOOP`.
* **Semantic retrieval** — retrieves relevant global memories using vector similarity.

## Installation

Clone the repository:

```bash
git clone https://github.com/SitanshuA091/memlib-ai.git
cd memlib-ai
```

Sync the project environment with `uv`:

```bash
uv sync
```

For the additional dependencies used by the test suite:

```bash
uv sync --group test
```

The project uses a `src` layout, so the package is installed as `memlib` through the project's build configuration.

## Environment Variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
## or the provider of your choice
```

Load environment variables in your application or test code with `python-dotenv`.

Keep `.env` out of version control:

```gitignore
.env
```

## Main Interface

`Memory` is the main public interface intended for developers building agents.

The developer provides a LangChain chat model, a user ID, a conversation ID, a persistent memory store, and a vector store.

### `Memory.add(user_message, assistant_response)`

Processes a completed conversation turn.

Internally it:

* stores the user and assistant messages;
* loads and updates the conversation summary;
* extracts durable candidate memories;
* retrieves similar memories for the current `user_id`;
* asks the update resolver to perform `ADD`, `UPDATE`, `DELETE`, or `NOOP`;
* updates SQLite and synchronizes the corresponding vector-store entry.

### `Memory.search(query, limit=5)`

Retrieves relevant long-term memories for the current user using semantic similarity.

The search is automatically scoped to the `user_id` associated with the `Memory` instance.

### `Memory.get_context(query, limit=5)`

Retrieves relevant global memories and formats them as prompt-ready context.

This is intended to be directly injected into an agent or LLM prompt before generating a response.

### `Memory.clear(user_id=None)`

Removes all global memories for a user and removes their corresponding vector embeddings.

### `Memory.delete(memory_id)`

Deletes one global memory and its corresponding vector embedding.

## Typical Agent Flow

The intended usage pattern is:

```text
User message
    ↓
memory.get_context(query)
    ↓
Agent / LLM generates response
    ↓
memory.add(user_message, assistant_response)
```

`get_context()` retrieves only relevant long-term memories rather than passing the entire global memory store into the model.

## Internal Components

The current implementation is intentionally small:

* `memory.py` — main public API and orchestration.
* `types.py` — shared data models such as `Message`, `CandidateMemory`, `MemoryItem`, and `MemoryOperation`.
* `extractor.py` — LLM-based extraction of durable user memories.
* `updater.py` — LLM-based `ADD` / `UPDATE` / `DELETE` / `NOOP` resolution.
* `retriever.py` — semantic retrieval of global memories.
* `summarizer.py` — LLM-based conversation summarization.
* `store.py` — SQLite persistence for conversation history, global memories, and conversation summaries.
* `vector_store.py` — vector-store abstraction for global memory embeddings.
* `llm.py` — LangChain-based model adapter.

## Testing

The repository contains `tests/test_memory.py` as an integration/experimental test script.

It exercises the public `Memory` interface with a real LLM and vector store and can be used to observe:

* memory extraction;
* `ADD`, `UPDATE`, `DELETE`, and `NOOP` behaviour;
* persistent global memories;
* semantic retrieval;
* conversation summaries;
* user-specific memory retrieval.

Run it from the project root with:

```bash
uv run python -m tests.test_memory
```

The test uses the configured `GROQ_API_KEY` and the model specified in the test script.

## Current Scope

V1 focuses on **fact-based long-term memory** and persistent conversation context.

Knowledge-graph memory, entity extraction, relationship memory, more advanced retrieval strategies, and additional storage backends are planned as future extensions rather than being part of the current core implementation.
