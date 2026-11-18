# memlib

**A lightweight persistent-memory layer for AI agents.**

`memlib` gives LangChain / LangGraph agents long-term memory across conversations and sessions, without taking over the agent architecture itself. Your application still owns routing, tools, and response generation — memlib just remembers.

[![PyPI](https://img.shields.io/pypi/v/memlib-ai)](https://pypi.org/project/memlib-ai/)
[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://pypi.org/project/memlib-ai/)

```bash
pip install memlib-ai
```

## Why memlib

Most memory layers force a choice: pick a provider, pick a vector database, pick a framework. memlib is built around three adapters instead:

- **Model-adaptive** — any LangChain `BaseChatModel` (Groq, OpenAI, Gemini, Anthropic, Mistral, ...) can drive extraction, update resolution, summarization, and routing.
- **Storage-adaptive** — any LangChain-compatible `VectorStore` (Chroma, FAISS, Pinecone, ...) can serve as the semantic index.
- **Graph-optional** — a Neo4j-backed knowledge graph can be layered on for relationship-aware retrieval, or left out entirely.

SQLite is always the canonical source of truth. The vector store and the graph are both *derived* indexes, kept in sync with it.

## The core loop

```text
Before the agent runs:   memory.get_context(user_message)
After the agent responds: memory.add(user_message, assistant_response)
```

That's the entire integration surface for a typical chatbot or agent.

## Where to go next

| I want to... | Read |
|---|---|
| Install memlib and run my first `Memory` instance | [Getting Started](getting-started.md) |
| Understand how the pieces fit together | [Architecture](architecture.md) |
| See exactly what happens on `add()` and `get_context()` | [Memory Lifecycle](memory-lifecycle.md) |
| Swap in a different embeddings/vector backend | [Vector Stores](vector-stores.md) |
| Enable relationship-aware retrieval | [Knowledge Graph](knowledge-graph.md) |
| Look up a method signature | [API Reference](api-reference.md) |
| Read the engineering writeup behind the design decisions | [Case Study](case-study.md) |

## Links

- [Source on GitHub](https://github.com/SitanshuA091/memlib-ai)
- [Package on PyPI](https://pypi.org/project/memlib-ai/)
