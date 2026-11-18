# Getting Started

## Install

```bash
pip install memlib-ai
```

or, with uv:

```bash
uv add memlib-ai
```

`memlib` declares `langchain-core` and `neo4j` as hard dependencies. The chat model, embeddings model, and vector store you actually use are *your* choice, so install whichever LangChain integration packages those require — e.g. `langchain-groq`, `langchain-chroma`, `langchain-huggingface`.

## Environment variables

Create a `.env` in your project root with whichever providers you're using:

```env
GROQ_API_KEY=your_groq_api_key

NEO4J_URI=your_neo4j_aura_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=your_neo4j_database

HF_TOKEN=your_huggingface_token
```

OpenAI, Gemini, Anthropic, or Mistral keys work the same way — memlib doesn't read them directly, your LangChain chat model does.

## Build a `Memory` instance

```python
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from memlib.llm import LLMClient
from memlib.memory import Memory
from memlib.store import MemoryStore
from memlib.vector_store import MemoryVectorStore

chat_model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
llm = LLMClient(chat_model)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

chroma = Chroma(
    collection_name="agent_memories",
    embedding_function=embeddings,
    persist_directory="./memory_vectors",
)

memory = Memory(
    llm=llm,
    chat_model=chat_model,
    user_id="user-123",
    chat_id="chat-456",
    store=MemoryStore("memory.db"),
    vector_store=MemoryVectorStore(chroma),
)
```

Five things define a `Memory` instance:

- `user_id` — scopes durable memories to a person.
- `chat_id` — scopes conversation history and the running summary to one conversation.
- `llm` — an `LLMClient` wrapping any LangChain `BaseChatModel`, used for extraction, update resolution, and summarization.
- `store` — the SQLite-backed canonical store.
- `vector_store` — the semantic index, wrapping any LangChain `VectorStore`.
- `graph_store` *(optional)* — a Neo4j-backed knowledge graph for relationship retrieval.

## The two calls you need

```python
# Before the agent runs
context = memory.get_context(user_message, limit=5)

# ... agent / LLM produces assistant_response using `context` ...

# After the agent responds
memory.add(user_message, assistant_response)
```

Everything else — extraction, dedup, updates, graph sync — happens inside `add()`.

## Run the test suite

```bash
uv sync --group test
uv run pytest -q tests
```

The interactive end-to-end flow (SQLite + vector retrieval + LLM updates + routing + Neo4j sync) is runnable on its own:

```bash
uv run python -m tests.test_memory_flow
```
