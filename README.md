## **MEMLIB-AI**
memlib gives AI agents persistent memory across sessions. It is a simplified mini-implementation inspired by Mem0's extraction and update flows, as well as frameworks like langmem.

**Memories we store (V1)**
---

- Entire conversation history (SQLite for now).
- Conversation history summaries (summarized versions of each conversation session).
- Global summaries with candidate facts (SQLite for now).
- Global summaries in embeddings (provider-adaptive across Chroma, FAISS, and Pinecone).

### Base Features and Usage
- Model-adaptive – can be used with any model provider's API (Groq, Gemini, OpenAI, Mistral) as LLM clients, implemented via LangChain interfaces.
- Vector databases are also provider-adaptive across Chroma DB, FAISS, and Pinecone.
- Memory.py is the main interface with distinct public methods.
- Allows user_id-based filtering and storage for memories.

<em> this is a simplified version of a memory abstraction framework, looking forwards main interface will be updated and usage will be documented more clearly, Knowledge graphs would be included to better represent candidate memories and improve the context at query time. `test_Memory.py` is the experiment script that can be used to test extraction behaviour memory storages without using the main memory interface </em>