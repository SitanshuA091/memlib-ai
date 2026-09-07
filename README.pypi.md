# memlib

`memlib` is a lightweight persistent-memory layer for AI agents and conversational applications.

It is designed to plug into existing LangChain or LangGraph applications without controlling the agent architecture itself.

A typical integration may look like:

```text
User message
    ↓
memlib.get_context(...)
    ↓
LangChain / LangGraph agent
    ↓
Assistant response
    ↓
memlib.add(...)
````

Memlib handles persistent user memory while your application remains responsible for agent execution, tools, routing, and response generation.

---

## Installation

```bash
pip install memlib
```

or:

```bash
uv add memlib
```

---

# Basic Usage

`Memory` is the main interface.

A memory instance is associated with:

* a `user_id`;
* a `chat_id`;
* an LLM adapter;
* a persistent store;
* a vector store;
* optionally, a graph store.

Example:

```python
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from memlib.llm import LLMClient
from memlib.memory import Memory
from memlib.store import MemoryStore
from memlib.vector_store import MemoryVectorStore


chat_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

llm = LLMClient(chat_model)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

chroma = Chroma(
    collection_name="agent_memories",
    embedding_function=embeddings,
    persist_directory="./memory_vectors",
)

vector_store = MemoryVectorStore(chroma)

store = MemoryStore("memory.db")

memory = Memory(
    llm=llm,
    chat_model=chat_model,
    user_id="user-123",
    chat_id="chat-456",
    store=store,
    vector_store=vector_store,
)
```

---

# Using Memory in an Agent Loop

The normal integration requires two operations:

```text
Before generating the response:
    memory.get_context(...)

After generating the response:
    memory.add(...)
```

## Retrieve memory before the agent runs

```python
user_message = "Which football club do I support?"

context = memory.get_context(
    user_message,
    limit=5,
)
```

`context` can then be inserted into your model or agent prompt.

For example:

```python
response = llm.complete(
    system_prompt=(
        "You are a helpful assistant. "
        "Use persistent user memory when it is relevant."
    ),
    user_prompt=(
        f"Persistent memory:\n"
        f"{context or 'No relevant memories found.'}\n\n"
        f"User message:\n"
        f"{user_message}"
    ),
)
```

After the response has been generated:

```python
memory.add(
    user_message=user_message,
    assistant_response=response,
)
```

Memlib can then use the completed turn to maintain persistent memory for future interactions.

---

# Integration with LangChain

Memlib does not replace LangChain.

Instead, it can sit alongside your existing chain or agent.

For example:

```python
user_message = "What technology am I currently learning?"

memory_context = memory.get_context(
    user_message,
    limit=5,
)

result = chain.invoke(
    {
        "input": user_message,
        "memory_context": memory_context,
    }
)

assistant_response = result["output"]

memory.add(
    user_message=user_message,
    assistant_response=assistant_response,
)
```

Your prompt can expose the memory context through a normal prompt variable:

```text
Persistent user context:
{memory_context}

Current user message:
{input}
```

This keeps memory retrieval separate from the agent's main reasoning and tool execution.

---

# Integration with LangGraph

Memlib can also be used as a memory layer around a LangGraph workflow.

A common flow is:

```text
START
  ↓
Retrieve persistent memory
  ↓
Agent / tool nodes
  ↓
Generate final response
  ↓
Persist completed turn
  ↓
END
```

For example:

```python
from typing import TypedDict


class AgentState(TypedDict):
    user_message: str
    memory_context: str
    response: str
```

A memory retrieval node can run before the agent:

```python
def retrieve_memory(state: AgentState) -> dict:
    context = memory.get_context(
        state["user_message"],
        limit=5,
    )

    return {
        "memory_context": context,
    }
```

Your agent node then receives both the current request and persistent context:

```python
def agent_node(state: AgentState) -> dict:
    prompt = f"""
Persistent user memory:
{state["memory_context"] or "No relevant memories found."}

User:
{state["user_message"]}
"""

    response = chat_model.invoke(prompt)

    return {
        "response": response.content,
    }
```

After the final response is generated, persist the completed turn:

```python
def save_memory(state: AgentState) -> dict:
    memory.add(
        user_message=state["user_message"],
        assistant_response=state["response"],
    )

    return {}
```

The graph can then be structured conceptually as:

```text
START
  ↓
retrieve_memory
  ↓
agent
  ↓
save_memory
  ↓
END
```

For larger LangGraph applications, the same pattern can wrap more complex flows:

```text
START
  ↓
retrieve_memory
  ↓
planner
  ↓
tools
  ↓
agent
  ↓
final_response
  ↓
save_memory
  ↓
END
```

Memlib remains independent of the graph topology.

---

# Provider-Adaptive LLM Usage

Memlib is designed to work with LangChain-compatible chat models.

The application selects the provider and model.

For example, with Groq:

```python
from langchain_groq import ChatGroq

chat_model = ChatGroq(
    model="openai/gpt-oss-120b",
)
```

With OpenAI:

```python
from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(
    model="gpt-5-mini",
)
```

With Gemini:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)
```

Then create the memlib adapter:

```python
from memlib.llm import LLMClient

llm = LLMClient(chat_model)
```

and pass both interfaces to `Memory`:

```python
memory = Memory(
    llm=llm,
    chat_model=chat_model,
    user_id="user-123",
    chat_id="chat-456",
    store=store,
    vector_store=vector_store,
)
```

This allows the surrounding agent application to choose its own LangChain-compatible provider.

---

# Different Models for Different Agent Responsibilities

Your main agent model does not need to be the same model used for every part of your application.

For example, your LangGraph agent may use one provider while memlib is configured with another:

```python
agent_model = ChatOpenAI(
    model="gpt-5-mini",
)

memory_model = ChatGroq(
    model="openai/gpt-oss-120b",
)

memory_llm = LLMClient(memory_model)

memory = Memory(
    llm=memory_llm,
    chat_model=memory_model,
    user_id="user-123",
    chat_id="chat-456",
    store=store,
    vector_store=vector_store,
)
```

Your LangGraph nodes can continue using:

```python
agent_model
```

while memlib uses:

```python
memory_model
```

for memory-related operations.

This keeps the memory layer decoupled from the application's main agent model.

---

# Knowledge Graph Support

Knowledge-graph retrieval can be enabled by supplying a graph store.

Example with Neo4j:

```python
from memlib.graph_store import Neo4jGraphStore


graph_store = Neo4jGraphStore(
    uri="neo4j+s://...",
    username="neo4j",
    password="your-password",
    database="your-database",
)
```

Then:

```python
memory = Memory(
    llm=llm,
    chat_model=chat_model,
    user_id="user-123",
    chat_id="chat-456",
    store=store,
    vector_store=vector_store,
    graph_store=graph_store,
)
```

You do not need to change the normal application flow.

The same call remains:

```python
context = memory.get_context(user_message)
```

Memlib can decide whether normal semantic memory retrieval is sufficient or whether additional relationship-based context should be included.

For example, previously stored information such as:

```text
User follows football.
User's favourite club is Manchester United.
```

can also contribute relationship context such as:

```text
User --FOLLOWS--> Football
User --FAVOURITE_CLUB--> Manchester United
```

From the application developer's perspective, the interface remains:

```python
memory.get_context(...)
```

---

# Multiple Users

Use a different `user_id` for each application user.

```python
memory = Memory(
    llm=llm,
    chat_model=chat_model,
    user_id=current_user.id,
    chat_id=current_chat.id,
    store=store,
    vector_store=vector_store,
)
```

Persistent memories are associated with the user rather than with a single conversation.

---

# Multiple Conversations

Keep the same `user_id` and change the `chat_id` for new conversations:

```python
memory = Memory(
    llm=llm,
    chat_model=chat_model,
    user_id="user-123",
    chat_id="chat-789",
    store=store,
    vector_store=vector_store,
)
```

This allows conversation-specific context to remain separate while long-term user memory can persist across sessions.

---

# Searching Memories Directly

Use:

```python
memories = memory.search(
    "What does the user enjoy?",
    limit=5,
)
```

The result contains relevant memory objects.

```python
for item in memories:
    print(item.id)
    print(item.content)
```

For most chatbot or agent integrations, however, `get_context()` is usually more convenient because it returns prompt-ready context.

---

# Deleting Memory

Delete a specific memory by ID:

```python
memory.delete(memory_id)
```

---

# Clearing User Memory

Clear the long-term memories associated with the current `Memory` instance:

```python
memory.clear()
```

---

# Main API

The primary application-facing methods are:

```python
memory.get_context(
    query,
    limit=5,
)
```

Retrieve prompt-ready persistent context.

```python
memory.search(
    query,
    limit=5,
)
```

Retrieve matching memory objects.

```python
memory.add(
    user_message,
    assistant_response,
)
```

Process a completed interaction and maintain persistent memory.

```python
memory.delete(
    memory_id,
)
```

Delete an individual memory.

```python
memory.clear()
```

Clear long-term memory for the current user.

---

# Recommended Agent Pattern

For most LangChain or LangGraph applications:

1. Receive user message

2. Retrieve memory
   `memory.get_context(user_message)`

3. Pass memory context into the agent state or prompt

4. Run the normal agent / tool workflow

5. Produce the final assistant response

6. Persist the completed turn
   `memory.add(user_message, assistant_response)`


This lets the agent architecture remain independent while memlib provides a persistent memory layer around it.

Source Code - https://github.com/SitanshuA091/memlib-ai

