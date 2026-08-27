A minimal agent-facing API:
1. `Memory.add(...)`

* Called after a conversation turn.
* Takes:

  * user message
  * assistant response
* Internally:

  * extracts memories
  * updates global memory
  * updates summary

Example intent:

```python
memory.add(
    user_message,
    assistant_response
)
```

---

2. `Memory.search(query)`

* Used when the agent needs relevant past memories.
* Returns relevant memories from the memory store.

Example intent:

```python
memories = memory.search(user_query)
```

---

3. `Memory.get_context(query)`

* Convenience method for agents.
* Retrieves memories + formats them into a prompt-ready context string.

Example intent:

```python
context = memory.get_context(user_query)

messages.append(
    {"role": "system", "content": context}
)
```

---

4. `Memory.clear(user_id)` (optional)

* Removes stored memories for a user.

---

5. `Memory.delete(memory_id)` (optional)

* Allows explicit removal of a stored memory.

---

That is basically enough for an agent developer:

```text
User message
      |
      v
memory.search(query)
      |
      v
Inject retrieved context
      |
      v
LLM generates response
      |
      v
memory.add(user_message, response)
```

internal components (LLM extraction, update resolver, Chroma, SQLite, summaries) will keep hidden behind.
Required to know
* "Give memory the conversation"
* "Retrieve memory before answering"
* "Clear/delete if needed"
