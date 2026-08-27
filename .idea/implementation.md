## MEMORY LIBRARY (TINY & SIMPLE)
Memories we will store
- convo history in sqlite (entire set of user messgae and assistant response)(SQLITE)
- global summary for user (bunch of candidate facts)(SQLITE) (id1 -  user is Spider-Man)
- Vector Store of Global Summary (ChromaDB) (id1 - {embediings})
- Conversation memory summaries (at end of each user message agent response (LLM summarizes this query-response combo))

## BASIC FLOW
1. LLM summarizes the current conversation context and stores the updated conversation summary with metadata in the conversation summary storage.

2. Extracted `{user_message, assistant_response}` is used to identify candidate memories, then perform similarity search against the global memory store (which contains candidate facts/preferences/instructions with unique IDs and rows). Retrieve the top 5 most similar existing memories `{id, memory_entry}`.

3. Feed the retrieved top 5 memories along with `{user_message, assistant_response}` to another LLM (Update Resolver). The LLM decides whether to UPDATE an existing memory, ADD a new memory, DELETE an outdated memory, or perform NOOP. The update prompt should define these operations clearly and return the required memory ID and modified/new memory content.

4. Based on the update resolver output, update the global memory store and synchronize ChromaDB embeddings: replace the embedding for updated memories, generate and add embeddings for new memories, delete embeddings for removed memories, or make no changes for NOOP.

5. Query time: User query is converted into an embedding and similarity search is performed against ChromaDB to retrieve relevant memories. Do not feed the entire global memory store into the query LLM because of context limits; provide only retrieved relevant memories along with the conversation summary and optionally recent conversation context if useful. Full conversation history should not be included by default.
