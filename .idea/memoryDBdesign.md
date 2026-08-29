1. Conversation History Store (SQLite)
Stores raw chat history,
Schema:
messages
---------
chat_id
user_id
role
content
timestamp
2. Global Memory Store (SQLite)
Stores extracted durable candidate memories, 
memories
---------
id
user_id
content
type
created_at
row example - User works in ML
type=fact
3. Vector Store (Chroma/Pinecone/Faiss)
Embeddings of the global summary 
example - 
id=10
embedding=[0.23,0.51,...]
id =10 for global summary sqlite should be same for the global summary id fact's text to embeddings
its purpose is for semantic retrieval
4. Conversation Summary Store (Optional but useful)
This is separate from memories.
It stores a compressed overview of the conversation.
summaries
---------
user_id
summary_text
updated_at
example 
user_id 
<em> match chat_id with convo history chat_id </em>