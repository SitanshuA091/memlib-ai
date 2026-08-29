import os

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from memlib.extractor import MemoryExtractor
from memlib.llm import LLMClient
from memlib.retriever import MemoryRetriever
from memlib.store import MemoryStore
from memlib.summarizer import ConversationSummarizer
from memlib.types import Message
from memlib.updater import MemoryUpdater
from memlib.vector_store import MemoryVectorStore
from dotenv import load_dotenv
load_dotenv()


USER_ID = "test-user"
CHAT_ID = "test-chat"

store = MemoryStore("test_memlib.db")

# Groq LLM chosen by the end developer.
groq_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

llm = LLMClient(groq_llm)

# Embedding model used only for Chroma semantic search.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

chroma = Chroma(
    collection_name="memlib_test",
    embedding_function=embeddings,
    persist_directory="./test_chroma",
)

vector_store = MemoryVectorStore(chroma)

extractor = MemoryExtractor(llm)
summarizer = ConversationSummarizer(llm)
retriever = MemoryRetriever(vector_store)
updater = MemoryUpdater(
    llm=llm,
    store=store,
    user_id=USER_ID,
)

conversation_summary = ""


def generate_response(
    user_message: str,
    memories: list,
    summary: str,
) -> str:

    memory_context = "\n".join(
        f"- {memory.content}"
        for memory in memories
    )

    prompt = f"""
Conversation summary:
{summary or "No previous conversation summary."}

Relevant long-term memories:
{memory_context or "No relevant memories found."}

User message:
{user_message}

Answer the user naturally. Use the memories only when relevant.
"""

    return llm.complete(
        system_prompt=(
            "You are a helpful assistant with long-term memory."
        ),
        user_prompt=prompt,
    )


while True:
    user_message = input("\nYou: ").strip()

    if user_message.lower() == "exit":
        break

    # 1. Retrieve relevant global memories before answering.
    relevant_memories = retriever.search(
        user_message,
        limit=5,
    )

    # 2. Generate assistant response using retrieved memories.
    assistant_response = generate_response(
        user_message=user_message,
        memories=relevant_memories,
        summary=conversation_summary,
    )

    print(f"Assistant: {assistant_response}")

    messages = [
        Message(
            role="user",
            content=user_message,
        ),
        Message(
            role="assistant",
            content=assistant_response,
        ),
    ]

    # 3. Store raw conversation history.
    store.add_messages(
        chat_id=CHAT_ID,
        user_id=USER_ID,
        messages=messages,
    )

    # 4. Update conversation-specific summary.
    conversation_summary = summarizer.summarize(
        current_summary=conversation_summary,
        messages=messages,
    )

    # 5. Extract durable candidate memories.
    candidates = extractor.extract(messages)

    print("\nExtracted candidate memories:")

    if not candidates:
        print("  None")
    else:
        for candidate in candidates:
            print(f"  - {candidate.content}")

    # 6. Consolidate every candidate memory.
    for candidate in candidates:

        # Find similar existing global memories.
        similar_memories = retriever.search_candidate(
            candidate,
            limit=5,
        )

        # Ask the updater LLM for ADD / UPDATE / DELETE / NOOP.
        result = updater.update(
            candidate=candidate,
            similar_memories=similar_memories,
            messages=messages,
        )

        operation = result["operation"]

        print(f"Memory operation: {operation}")

        # 7. Synchronize Chroma using the SAME memory ID as SQLite.
        if operation.value == "ADD":
            vector_store.add(
                memory_id=result["id"],
                content=result["content"],
                metadata=candidate.metadata,
            )

        elif operation.value == "UPDATE":
            vector_store.update(
                memory_id=result["id"],
                content=result["content"],
                metadata=candidate.metadata,
            )

        elif operation.value == "DELETE":
            vector_store.delete(
                memory_id=result["id"],
            )

    # 8. Show everything currently stored for this user.
    print("\nGlobal memories:")

    memories = store.get_memories(USER_ID)

    if not memories:
        print("  None")
    else:
        for memory in memories:
            print(f"  [{memory.id}] {memory.content}")

    print(f"\nConversation summary:\n{conversation_summary}")