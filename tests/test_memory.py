import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from memlib.llm import LLMClient
from memlib.memory import Memory
from memlib.store import MemoryStore
from memlib.vector_store import MemoryVectorStore


load_dotenv()


USER_ID = "test-user-1"
CHAT_ID = "test-chat-1"


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    # Developer-selected LangChain chat model.
    llm = LLMClient(
        ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=api_key,
        )
    )

    # Embedding model for global-memory semantic search.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Persistent SQLite storage.
    store = MemoryStore("memlib.db")

    # Persistent Chroma storage.
    chroma = Chroma(
        collection_name="memlib_memories",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    vector_store = MemoryVectorStore(chroma)

    # Main public memlib interface.
    memory = Memory(
        llm=llm,
        user_id=USER_ID,
        chat_id=CHAT_ID,
        store=store,
        vector_store=vector_store,
    )

    print("Memlib test started.")
    print("Type 'exit' to quit.\n")

    while True:
        user_message = input("You: ").strip()

        if user_message.lower() == "exit":
            break

        if not user_message:
            continue

        # Retrieve relevant persistent global memories.
        context = memory.get_context(
            user_message,
            limit=5,
        )

        response = llm.complete(
            system_prompt=(
                "You are a helpful assistant with access to persistent "
                "user memories. Use the supplied memories only when relevant. "
                "Do not invent personal information."
            ),
            user_prompt=(
                f"Relevant user memories:\n"
                f"{context or 'No relevant memories found.'}\n\n"
                f"User message:\n"
                f"{user_message}"
            ),
        ).strip()

        print(f"Assistant: {response}\n")

        # Persist conversation + extract/update global memories +
        # persist conversation summary.
        memory.add(
            user_message=user_message,
            assistant_response=response,
        )

        # Show current global memories for debugging.
        memories = store.get_memories(USER_ID)

        print("Global memories:")

        if memories:
            for item in memories:
                print(f"  [{item.id}] {item.content}")
        else:
            print("  None")

        # Show persisted conversation summary.
        summary = store.get_summary(
            chat_id=CHAT_ID,
            user_id=USER_ID,
        )

        print("\nConversation summary:")
        print(summary or "  None")

        print("\n" + "-" * 70)
        print("Ready for the next message.\n")

    store.close()
    print("Memlib test ended.")


if __name__ == "__main__":
    main()