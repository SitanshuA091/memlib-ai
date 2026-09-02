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

    # End user chooses the LangChain chat model.
    llm = LLMClient(
        ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=api_key,
        )
    )

    # Embedding model used by the vector store.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Persistent SQLite storage.
    store = MemoryStore("memlib.db")

    # Persistent Chroma vector storage.
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
    print("Type 'exit' to stop.\n")

    while True:
        user_message = input("You: ").strip()

        if user_message.lower() == "exit":
            break

        if not user_message:
            continue

        # Retrieve persistent memories before generating the response.
        context = memory.get_context(user_message)

        response = llm.complete(
            system_prompt=(
                "You are a helpful assistant. "
                "Use the supplied user memory only when relevant. "
                "Do not invent personal information."
            ),
            user_prompt=(
                f"Relevant user memories:\n"
                f"{context or 'No relevant memories found.'}\n\n"
                f"User message:\n{user_message}"
            ),
        )

        print(f"Assistant: {response}")

        # Main memory API.
        memory.add(
            user_message=user_message,
            assistant_response=response,
        )

        # Inspect persistent global memories for this user.
        print("\nGlobal memories:")

        memories = store.get_memories(USER_ID)

        if not memories:
            print("  None")
        else:
            for item in memories:
                print(f"  [{item.id}] {item.content}")

        # Inspect the persisted conversation summary.
        summary = store.get_summary(
            chat_id=CHAT_ID,
            user_id=USER_ID,
        )

        print("\nConversation summary:")
        print(summary or "  None")

        print("\n" + "-" * 70 + "\n")

    store.close()


if __name__ == "__main__":
    main()
