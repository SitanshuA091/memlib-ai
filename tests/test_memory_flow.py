import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from memlib.graph_store import Neo4jGraphStore
from memlib.llm import LLMClient
from memlib.memory import Memory
from memlib.store import MemoryStore
from memlib.vector_store import MemoryVectorStore


load_dotenv()


USER_ID = "test-user-1"
CHAT_ID = "test-chat-1"


def main() -> None:
    groq_api_key = os.getenv("GROQ_API_KEY")

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE")

    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    if not neo4j_uri:
        raise RuntimeError("NEO4J_URI is not set.")

    if not neo4j_username:
        raise RuntimeError("NEO4J_USERNAME is not set.")

    if not neo4j_password:
        raise RuntimeError("NEO4J_PASSWORD is not set.")
    
    if not neo4j_database:
        raise RuntimeError("NEO4J_DATABASE is not set.")

    # ---------------------------------------------------------
    # LLM
    # ---------------------------------------------------------

    # Developer-selected LangChain chat model.
    chat_model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=groq_api_key,
    )

    # Existing memlib completion wrapper used by this test
    # to generate the assistant response.
    llm = LLMClient(chat_model)

    # ---------------------------------------------------------
    # Embeddings / Vector Store
    # ---------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    chroma = Chroma(
        collection_name="memlib_memories",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    vector_store = MemoryVectorStore(chroma)

    # ---------------------------------------------------------
    # SQLite
    # ---------------------------------------------------------

    store = MemoryStore("memlib.db")

    # ---------------------------------------------------------
    # Knowledge Graph
    # ---------------------------------------------------------

    graph_store = Neo4jGraphStore(
        uri=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
        database=neo4j_database,
    )

    # ---------------------------------------------------------
    # Main memlib interface
    # ---------------------------------------------------------
    
    memory = Memory(
    llm=llm,
    chat_model=chat_model,
    user_id=USER_ID,
    chat_id=CHAT_ID,
    store=store,
    vector_store=vector_store,
    graph_store=graph_store,
)


    print("Memlib memory-flow test started.")
    print("SQLite + Vector Store + Knowledge Graph enabled.")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            user_message = input("You: ").strip()

            if user_message.lower() == "exit":
                break

            if not user_message:
                continue

            # -------------------------------------------------
            # QUERY PIPELINE
            #
            # Query
            #   -> vector retrieval
            #   -> canonical SQLite memories
            #   -> router
            #   -> optional Neo4j retrieval
            #   -> combined context
            # -------------------------------------------------

            context = memory.get_context(
                user_message,
                limit=5,
            )

            print("\nRetrieved context:")
            print(context or "  No relevant memory context.")
            print()

            response = llm.complete(
                system_prompt=(
                    "You are a helpful assistant with access to persistent "
                    "user memories. Use the supplied memories only when relevant. "
                    "Do not invent personal information."
                ),
                user_prompt=(
                    f"Persistent memory context:\n"
                    f"{context or 'No relevant memories found.'}\n\n"
                    f"User message:\n"
                    f"{user_message}"
                ),
            ).strip()

            print(f"Assistant: {response}\n")

            # -------------------------------------------------
            # WRITE PIPELINE
            #
            # conversation
            #   -> extractor
            #   -> updater
            #   -> SQLite canonical memory
            #   -> vector embedding
            #   -> graph extraction
            #   -> Neo4j relationship
            # -------------------------------------------------

            memory.add(
                user_message=user_message,
                assistant_response=response,
            )

            # -------------------------------------------------
            # Debug: canonical global memories
            # -------------------------------------------------

            memories = store.get_memories(USER_ID)

            print("Global memories:")

            if memories:
                for item in memories:
                    print(
                        f"  [{item.id}] "
                        f"{item.content}"
                    )
            else:
                print("  None")

            # -------------------------------------------------
            # Debug: conversation summary
            # -------------------------------------------------

            summary = store.get_summary(
                chat_id=CHAT_ID,
                user_id=USER_ID,
            )

            print("\nConversation summary:")
            print(summary or "  None")

            # -------------------------------------------------
            # Debug: Neo4j relationships
            #
            # Graph facts normally use User as their subject,
            # so this gives us a simple integration check.
            # -------------------------------------------------

            graph_facts = graph_store.search(
                query="User",
                user_id=USER_ID,
                limit=50,
            )

            print("\nKnowledge graph relationships:")

            if graph_facts:
                for fact in graph_facts:
                    print(
                        f"  "
                        f"{fact['subject']} "
                        f"--{fact['relation']}--> "
                        f"{fact['object']} "
                        f"[memory_id={fact['memory_id']}]"
                    )
            else:
                print("  None")

            print("\n" + "-" * 70)
            print("Ready for the next message.\n")

    finally:
        graph_store.close()
        store.close()

    print("Memlib memory-flow test ended.")


if __name__ == "__main__":
    main()