from collections.abc import Sequence
from memlib.extractor import MemoryExtractor
from memlib.retriever import MemoryRetriever
from memlib.store import MemoryStore
from memlib.summarizer import ConversationSummarizer
from memlib.types import CandidateMemory, MemoryItem, Message
from memlib.updater import MemoryUpdater
from memlib.vector_store import MemoryVectorStore


class Memory:
    """Main user-facing interface for memlib."""

    def __init__(
        self,
        *,
        llm,
        user_id: str,
        chat_id: str,
        store: MemoryStore,
        vector_store: MemoryVectorStore,
    ) -> None:
        self.llm = llm
        self.user_id = user_id
        self.chat_id = chat_id

        self.store = store
        self.vector_store = vector_store

        self.extractor = MemoryExtractor(llm)
        self.retriever = MemoryRetriever(vector_store)
        self.updater = MemoryUpdater(
            llm=llm,
            store=store,
            user_id=user_id,
        )
        self.summarizer = ConversationSummarizer(llm)

        # V1 keeps the current conversation summary in memory.
        # Persistent summary storage can be added later.
        self._conversation_summary = ""

    def add(
        self,
        user_message: str,
        assistant_response: str,
    ) -> list[MemoryItem]:
        """
        Process a completed conversation turn.

        Internally:
        1. stores conversation history
        2. updates the conversation summary
        3. extracts candidate memories
        4. retrieves similar global memories
        5. resolves ADD/UPDATE/DELETE/NOOP
        6. synchronizes the vector store
        """

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

        # Store raw conversation history.
        self.store.add_messages(
            chat_id=self.chat_id,
            user_id=self.user_id,
            messages=messages,
        )

        # Update conversation-specific summary.
        self._conversation_summary = self.summarizer.summarize(
            current_summary=self._conversation_summary,
            messages=messages,
        )

        # Extract durable candidate memories.
        candidates = self.extractor.extract(messages)

        updated_memories: list[MemoryItem] = []

        for candidate in candidates:
            # Find existing global memories similar to the candidate.
            similar_memories = self.retriever.search_candidate(
                candidate,
                limit=5,
            )

            # Resolve ADD / UPDATE / DELETE / NOOP.
            result = self.updater.update(
                candidate=candidate,
                similar_memories=similar_memories,
                messages=messages,
            )

            operation = result["operation"]

            if operation.value == "ADD":
                self.vector_store.add(
                    memory_id=result["id"],
                    content=result["content"],
                    metadata={
                        **candidate.metadata,
                        "user_id": self.user_id,
                    },
                )

                memory = self.store.get_memory(result["id"])
                if memory is not None:
                    updated_memories.append(memory)

            elif operation.value == "UPDATE":
                self.vector_store.update(
                    memory_id=result["id"],
                    content=result["content"],
                    metadata={
                        **candidate.metadata,
                        "user_id": self.user_id,
                    },
                )

                memory = self.store.get_memory(result["id"])
                if memory is not None:
                    updated_memories.append(memory)

            elif operation.value == "DELETE":
                self.vector_store.delete(result["id"])

        return updated_memories

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[MemoryItem]:
        """Retrieve relevant global memories for this user."""

        memories = self.retriever.search(
            query,
            limit=limit,
        )

        # V1 user scoping: filter the returned memories by user_id.
        return [
            memory
            for memory in memories
            if memory.metadata.get("user_id") == self.user_id
        ]

    def get_context(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> str:
        """Return relevant memories as prompt-ready context."""

        memories = self.search(
            query,
            limit=limit,
        )

        if not memories:
            return ""

        return "\n".join(
            f"- {memory.content}"
            for memory in memories
        )

    def clear(self, user_id: str | None = None) -> None:
        """Delete all global memories belonging to a user."""

        target_user_id = user_id or self.user_id

        memories = self.store.get_memories(target_user_id)

        for memory in memories:
            self.store.delete_memory(memory.id)
            self.vector_store.delete(memory.id)

    def delete(self, memory_id: str) -> None:
        """Delete one global memory."""

        self.store.delete_memory(memory_id)
        self.vector_store.delete(memory_id)