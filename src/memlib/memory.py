from memlib.extractor import MemoryExtractor
from memlib.retriever import MemoryRetriever
from memlib.store import MemoryStore
from memlib.summarizer import ConversationSummarizer
from memlib.types import MemoryItem, Message
from memlib.updater import MemoryUpdater
from memlib.vector_store import MemoryVectorStore


class Memory:
    def __init__(
        self,
        *,
        llm,
        user_id: str,
        chat_id: str,
        store: MemoryStore,
        vector_store: MemoryVectorStore,
    ) -> None:
        self.user_id = user_id
        self.chat_id = chat_id

        self.store = store
        self.vector_store = vector_store

        self.extractor = MemoryExtractor(llm)
        self.summarizer = ConversationSummarizer(llm)
        self.retriever = MemoryRetriever(vector_store)

        self.updater = MemoryUpdater(
            llm=llm,
            store=store,
            user_id=user_id,
        )

    def add(
        self,
        user_message: str,
        assistant_response: str,
    ) -> list[MemoryItem]:

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

        # Persist raw conversation history.
        self.store.add_messages(
            chat_id=self.chat_id,
            user_id=self.user_id,
            messages=messages,
        )

        # Load the existing persistent summary for this conversation.
        current_summary = self.store.get_summary(
            chat_id=self.chat_id,
            user_id=self.user_id,
        )

        # Update the conversation summary using the summarizer LLM.
        updated_summary = self.summarizer.summarize(
            current_summary=current_summary,
            messages=messages,
        )

        # Persist the updated conversation summary.
        self.store.save_summary(
            chat_id=self.chat_id,
            user_id=self.user_id,
            summary=updated_summary,
        )

        # Extract durable global memories from the latest turn.
        candidates = self.extractor.extract(messages)

        updated_memories: list[MemoryItem] = []

        for candidate in candidates:
            # Retrieve similar global memories belonging only to this user.
            similar_memories = self.retriever.search_candidate(
                candidate,
                user_id=self.user_id,
                limit=5,
            )

            # Ask the updater LLM for ADD / UPDATE / DELETE / NOOP.
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
                    user_id=self.user_id,
                    metadata=candidate.metadata,
                )

                memory = self.store.get_memory(result["id"])
                if memory is not None:
                    updated_memories.append(memory)

            elif operation.value == "UPDATE":
                self.vector_store.update(
                    memory_id=result["id"],
                    content=result["content"],
                    user_id=self.user_id,
                    metadata=candidate.metadata,
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

        return self.retriever.search(
            query,
            user_id=self.user_id,
            limit=limit,
        )

    def get_context(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> str:
        """Return relevant memories formatted for an agent prompt."""

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
        target_user_id = user_id or self.user_id

        memories = self.store.get_memories(target_user_id)

        for memory in memories:
            self.store.delete_memory(memory.id)
            self.vector_store.delete(memory.id)

    def delete(self, memory_id: str) -> None:
        """Delete one global memory."""

        self.store.delete_memory(memory_id)
        self.vector_store.delete(memory_id)