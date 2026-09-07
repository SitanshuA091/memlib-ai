from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel

from memlib.extractor import MemoryExtractor
from memlib.graph_extractor import GraphExtractor
from memlib.graph_store import GraphStore
from memlib.retriever import MemoryRetriever
from memlib.router import MemoryRouter
from memlib.store import MemoryStore
from memlib.summarizer import MemorySummarizer
from memlib.types import MemoryItem, Message
from memlib.updater import MemoryUpdater
from memlib.vector_store import MemoryVectorStore


class Memory:
    """
    Main public interface for persistent memory.

    Memory hides:
    - extraction
    - update resolution
    - SQLite persistence
    - vector retrieval
    - knowledge graph extraction/storage
    - retrieval routing
    - conversation summarization
    """

    def __init__(
        self,
        llm: BaseChatModel,
        user_id: str,
        chat_id: str,
        store: MemoryStore,
        vector_store: MemoryVectorStore,
        graph_store: GraphStore | None = None,
    ) -> None:
        self.llm = llm
        self.user_id = user_id
        self.chat_id = chat_id

        self.store = store
        self.vector_store = vector_store
        self.graph_store = graph_store

        self.extractor = MemoryExtractor(llm)
        self.updater = MemoryUpdater(llm)
        self.summarizer = MemorySummarizer(llm)

        self.retriever = MemoryRetriever(
            vector_store=vector_store,
            store=store,
        )

        # Graph functionality is optional.
        self.router = (
            MemoryRouter(llm)
            if graph_store is not None
            else None
        )

        self.graph_extractor = (
            GraphExtractor(llm)
            if graph_store is not None
            else None
        )

    def add(
        self,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """
        Process a completed conversation turn.

        The turn is:
        1. stored in conversation history
        2. summarized
        3. inspected for durable memory candidates
        4. resolved against existing memories
        5. synchronized across SQLite, vector storage and graph storage
        """

        self._store_messages(
            user_message=user_message,
            assistant_response=assistant_response,
        )

        self._update_summary(
            user_message=user_message,
            assistant_response=assistant_response,
        )

        candidates = self.extractor.extract(
            user_message=user_message,
            assistant_response=assistant_response,
        )

        for candidate in candidates:
            existing_memories = self.retriever.search_candidate(
                candidate,
                user_id=self.user_id,
                limit=5,
            )

            decision = self.updater.decide(
                candidate=candidate,
                existing_memories=existing_memories,
            )

            action = decision.action.upper()

            if action == "ADD":
                self._add_memory(decision)

            elif action == "UPDATE":
                self._update_memory(decision)

            elif action == "DELETE":
                self._delete_memory_from_decision(decision)

            elif action == "NOOP":
                continue

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[MemoryItem]:
        """
        Retrieve semantically relevant canonical memories.

        This remains the normal memory-search API and does not
        automatically expose graph implementation details.
        """

        return self.retriever.search(
            query=query,
            user_id=self.user_id,
            limit=limit,
        )

    def get_context(
        self,
        query: str,
        *,
        limit: int = 5,
        graph_limit: int = 10,
    ) -> str:
        """
        Build prompt-ready memory context.

        Vector retrieval happens first.

        When graph support exists, the router determines whether the
        retrieved canonical memories are sufficient. Graph retrieval
        is only performed when additional relational context is useful.
        """

        memories = self.search(
            query,
            limit=limit,
        )

        graph_context: list[dict] = []

        if (
            self.graph_store is not None
            and self.router is not None
        ):
            decision = self.router.decide(
                query=query,
                memories=memories,
            )

            if decision.needs_graph:
                graph_context = self.graph_store.search(
                    query=query,
                    user_id=self.user_id,
                    limit=graph_limit,
                )

        return self._format_context(
            memories=memories,
            graph_context=graph_context,
        )

    def delete(self, memory_id: str) -> None:
        """
        Delete a canonical memory and all derived representations.
        """

        self.store.delete_memory(memory_id)

        self.vector_store.delete(memory_id)

        if self.graph_store is not None:
            self.graph_store.delete_memory(memory_id)

    def clear(self) -> None:
        """
        Remove all global memories belonging to the current user.
        """

        memories = self.store.get_memories(
            self.user_id
        )

        for memory in memories:
            self.vector_store.delete(memory.id)

        if self.graph_store is not None:
            self.graph_store.clear_user(
                self.user_id
            )

        for memory in memories:
            self.store.delete_memory(memory.id)

    def _add_memory(self, decision) -> None:
        """Persist a new canonical memory and derived representations."""

        memory_id = str(uuid4())

        content = decision.content

        if not content:
            return

        memory_type = (
            decision.memory_type
            or "fact"
        )

        metadata = decision.metadata or {}

        # 1. Canonical source of truth.
        self.store.add_memory(
            memory_id=memory_id,
            user_id=self.user_id,
            content=content,
            memory_type=memory_type,
            metadata=metadata,
        )

        # 2. Derived semantic index.
        self.vector_store.add(
            memory_id=memory_id,
            user_id=self.user_id,
            content=content,
            metadata=metadata,
        )

        # 3. Derived graph representation.
        memory = MemoryItem(
            id=memory_id,
            content=content,
            metadata=metadata,
        )

        self._sync_graph(memory)

    def _update_memory(self, decision) -> None:
        """
        Update canonical memory and rebuild its derived representations.
        """

        memory_id = decision.memory_id
        content = decision.content

        if not memory_id or not content:
            return

        metadata = decision.metadata or {}

        # 1. Update SQLite canonical memory.
        self.store.update_memory(
            memory_id=memory_id,
            content=content,
            metadata=metadata,
        )

        # 2. Replace vector representation.
        self.vector_store.update(
            memory_id=memory_id,
            user_id=self.user_id,
            content=content,
            metadata=metadata,
        )

        # 3. Remove old graph relationships.
        if self.graph_store is not None:
            self.graph_store.delete_memory(
                memory_id
            )

        # 4. Re-extract graph representation from new memory.
        memory = MemoryItem(
            id=memory_id,
            content=content,
            metadata=metadata,
        )

        self._sync_graph(memory)

    def _delete_memory_from_decision(
        self,
        decision,
    ) -> None:
        """Apply updater DELETE decision."""

        if not decision.memory_id:
            return

        self.delete(
            decision.memory_id
        )

    def _sync_graph(
        self,
        memory: MemoryItem,
    ) -> None:
        """
        Derive graph relationships from a canonical memory.

        Neo4j remains a derived index and never becomes the canonical
        memory store.
        """

        if (
            self.graph_store is None
            or self.graph_extractor is None
        ):
            return

        facts = self.graph_extractor.extract(
            memory=memory,
            user_id=self.user_id,
        )

        for fact in facts:
            self.graph_store.add_fact(
                fact
            )

    def _store_messages(
        self,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Persist the raw conversation turn."""

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

        self.store.add_messages(
            chat_id=self.chat_id,
            user_id=self.user_id,
            messages=messages,
        )

    def _update_summary(
        self,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Update the persistent per-conversation summary."""

        existing_summary = self.store.get_summary(
            chat_id=self.chat_id,
            user_id=self.user_id,
        )

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

        summary = self.summarizer.summarize(
            existing_summary=existing_summary or "",
            messages=messages,
        )

        self.store.save_summary(
            chat_id=self.chat_id,
            user_id=self.user_id,
            summary=summary,
        )

    @staticmethod
    def _format_context(
        memories: list[MemoryItem],
        graph_context: list[dict],
    ) -> str:
        """Format retrieved memory and graph evidence for the LLM."""

        sections: list[str] = []

        if memories:
            memory_lines = [
                f"- {memory.content}"
                for memory in memories
            ]

            sections.append(
                "Relevant user memories:\n"
                + "\n".join(memory_lines)
            )

        if graph_context:
            graph_lines = [
                (
                    f"- {fact['subject']} "
                    f"--{fact['relation']}--> "
                    f"{fact['object']}"
                )
                for fact in graph_context
            ]

            sections.append(
                "Relevant memory relationships:\n"
                + "\n".join(graph_lines)
            )

        if not sections:
            return ""

        return "\n\n".join(sections)