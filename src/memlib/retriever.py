from memlib.store import MemoryStore
from memlib.types import CandidateMemory, MemoryItem
from memlib.vector_store import MemoryVectorStore


class MemoryRetriever:
    def __init__(
        self,
        vector_store: MemoryVectorStore,
        store: MemoryStore,
    ) -> None:
        self.vector_store = vector_store
        self.store = store

    def search(
        self,
        query: str,
        user_id: str,
        *,
        limit: int = 5,
    ) -> list[MemoryItem]:

        if not query.strip():
            return []

        documents = self.vector_store.search(
            query=query,
            user_id=user_id,
            limit=limit,
        )

        memories: list[MemoryItem] = []

        for document in documents:
            memory_id = document.metadata.get("memory_id")

            if not memory_id:
                continue

            memory = self.store.get_memory(
                str(memory_id)
            )

            if memory is not None:
                memories.append(memory)

        return memories

    def search_candidate(
        self,
        candidate: CandidateMemory,
        user_id: str,
        *,
        limit: int = 5,
    ) -> list[MemoryItem]:
        return self.search(
            query=candidate.content,
            user_id=user_id,
            limit=limit,
        )