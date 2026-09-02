from collections.abc import Sequence

from langchain_core.documents import Document

from memlib.types import CandidateMemory, MemoryItem
from memlib.vector_store import MemoryVectorStore


class MemoryRetriever:
    """Retrieves semantically similar global memories."""

    def __init__(
        self,
        vector_store: MemoryVectorStore,
    ) -> None:
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        user_id: str,
        *,
        limit: int = 5,
    ) -> list[MemoryItem]:
        """
        Search global memories for a specific user using semantic similarity.

        Returns memories with the same IDs used by the SQLite
        global memories table.
        """

        if not query.strip():
            return []

        documents = self.vector_store.search(
            query,
            user_id=user_id,
            limit=limit,
        )

        return [
            self._document_to_memory(document)
            for document in documents
        ]

    def search_candidate(
        self,
        candidate: CandidateMemory,
        user_id: str,
        *,
        limit: int = 5,
    ) -> list[MemoryItem]:
        """Find existing global memories similar to a candidate for a user."""

        return self.search(
            candidate.content,
            user_id=user_id,
            limit=limit,
        )

    @staticmethod
    def _document_to_memory(document: Document) -> MemoryItem:
        """Convert a vector-store document into a MemoryItem."""

        memory_id = str(
            document.metadata.get("memory_id", "")
        )

        metadata = {
            key: value
            for key, value in document.metadata.items()
            if key != "memory_id"
        }

        return MemoryItem(
            id=memory_id,
            content=document.page_content,
            metadata=metadata,
        )