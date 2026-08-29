from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class MemoryVectorStore:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def add(
        self,
        memory_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a global memory embedding using its SQLite memory_id."""

        document = Document(
            page_content=content,
            metadata={
                **(metadata or {}),
                "memory_id": memory_id,
            },
        )

        self.vector_store.add_documents(
            documents=[document],
            ids=[memory_id],
        )

    def update(
        self,
        memory_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Replace the embedding for an existing global memory."""

        self.delete(memory_id)

        self.add(
            memory_id=memory_id,
            content=content,
            metadata=metadata,
        )

    def delete(self, memory_id: str) -> None:
        """Delete the embedding associated with a global memory_id."""

        self.vector_store.delete(ids=[memory_id])

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[Document]:
        """Retrieve the most semantically similar global memories."""

        return self.vector_store.similarity_search(
            query,
            k=limit,
        )