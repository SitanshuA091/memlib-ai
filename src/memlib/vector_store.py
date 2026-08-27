from typing import Any

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document


class VectorStoreClient:

    def __init__(
        self,
        vector_store: VectorStore,
    ):
        self.vector_store = vector_store

    def add_memory(
        self,
        memory_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a memory and generate/store its embedding
        through the configured vector store.
        """

        document = Document(
            page_content=content,
            metadata={
                "memory_id": memory_id,
                **(metadata or {}),
            },
        )

        self.vector_store.add_documents(
            [document]
        )

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[Document]:
        return self.vector_store.similarity_search(
            query,
            k=k,
        )

    def delete(
        self,
        memory_id: str,
    ) -> None:
        if hasattr(self.vector_store, "delete"):
            self.vector_store.delete(
                ids=[memory_id]
            )