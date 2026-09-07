import sys
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def pytest_ignore_collect(collection_path, config):
    return collection_path.name == "test_memory_flow.py"


@pytest.fixture(autouse=True)
def no_real_api_keys(monkeypatch):
    for key in (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
        "NEO4J_DATABASE",
    ):
        monkeypatch.delenv(key, raising=False)


class ScriptedChatModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)

        if not self.responses:
            raise AssertionError("ScriptedChatModel received an unexpected invoke call.")

        return AIMessage(content=self.responses.pop(0))


class FakeVectorStoreBackend:
    def __init__(self):
        self.documents_by_id = {}
        self.deleted_ids = []

    def add_documents(self, documents, ids):
        for document, document_id in zip(documents, ids, strict=True):
            self.documents_by_id[document_id] = document

    def delete(self, ids):
        for document_id in ids:
            self.deleted_ids.append(document_id)
            self.documents_by_id.pop(document_id, None)

    def similarity_search(self, query, k=5, filter=None):
        user_id = (filter or {}).get("user_id")
        matches = [
            document
            for document in self.documents_by_id.values()
            if user_id is None or document.metadata.get("user_id") == user_id
        ]
        return matches[:k]


@pytest.fixture
def fake_vector_backend():
    return FakeVectorStoreBackend()


@pytest.fixture
def test_database_path():
    temp_dir = PROJECT_ROOT / "tests" / ".pytest-tmp"
    temp_dir.mkdir(exist_ok=True)
    database_path = temp_dir / f"{uuid4()}.db"

    yield database_path

    database_path.unlink(missing_ok=True)
