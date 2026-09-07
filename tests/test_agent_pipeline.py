import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from memlib.llm import LLMClient
from memlib.memory import Memory
from memlib.store import MemoryStore
from memlib.types import Message
from memlib.vector_store import MemoryVectorStore

from conftest import ScriptedChatModel


USER_ID = "pytest-user"
CHAT_ID = "pytest-chat"


@pytest.fixture
def memory(test_database_path, fake_vector_backend):
    chat_model = ScriptedChatModel(
        [
            "User likes concise Python examples.",
            '{"memories": [{"content": "User prefers concise Python examples.", "type": "preference"}]}',
            '{"operation": "ADD", "content": "User prefers concise Python examples."}',
        ]
    )

    store = MemoryStore(str(test_database_path))

    yield Memory(
        llm=LLMClient(chat_model),
        chat_model=chat_model,
        user_id=USER_ID,
        chat_id=CHAT_ID,
        store=store,
        vector_store=MemoryVectorStore(fake_vector_backend),
    )

    store.close()


def test_llm_client_uses_google_genai_compatible_message_format(monkeypatch):
    captured = {}

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def invoke(self, messages):
            captured["messages"] = messages
            return "Gemini response"

    import langchain_google_genai

    monkeypatch.setattr(
        langchain_google_genai,
        "ChatGoogleGenerativeAI",
        FakeChatGoogleGenerativeAI,
    )

    chat_model = langchain_google_genai.ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )

    response = LLMClient(chat_model).complete(
        system_prompt="You are a helpful assistant.",
        user_prompt="Remember that I prefer Python examples.",
    )

    assert response == "Gemini response"
    assert captured["init_kwargs"] == {
        "model": "gemini-2.5-flash",
        "temperature": 0,
    }
    assert [type(message) for message in captured["messages"]] == [
        SystemMessage,
        HumanMessage,
    ]
    assert captured["messages"][0].content == "You are a helpful assistant."
    assert captured["messages"][1].content == "Remember that I prefer Python examples."


def test_memory_add_persists_turn_summary_and_canonical_memory(memory, fake_vector_backend):
    memory.add(
        user_message="I prefer concise Python examples.",
        assistant_response="I will keep examples concise.",
    )

    recent_messages = memory.store.get_recent_messages(
        chat_id=CHAT_ID,
        user_id=USER_ID,
        limit=2,
    )
    memories = memory.store.get_memories(USER_ID)

    assert recent_messages == [
        Message(role="user", content="I prefer concise Python examples."),
        Message(role="assistant", content="I will keep examples concise."),
    ]
    assert memory.store.get_summary(CHAT_ID, USER_ID) == "User likes concise Python examples."
    assert [item.content for item in memories] == ["User prefers concise Python examples."]

    stored_document = next(iter(fake_vector_backend.documents_by_id.values()))
    assert stored_document.page_content == "User prefers concise Python examples."
    assert stored_document.metadata["memory_id"] == memories[0].id
    assert stored_document.metadata["user_id"] == USER_ID


def test_get_context_formats_retrieved_memories(memory):
    memory.store.add_memory(
        memory_id="memory-1",
        user_id=USER_ID,
        content="User prefers concise Python examples.",
        memory_type="preference",
    )
    memory.vector_store.add(
        memory_id="memory-1",
        user_id=USER_ID,
        content="User prefers concise Python examples.",
    )

    context = memory.get_context("How should you write examples for me?")

    assert context == "Relevant user memories:\n- User prefers concise Python examples."


def test_clear_removes_canonical_and_vector_memories(memory, fake_vector_backend):
    memory.store.add_memory(
        memory_id="memory-1",
        user_id=USER_ID,
        content="User prefers concise Python examples.",
        memory_type="preference",
    )
    memory.vector_store.add(
        memory_id="memory-1",
        user_id=USER_ID,
        content="User prefers concise Python examples.",
    )

    memory.clear()

    assert memory.store.get_memories(USER_ID) == []
    assert fake_vector_backend.documents_by_id == {}
    assert fake_vector_backend.deleted_ids == ["memory-1"]
