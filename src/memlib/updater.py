import json
import uuid
from collections.abc import Sequence
from typing import Any

from memlib.llm import LLMClient
from memlib.store import MemoryStore
from memlib.types import CandidateMemory, MemoryItem, MemoryUpdate, Message
from memlib.prompts import UPDATE_SYSTEM_PROMPT


class MemoryUpdater:

    def __init__(
        self,
        llm: LLMClient,
        store: MemoryStore,
        user_id: str,
    ) -> None:
        self.llm = llm
        self.store = store
        self.user_id = user_id

    def decide(
        self,
        candidate: CandidateMemory,
        existing_memories: Sequence[MemoryItem],
        messages: Sequence[Message],
    ) -> MemoryUpdate:
        response = self.llm.complete(
            system_prompt=UPDATE_SYSTEM_PROMPT,
            user_prompt=self._format_input(
                candidate=candidate,
                similar_memories=existing_memories,
                messages=messages,
            ),
        )

        decision = self._parse_response(response)
        operation = decision.get("operation")

        if operation == "ADD":
            content = str(decision.get("content", "")).strip()

            if not content:
                return MemoryUpdate(action="NOOP")

            return MemoryUpdate(
                action="ADD",
                content=content,
                memory_type=candidate.memory_type,
                metadata=candidate.metadata,
            )

        if operation == "UPDATE":
            memory_id = str(decision.get("id", "")).strip()
            content = str(decision.get("content", "")).strip()

            if not memory_id or not content:
                return MemoryUpdate(action="NOOP")

            return MemoryUpdate(
                action="UPDATE",
                memory_id=memory_id,
                content=content,
                memory_type=candidate.memory_type,
                metadata=candidate.metadata,
            )

        if operation == "DELETE":
            memory_id = str(decision.get("id", "")).strip()

            if not memory_id:
                return MemoryUpdate(action="NOOP")

            return MemoryUpdate(
                action="DELETE",
                memory_id=memory_id,
            )

        return MemoryUpdate(action="NOOP")

    def update(
        self,
        candidate: CandidateMemory,
        similar_memories: Sequence[MemoryItem],
        messages: Sequence[Message],
    ) -> dict[str, Any]:
        decision = self.decide(
            candidate=candidate,
            existing_memories=similar_memories,
            messages=messages,
        )

        if decision.action == "ADD":
            memory_id = str(uuid.uuid4())

            self.store.add_memory(
                memory_id=memory_id,
                user_id=self.user_id,
                content=decision.content or "",
                memory_type=decision.memory_type or "fact",
                metadata=decision.metadata,
            )

            return {
                "operation": "ADD",
                "id": memory_id,
                "content": decision.content,
            }

        if decision.action == "UPDATE" and decision.memory_id:
            self.store.update_memory(
                memory_id=decision.memory_id,
                content=decision.content or "",
                metadata=decision.metadata,
            )

            return {
                "operation": "UPDATE",
                "id": decision.memory_id,
                "content": decision.content,
            }

        if decision.action == "DELETE" and decision.memory_id:
            self.store.delete_memory(decision.memory_id)

            return {
                "operation": "DELETE",
                "id": decision.memory_id,
            }

        return {"operation": "NOOP"}

    @staticmethod
    def _format_input(
        candidate: CandidateMemory,
        similar_memories: Sequence[MemoryItem],
        messages: Sequence[Message],
    ) -> str:
        user_message = next(
            (
                message.content
                for message in reversed(messages)
                if message.role == "user"
            ),
            "",
        )

        assistant_response = next(
            (
                message.content
                for message in reversed(messages)
                if message.role == "assistant"
            ),
            "",
        )

        return json.dumps(
            {
                "user_message": user_message,
                "assistant_response": assistant_response,
                "candidate_memory": {
                    "content": candidate.content,
                    "metadata": candidate.metadata,
                },
                "similar_memories": [
                    {
                        "id": memory.id,
                        "content": memory.content,
                        "metadata": memory.metadata,
                    }
                    for memory in similar_memories
                ],
            },
            ensure_ascii=True,
        )

    @staticmethod
    def _parse_response(response: str) -> dict[str, Any]:

        stripped = response.strip()

        if stripped.startswith("```"):
            stripped = (
                stripped.removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return {}

        if not isinstance(data, dict):
            return {}

        operation = data.get("operation")

        if operation not in {"ADD", "UPDATE", "DELETE", "NOOP"}:
            return {}

        return data
