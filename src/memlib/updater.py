"""LLM-based global memory update resolver."""

import json
import uuid
from collections.abc import Sequence
from typing import Any

from memlib.llm import LLMClient
from memlib.store import MemoryStore
from memlib.types import CandidateMemory, MemoryItem, MemoryOperation, Message
from memlib.prompts import UPDATE_SYSTEM_PROMPT


class MemoryUpdater:
    """Resolves candidate memories and applies the decision to SQLite."""

    def __init__(
        self,
        llm: LLMClient,
        store: MemoryStore,
        user_id: str,
    ) -> None:
        self.llm = llm
        self.store = store
        self.user_id = user_id

    def update(
        self,
        candidate: CandidateMemory,
        similar_memories: Sequence[MemoryItem],
        messages: Sequence[Message],
    ) -> dict[str, Any]:
        """
        Resolve a candidate memory and apply the SQLite operation.

        Returns the operation and affected memory information so the caller
        can synchronize the vector store.
        """

        response = self.llm.complete(
            system_prompt=UPDATE_SYSTEM_PROMPT,
            user_prompt=self._format_input(
                candidate=candidate,
                similar_memories=similar_memories,
                messages=messages,
            ),
        )

        decision = self._parse_response(response)
        operation = decision.get("operation")

        if operation == MemoryOperation.ADD:
            content = str(decision.get("content", "")).strip()

            if not content:
                return {"operation": MemoryOperation.NOOP}

            memory_id = str(uuid.uuid4())

            self.store.add_memory(
                memory_id=memory_id,
                user_id=self.user_id,
                content=content,
                memory_type=str(
                    candidate.metadata.get("type", "fact")
                ),
                metadata=candidate.metadata,
            )

            return {
                "operation": MemoryOperation.ADD,
                "id": memory_id,
                "content": content,
            }

        if operation == MemoryOperation.UPDATE:
            memory_id = str(decision.get("id", "")).strip()
            content = str(decision.get("content", "")).strip()

            if not memory_id or not content:
                return {"operation": MemoryOperation.NOOP}

            self.store.update_memory(
                memory_id=memory_id,
                content=content,
                metadata=candidate.metadata,
            )

            return {
                "operation": MemoryOperation.UPDATE,
                "id": memory_id,
                "content": content,
            }

        if operation == MemoryOperation.DELETE:
            memory_id = str(decision.get("id", "")).strip()

            if not memory_id:
                return {"operation": MemoryOperation.NOOP}

            self.store.delete_memory(memory_id)

            return {
                "operation": MemoryOperation.DELETE,
                "id": memory_id,
            }

        return {"operation": MemoryOperation.NOOP}

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
        """Safely parse the JSON returned by the LLM."""

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

        if operation not in {
            MemoryOperation.ADD,
            MemoryOperation.UPDATE,
            MemoryOperation.DELETE,
            MemoryOperation.NOOP,
        }:
            return {}

        return data