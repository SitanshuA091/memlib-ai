import json
import sqlite3
import uuid
from collections.abc import Sequence
from typing import Any

from memlib.llm import LLMClient
from memlib.types import CandidateMemory, MemoryItem, MemoryOperation, Message
from memlib.prompts import UPDATE_SYSTEM_PROMPT



class MemoryUpdater:

    def __init__(
        self,
        llm: LLMClient,
        connection: sqlite3.Connection,
    ) -> None:
        self.llm = llm
        self.connection = connection
        self.user_id = self.user_id
        self._ensure_table()

    def update(
        self,
        candidate: CandidateMemory,
        similar_memories: Sequence[MemoryItem],
        messages: Sequence[Message],
    ) -> MemoryOperation:
        """Resolve and apply ADD, UPDATE, DELETE, or NOOP."""

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
            self._add_memory(
                content=str(decision.get("content", "")).strip(),
                metadata=candidate.metadata,
            )
            return MemoryOperation.ADD

        if operation == MemoryOperation.UPDATE:
            memory_id = str(decision.get("id", "")).strip()
            content = str(decision.get("content", "")).strip()

            if memory_id and content:
                self._update_memory(
                    memory_id=memory_id,
                    content=content,
                    metadata=candidate.metadata,
                )
                return MemoryOperation.UPDATE

        if operation == MemoryOperation.DELETE:
            memory_id = str(decision.get("id", "")).strip()

            if memory_id:
                self._delete_memory(memory_id)
                return MemoryOperation.DELETE

        return MemoryOperation.NOOP

    def _ensure_table(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS global_memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                type TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.commit()

    def _add_memory(
        self,
        content: str,
        metadata: dict[str, object],
    ) -> str | None:
        if not content:
            return None

        memory_id = str(uuid.uuid4())

        memory_type = str(metadata.get("type", "fact"))

        self.connection.execute(
            """
            INSERT INTO global_memories
            (memory_id, user_id, content, type, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                self.user_id,
                content,
                memory_type,
                json.dumps(metadata),
            ),
        )
        self.connection.commit()

        return memory_id

    def _update_memory(
        self,
        memory_id: str,
        content: str,
        metadata: dict[str, object],
    ) -> None:
        self.connection.execute(
            """
            UPDATE global_memories
            SET content = ?,
                metadata = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE memory_id = ?
            """,
            (
                content,
                json.dumps(metadata),
                memory_id,
            ),
        )
        self.connection.commit()

    def _delete_memory(self, memory_id: str) -> None:
        self.connection.execute(
            """
            DELETE FROM global_memories
            WHERE memory_id = ?
            """,
            (memory_id,),
        )
        self.connection.commit()

    @staticmethod
    def _format_input(
        candidate: CandidateMemory,
        similar_memories: Sequence[MemoryItem],
        messages: Sequence[Message],
    ) -> str:
        return json.dumps(
            {
                "user_message": next(
                    (
                        message.content
                        for message in messages
                        if message.role == "user"
                    ),
                    "",
                ),
                "assistant_response": next(
                    (
                        message.content
                        for message in reversed(messages)
                        if message.role == "assistant"
                    ),
                    "",
                ),
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

        if operation not in {
            MemoryOperation.ADD,
            MemoryOperation.UPDATE,
            MemoryOperation.DELETE,
            MemoryOperation.NOOP,
        }:
            return {}

        return data