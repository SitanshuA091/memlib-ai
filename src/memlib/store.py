
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from memlib.types import MemoryItem, Message


class MemoryStore:
    """SQLite-backed source of truth for conversation history and memories."""

    def __init__(self, database: str = "memlib.db") -> None:
        self.connection = sqlite3.connect(database)
        self.connection.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        self.connection.commit()

    def add_messages(
        self,
        chat_id: str,
        user_id: str,
        messages: list[Message],
    ) -> None:
        """Store conversation messages in chronological order."""

        timestamp = datetime.now(timezone.utc).isoformat()

        self.connection.executemany(
            """
            INSERT INTO messages
            (chat_id, user_id, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    chat_id,
                    user_id,
                    message.role,
                    message.content,
                    timestamp,
                )
                for message in messages
            ],
        )

        self.connection.commit()

    def get_recent_messages(
        self,
        chat_id: str,
        limit: int = 2,
    ) -> list[Message]:
        """Return the most recent conversation messages."""

        rows = self.connection.execute(
            """
            SELECT role, content
            FROM messages
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        ).fetchall()

        rows.reverse()

        return [
            Message(
                role=row["role"],
                content=row["content"],
            )
            for row in rows
        ]

    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a durable global memory."""

        now = datetime.now(timezone.utc).isoformat()

        self.connection.execute(
            """
            INSERT INTO memories
            (id, user_id, content, type, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                user_id,
                content,
                memory_type,
                json.dumps(metadata or {}),
                now,
                now,
            ),
        )

        self.connection.commit()

    def update_memory(
        self,
        memory_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update an existing global memory while preserving its ID."""

        now = datetime.now(timezone.utc).isoformat()

        self.connection.execute(
            """
            UPDATE memories
            SET content = ?,
                metadata = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                content,
                json.dumps(metadata or {}),
                now,
                memory_id,
            ),
        )

        self.connection.commit()

    def delete_memory(self, memory_id: str) -> None:
        """Delete a global memory by its canonical ID."""

        self.connection.execute(
            """
            DELETE FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        )

        self.connection.commit()

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        """Return a single global memory by ID."""

        row = self.connection.execute(
            """
            SELECT id, content, metadata
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()

        if row is None:
            return None

        return MemoryItem(
            id=row["id"],
            content=row["content"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def get_memories(
        self,
        user_id: str,
    ) -> list[MemoryItem]:
        """Return all global memories belonging to a user."""

        rows = self.connection.execute(
            """
            SELECT id, content, metadata
            FROM memories
            WHERE user_id = ?
            ORDER BY created_at ASC
            """,
            (user_id,),
        ).fetchall()

        return [
            MemoryItem(
                id=row["id"],
                content=row["content"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def close(self) -> None:
        """Close the SQLite connection."""

        self.connection.close()