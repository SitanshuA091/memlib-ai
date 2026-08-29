"""Shared data models used across memlib."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


Role = Literal["user", "assistant"]


@dataclass(slots=True)
class Message:
    """A conversation message."""

    role: Role
    content: str


@dataclass(slots=True)
class CandidateMemory:
    """A durable memory candidate extracted from a conversation turn."""

    content: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryItem:
    """A persisted global user memory."""

    id: str
    content: str
    metadata: dict[str, object] = field(default_factory=dict)


class MemoryOperation(StrEnum):
    """Operations the update resolver can perform on global memory."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"