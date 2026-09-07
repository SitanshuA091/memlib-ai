from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Message:
    role: str
    content: str


@dataclass
class CandidateMemory:
    content: str
    memory_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryItem:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryUpdate:
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"]
    memory_id: str | None = None
    content: str | None = None
    memory_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalDecision:
    needs_graph: bool
    reason: str | None = None
    # needs_graph = False - candidate memories sufficient, 
    # needs_graph = True means add additional relationship/entity context

@dataclass
class GraphFact:
    subject: str
    relation: str
    object: str
    memory_id: str
    user_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # memory_id shud be same as candidate memories(sqlite and embeddings and)