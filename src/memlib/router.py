from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from memlib.prompts import ROUTER_PROMPT
from memlib.types import MemoryItem, RetrievalDecision


class _RouterOutput(BaseModel):
    """Structured output expected from the routing LLM."""

    needs_graph: bool = Field(
        description=(
            "Whether additional knowledge graph retrieval is required "
            "to answer the user's query."
        )
    )

    reason: str = Field(
        description="Short explanation for the routing decision."
    )


class MemoryRouter:

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

        # Provider-independent structured output through LangChain.
        self.structured_llm = llm.with_structured_output(
            _RouterOutput
        )

    def decide(
        self,
        query: str,
        memories: list[MemoryItem],
    ) -> RetrievalDecision:

        if not query.strip():
            return RetrievalDecision(
                needs_graph=False,
                reason="Query is empty.",
            )

        formatted_memories = self._format_memories(memories)

        prompt = f"""
{ROUTER_PROMPT}

USER QUERY:
{query}

CANDIDATE MEMORIES:
{formatted_memories}
"""

        result = self.structured_llm.invoke(prompt)

        return RetrievalDecision(
            needs_graph=result.needs_graph,
            reason=result.reason,
        )

    @staticmethod
    def _format_memories(
        memories: list[MemoryItem],
    ) -> str:
        """Format memories for the routing prompt."""

        if not memories:
            return "No candidate memories were retrieved."

        return "\n".join(
            f"- [{memory.id}] {memory.content}"
            for memory in memories
        )