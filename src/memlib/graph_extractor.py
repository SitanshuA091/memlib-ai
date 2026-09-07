from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from memlib.prompts import GRAPH_EXTRACTOR_PROMPT
from memlib.types import GraphFact, MemoryItem


class _ExtractedRelationship(BaseModel):
    """A single relationship extracted by the LLM."""

    subject: str = Field(
        description="Source entity of the relationship."
    )

    relation: str = Field(
        description="Concise uppercase relationship name."
    )

    object: str = Field(
        description="Target entity of the relationship."
    )


class _GraphExtractionOutput(BaseModel):
    """Structured output returned by the graph extraction LLM."""

    relationships: list[_ExtractedRelationship] = Field(
        default_factory=list,
        description="Relationships explicitly supported by the memory.",
    )


class GraphExtractor:
    """
    Extract graph relationships from canonical memories.

    The extractor accepts any LangChain BaseChatModel implementation,
    keeping the component independent of the LLM provider.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

        self.structured_llm = llm.with_structured_output(
            _GraphExtractionOutput
        )

    def extract(
        self,
        memory: MemoryItem,
        user_id: str,
    ) -> list[GraphFact]:
        """
        Convert a canonical memory into graph relationships.

        Every GraphFact retains the original memory_id so graph
        relationships can always be traced back to SQLite.
        """

        if not memory.content.strip():
            return []

        prompt = f"""
{GRAPH_EXTRACTOR_PROMPT}

CANONICAL MEMORY:
{memory.content}
"""

        result = self.structured_llm.invoke(prompt)

        facts: list[GraphFact] = []

        for relationship in result.relationships:
            subject = relationship.subject.strip()
            relation = relationship.relation.strip().upper()
            object_ = relationship.object.strip()

            if not subject or not relation or not object_:
                continue

            facts.append(
                GraphFact(
                    subject=subject,
                    relation=relation,
                    object=object_,
                    memory_id=memory.id,
                    user_id=user_id,
                    metadata={
                        "source": "global_memory",
                    },
                )
            )

        return facts