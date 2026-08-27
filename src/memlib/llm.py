from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


class LLMClient:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Execute a simple LLM completion.

        Used by:
        - memory extraction
        - memory update resolver
        - summary generation
        """

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        return self._extract_content(response)

    def invoke(self, messages: list[Any]) -> str:
        """
        Generic invoke wrapper for advanced usage.
        """

        response = self.llm.invoke(messages)

        return self._extract_content(response)

    @staticmethod
    def _extract_content(response: Any) -> str:
        """
        Normalize LangChain response output.
        """

        if hasattr(response, "content"):
            return str(response.content)

        return str(response)