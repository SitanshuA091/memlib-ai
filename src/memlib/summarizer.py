from memlib.llm import LLMClient
from memlib.prompts import SUMMARIZER_PROMPT
from memlib.types import Message


class ConversationSummarizer:
    """Maintains a compressed summary for a conversation."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def summarize(
        self,
        current_summary: str,
        messages: list[Message],
    ) -> str:
        """Update the conversation summary using the latest messages."""

        user_prompt = self._format_input(
            current_summary=current_summary,
            messages=messages,
        )

        return self.llm.complete(
            system_prompt=SUMMARIZER_PROMPT,
            user_prompt=user_prompt,
        ).strip()

    @staticmethod
    def _format_input(
        current_summary: str,
        messages: list[Message],
    ) -> str:
        conversation = "\n".join(
            f"{message.role}: {message.content}"
            for message in messages
        )

        return (
            f"Current summary:\n{current_summary or '(none)'}\n\n"
            f"Latest conversation:\n{conversation}"
        )