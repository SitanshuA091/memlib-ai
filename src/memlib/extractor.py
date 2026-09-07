import json
from collections.abc import Sequence
from typing import Any

from memlib.llm import LLMClient
from memlib.types import CandidateMemory, Message
from memlib.prompts import EXTRACTION_SYSTEM_PROMPT


class MemoryExtractor:

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract(self, messages: Sequence[Message]) -> list[CandidateMemory]:

        pair = self._latest_pair(messages)

        if not pair:
            return []

        response = self.llm.complete(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=self._format_input(pair),
        )

        return self._parse_response(response)

    @staticmethod
    def _latest_pair(messages: Sequence[Message]) -> list[Message]:

        if len(messages) < 2:
            return []

        return list(messages[-2:])

    @staticmethod
    def _format_input(messages: Sequence[Message]) -> str:

        return json.dumps(
            {
                "user_message": messages[0].content,
                "assistant_response": messages[1].content,
            },
            ensure_ascii=True,
        )

    @staticmethod
    def _parse_response(response: str) -> list[CandidateMemory]:
        data = MemoryExtractor._load_json(response)

        if not isinstance(data, dict):
            return []

        raw_memories = data.get("memories", [])

        if not isinstance(raw_memories, list):
            return []

        candidates: list[CandidateMemory] = []

        for raw_memory in raw_memories:
            if not isinstance(raw_memory, dict):
                continue

            content = str(raw_memory.get("content", "")).strip()
            memory_type = str(raw_memory.get("type", "fact")).strip() or "fact"

            if not content:
                continue

            metadata = {
                key: value
                for key, value in raw_memory.items()
                if key != "content" and isinstance(key, str)
            }

            candidates.append(
                CandidateMemory(
                    content=content,
                    memory_type=memory_type,
                    metadata=metadata,
                )
            )

        return candidates

    @staticmethod
    def _load_json(response: str) -> Any:
        stripped = response.strip()

        if stripped.startswith("```"):
            stripped = (
                stripped.removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return {}
