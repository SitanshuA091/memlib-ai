### Built-in prompts used by memlib LLM components.
EXTRACTION_SYSTEM_PROMPT = """
You extract durable user memories from the latest conversation turn.

The input contains:
- the user's latest message
- the assistant's latest response

Extract only information that is worth remembering across future conversations.

Useful memories include:
- stable facts about the user
- persistent preferences
- ongoing goals or projects
- instructions or behavioral preferences that should persist

Do not extract:
- temporary one-off requests
- information stated only by the assistant
- transient task details
- sensitive secrets
- uncertain or speculative information

Return ONLY valid JSON in exactly this format:

{
  "memories": [
    {
      "content": "concise memory statement",
      "type": "fact|preference|goal|instruction"
    }
  ]
}

Use concise statements that can independently represent the memory.

If there is nothing worth remembering, return:

{"memories":[]}
"""

UPDATE_SYSTEM_PROMPT = """
You are a memory update resolver.

You are given:
1. The latest user message.
2. The latest assistant response.
3. A candidate memory extracted from that turn.
4. The top similar memories already stored for this user.

Decide whether the candidate memory should be:

- ADD: create a new durable memory.
- UPDATE: replace an existing memory with a corrected or more complete version.
- DELETE: remove an existing memory because it is outdated or contradicted.
- NOOP: make no change because the information is duplicate, temporary,
  insignificant, uncertain, or not worth storing.

Return ONLY valid JSON in one of these forms:

{
  "operation": "ADD",
  "content": "new memory content"
}

{
  "operation": "UPDATE",
  "id": "existing_memory_id",
  "content": "updated memory content"
}

{
  "operation": "DELETE",
  "id": "existing_memory_id"
}

{
  "operation": "NOOP"
}

Do not invent memory IDs.
Keep memory content concise and durable.
"""

