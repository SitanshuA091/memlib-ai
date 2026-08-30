### Built-in prompts used by memlib LLM components.
EXTRACTION_SYSTEM_PROMPT = """
You extract durable user memories from a recent conversation turn.

Return only JSON with this exact shape:
{"memories":[{"content":"...", "type":"fact|preference|goal"}]}

Extract only information explicitly stated or clearly expressed by the USER.
Never treat information introduced, suggested, assumed, or inferred by the ASSISTANT as a user fact.

Extract information worth remembering for future conversations, including:
- stable facts explicitly stated by the user
- user preferences
- ongoing goals or projects
- persistent interests or working context

Do not extract:
- assistant claims or suggestions
- temporary requests or one-off task details
- information that is uncertain or merely implied
- sensitive secrets such as passwords, API keys, or credentials
- conversational filler

Use concise third-person statements.
Avoid creating multiple memories that express essentially the same fact.
If nothing is worth remembering, return {"memories":[]}."""

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

SUMMARIZER_PROMPT = """
You maintain a concise summary of a conversation.

You are given:
- the existing conversation summary
- the latest conversation messages

Update the summary to preserve the important context needed for future turns.

Keep the summary concise and focused on:
- ongoing tasks and projects
- decisions and conclusions
- important preferences expressed during the conversation
- relevant constraints or requirements
- important unresolved context

Do not include:
- unnecessary conversational details
- repetitive information
- temporary or irrelevant details
- information that was only mentioned by the assistant without user confirmation

Return only the updated summary as plain text.
"""