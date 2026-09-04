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

Update the summary to preserve only the most important context needed for future turns.

Keep the summary very short: ideally 1-3 sentences and only a few lines.

Prioritize:
- ongoing tasks or projects
- important decisions or conclusions
- explicit user preferences
- relevant constraints or requirements
- important questions or unresolved context

Important:
- Distinguish between information the user explicitly stated and information they merely asked about.
- A question, request for information, or mention of a topic does not by itself indicate that the user likes, prefers, believes in, uses, or is interested in that topic.
- Do not infer user interests, preferences, goals, or beliefs from the subject of their questions.
- Do not turn assistant suggestions, assumptions, or claims into user facts.

Avoid:
- detailed explanations
- examples or background information
- repetitive information
- temporary or trivial details
- information that does not help continue the conversation
- unnecessary details from the assistant's response

Preserve important user context even when the latest turn is unrelated.

Return only the updated summary as plain text.
"""