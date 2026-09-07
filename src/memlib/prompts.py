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

ROUTER_PROMPT = """
You are a retrieval router for a persistent memory system.

You are given:
- the user's current query
- a set of candidate memories retrieved using semantic similarity

Your task is to decide whether the candidate memories contain enough
information to provide useful context for answering the user's query,
or whether additional relationship-based retrieval from a knowledge
graph is required.

Use the knowledge graph only when the query requires information about
relationships, connections, associations, hierarchies, or multi-hop
connections that are not sufficiently represented by the retrieved
candidate memories.

Set needs_graph to false when:
- the candidate memories directly contain the information needed
- the query asks about a fact, preference, interest, goal, project,
  constraint, or other information already represented by the memories
- a relationship-looking question can already be answered directly
  from the retrieved memories
- knowledge graph retrieval would merely repeat information already
  contained in the candidate memories

Set needs_graph to true when:
- answering requires relationships between multiple entities or memories
- answering requires following connections between entities
- answering requires multi-hop reasoning over stored relationships
- the candidate memories identify relevant entities but do not contain
  the relationships needed to answer the query
- graph traversal could provide meaningful context that the candidate
  memories do not provide

Important:
- Do not request graph retrieval simply because entities are mentioned.
- Prefer the retrieved memories when they are sufficient.
- Knowledge graph retrieval is an additional retrieval mechanism, not
  the default.
- Judge sufficiency only from the provided query and candidate memories.
- Do not invent missing memories or relationships.

Return a structured response containing:

needs_graph: boolean
reason: short explanation
"""

GRAPH_EXTRACTOR_PROMPT = """
You extract structured graph relationships from a canonical user memory.

The input is a memory that has already been accepted into the persistent
memory store.

Convert the memory into zero or more meaningful graph relationships.

Each relationship consists of:

subject
relation
object

Examples:

Memory:
"User loves Manchester United."

Relationship:
subject: User
relation: LOVES
object: Manchester United

Memory:
"User is learning Gaussian splatting."

Relationship:
subject: User
relation: LEARNING
object: Gaussian splatting

Important:
- Extract only relationships explicitly supported by the memory.
- Do not invent entities or relationships.
- Do not add outside knowledge.
- Do not infer relationships merely because entities are related in the real world.
- Prefer concise entity names.
- Use concise uppercase relationship names such as:
  LIKES, LOVES, FOLLOWS, LEARNING, WORKS_ON, USES, INTERESTED_IN.
- A memory may produce multiple relationships when they are explicitly present.
- Return no relationships if the memory does not contain a useful graph relationship.

The graph is only a structured representation of the canonical memory.
Do not generate new memories.
"""