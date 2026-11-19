# **Knowledge Graph**

The knowledge graph is entirely optional. If `graph_store` is `None`, `Memory` never builds a `MemoryRouter` or `GraphExtractor`, and `get_context()` skips graph retrieval without any extra branching in application code.

## Enabling it

```python
from memlib.graph_store import Neo4jGraphStore

graph_store = Neo4jGraphStore(
    uri="neo4j+s://...",
    username="neo4j",
    password="your-password",
    database="your-database",
)

memory = Memory(
    llm=llm,
    chat_model=chat_model,
    user_id="user-123",
    chat_id="chat-456",
    store=store,
    vector_store=vector_store,
    graph_store=graph_store,
)
```

The call site doesn't change — `memory.get_context(query)` is the same call whether or not a graph is attached.

## From memory to graph facts

`GraphExtractor` turns a finalized canonical memory into subject–relation–object triples using structured LLM output:

```text
"User loves Manchester United."
    ↓
User --LOVES--> Manchester United
```

Each extracted `GraphFact` carries the originating `memory_id` and `user_id`, so relationships are always traceable back to the memory that produced them — and removable as a set when that memory changes or is deleted.

## Storage model

`Neo4jGraphStore.add_fact` runs a single `MERGE` per fact:

```cypher
MERGE (subject:Entity {user_id: $user_id, name: $subject})
MERGE (object:Entity {user_id: $user_id, name: $object})
MERGE (subject)-[r:RELATES {
    memory_id: $memory_id,
    relation: $relation,
    user_id: $user_id
}]->(object)
SET r.metadata = $metadata
```

Entities are `MERGE`d rather than created outright, so the same entity (e.g. "Manchester United") is reused across memories instead of duplicated — while each relationship edge still carries its own `memory_id`, so deleting one memory's facts never touches another memory's edges between the same two entities.

## Retrieval: keyword matching, by design

Graph search is intentionally simple in V1. A query like *"Which football club do I love?"* is reduced to lowercase, non-stopword terms (`football`, `club`, `love`) and matched against entity and relation names with `CONTAINS`:

```cypher
MATCH (s:Entity)-[r:RELATES]->(o:Entity)
WHERE r.user_id = $user_id
  AND any(term IN $search_terms WHERE
        toLower(s.name) CONTAINS term
     OR toLower(o.name) CONTAINS term
     OR toLower(r.relation) CONTAINS term)
RETURN s.name, r.relation, o.name, r.memory_id, r.metadata
LIMIT $limit
```

This trades retrieval sophistication for predictability and zero extra infrastructure (no entity-linking model, no embedding of graph nodes). It's a deliberate V1 scope boundary, not an oversight — see [Case Study](case-study.md) for the trade-off discussion.

## The router: when is the graph worth querying

`MemoryRouter` is a structured-output LLM call that looks at the query and the memories already retrieved from the vector store, and decides `needs_graph: bool`. Graph search only runs when the router says the vector-retrieved memories aren't sufficient on their own — so a graph store being configured doesn't mean every `get_context()` call pays for a Neo4j round trip.

## Keeping the graph in sync

- **On `UPDATE`**: all graph relationships tagged with the memory's `memory_id` are deleted, then re-extracted from the new content.
- **On `DELETE`**: the same relationships are deleted outright.
- **On `clear()`**: `graph_store.clear_user(user_id)` detaches and deletes every `Entity` node scoped to that user.

The graph never accumulates stale edges from superseded memories.
