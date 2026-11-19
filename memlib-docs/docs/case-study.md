# **Case Study**: Building memlib

## The problem

Agent frameworks are improving fast; memory layers for them are mostly monolithic. Pick Mem0, you get Mem0's provider list and Mem0's vector store. Pick LangMem, same story. For an application that already has opinions about which LLM provider and which vector database it uses — or needs different ones in different environments — a memory library that hard-codes both is friction, not help.

memlib's starting constraint was: **the memory layer should adapt to the application's stack, not the other way around.** That constraint shaped every subsequent decision.

## Decision 1 — SQLite as the only canonical store

Three storage technologies are involved (SQLite, a vector store, optionally Neo4j), and only one of them can be the source of truth without creating consistency problems. SQLite was the obvious pick: it's always available, needs no external service, and — critically — it's the one layer that *isn't* being made swappable. The vector store and graph change per deployment; the canonical record shouldn't.

Every derived record (vector embedding, graph relationship) carries the SQLite `memory_id` as metadata. Retrieval always resolves back through that ID rather than trusting a copy of the content sitting in the index. The payoff shows up immediately in `MemoryRetriever.search()`: it pulls `memory_id` out of vector-store metadata and fetches the real content from SQLite, so a vector backend that ever returned stale or partially-written text would still never surface outdated data to the application.

## Decision 2 — Model-adaptive via LangChain's `BaseChatModel`

Rather than wrapping OpenAI's SDK or Anthropic's SDK directly, memlib's `LLMClient` accepts anything implementing LangChain's `BaseChatModel`. This was a trade: it couples memlib to LangChain's abstraction rather than being framework-free, but in exchange every provider LangChain already supports — Groq, OpenAI, Gemini, Anthropic, Mistral — works with zero memlib-side code.

It also enabled a decision that wouldn't have been possible wrapping one provider's SDK: letting the *agent's* model and *memlib's internal* model differ. A LangGraph agent can run on `gpt-5-mini` for user-facing responses while memlib extracts and resolves memories on a cheaper Groq model — the two `LLMClient`/`chat_model` pairs passed into `Memory` never have to match.

## Decision 3 — Storage-adaptive via LangChain's `VectorStore`

Same reasoning, applied to vector storage. `MemoryVectorStore` is intentionally thin — about 40 lines — because the goal wasn't to build a vector store abstraction; LangChain already has one. The only value memlib adds on top is tagging every document with `memory_id` and `user_id`, which is the minimum needed to make retrieval resolvable and multi-tenant.

The `update()` method is implemented as `delete()` then `add()` rather than an in-place update, because not every `VectorStore` backend supports one uniformly — accepting the two-call overhead kept the abstraction backend-agnostic instead of leaking per-backend capability checks into memlib.

## Decision 4 — The knowledge graph is a derived index, not a second source of truth

Adding Neo4j support could easily have turned into a second canonical store, with memlib having to reconcile two sources of truth on every write. Instead, graph facts are explicitly one-directional: canonical memory → `GraphExtractor` → graph relationships. An `UPDATE` or `DELETE` on a memory deletes that memory's graph relationships outright and (for `UPDATE`) re-extracts them — there's no attempt to diff or patch existing edges, because the graph is cheap to regenerate from the memory that produced it.

This also makes the graph genuinely optional. `Memory.__init__` only constructs a `MemoryRouter` and `GraphExtractor` when a `graph_store` is passed — an application that skips the graph pays zero overhead for it, not even a null check inside `get_context()`'s hot path beyond one `if`.

## Decision 5 — Routing instead of always querying the graph

Once a graph store exists, the tempting default is to query it on every `get_context()` call. That adds latency and cost to every turn, including the many where vector retrieval alone already answers the query. `MemoryRouter` is a small structured-output LLM call that looks at the query and the memories already retrieved, and decides whether relational context would add anything — Neo4j is only queried when the answer is yes.

This is a clear trade-off: it adds one more LLM call to the critical path precisely to *save* a database round trip, betting that the LLM call is cheaper and faster than an unconditional graph query most of the time. For applications where that assumption doesn't hold, the router is replaceable because `MemoryRouter` only depends on `BaseChatModel` and the existing retrieved memories.

## Decision 6 — Per-candidate resolution, not per-turn

A single conversation turn can produce several candidate memories. Resolving all of them together against one combined similarity search risked one candidate's neighbors crowding out another's, or an unrelated existing memory absorbing an edit meant for something else. Each candidate instead gets its own top-5 similarity search and its own independent `ADD`/`UPDATE`/`DELETE`/`NOOP` decision from the updater LLM. It costs more LLM calls per turn, but keeps every resolution well-scoped to exactly the memory it concerns.

## Known limitations (V1 scope, not oversights)

- **Graph retrieval is keyword matching, not semantic.** `Neo4jGraphStore.search` reduces a query to lowercase non-stopword terms and matches them against entity/relation names with `CONTAINS`. No entity linking, no embedding of graph nodes. This keeps the graph layer dependency-free and predictable, at the cost of missing paraphrased queries (e.g. "club" won't match an entity named "team").
- **No cross-store consistency checker.** If a process crashes mid-`UPDATE`, it's possible (though narrow) for SQLite to be updated while the vector or graph write hasn't completed. There's no background reconciliation job that walks all three stores and repairs drift.
- **`update_memory` doesn't fail without a prior retrieval.** `MemoryUpdater` is only ever invoked with candidates the extractor produced, so there's no public-API guard against calling `delete()`/`update_memory()` with a `memory_id` that was never this `user_id`'s — the `store`/`vector_store`/`graph_store` layers trust their caller.
- **Summarization is append-only context, not a token-budget-aware compressor.** `ConversationSummarizer` folds new messages into the existing summary on every turn; there's no mechanism yet to cap summary length or re-compress it as it grows.

## What would change in a V2

The README itself names the natural next steps: richer entity extraction for the graph, configurable relationship ontologies (right now every relation name is whatever the extraction LLM uppercases), more advanced retrieval strategies beyond top-k cosine similarity, and automated consistency checks across the three storage layers. None of these require breaking the public `Memory` API — which was itself a design goal: `get_context()` and `add()` are stable enough that the internals underneath them can keep evolving.
