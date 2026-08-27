## **Mem0 Paper Notes**
LLMs have a fixed context window which poses fundamental challenges for maintaining consistency over prolonged multi session conos, Mem0 introduces scaleble memory centric architecture addressing the issue by dynamically extracting, consolidating and retrieving info from ongoing convos.  

- In same convo if a user mentions his dietary plans and diner plans for some time and then go to some coding taska dn then back to food, a full context based approach(retreieving info from entire chat session context) woul need to reason through mountains of irrelevant information.
- simply presenting longer contexts does not ensure retrieval or utilization of past info, attention mechansims also degrade over tokens
- Mem0 extends standard memory storage of managing salient info from convos by incorporating graph based representations where memories are stored as directed labelled graphs with entities as nodes and relationships as edges.

## Mem0 Basic pipeline
Has mainly 2 phases Extraction phase and update phase
### **Extraction Phase** -
- initiates upon ingestion of new message pair(mt-1, mt) mt is current and mt-1 is a preceeding one. pair is a user message and a assistant response.
- the extraction system employs 2 complementary sources convo summary **S** retrieved from the DB which encapsulates semantic context of entire convo history 
- sequence of recent messages from convo history where m is hyperparameter controlling recency of window
- there is asynchronous summory generation module which periodically refreshes convo summary. this component is independent of the main processing pipeline.
- the recent message sequence has more finer granular details with time defined memories too which is independent of the global memory summary which is S
- this dual contextual info along with the new message pair forms a comprehensive prompt P = {S, {mt-m,..mt-2}, mt-1, mt} for extraction function θ (theta) implemented via an LLM(sent to LLM to extract memories)
-  **θ**(P) theta then extracts a set of salient memories **Ω**(omega) = {w1, w2...wn} based on LLM response for the prompt 

### **Update Phase** 
- It evaluates ecah candidate phase against existing memories.
- the phases determines the appropriate memory management operation for each extracted fact w1 belonging to Ω
- for each fact system first retrieves the top **s** semantically similar memories using vector embeddings from the db.
- retrieved memories along with the candidate fact are presented to the LLM through a functon calling interface we refer as tool call
- LLM decides wether to do a ADD (creation of new memory since no significantly semantically similar memories), UPDATE of existing memory with new complementary memory. DELETE for removal of memories contradicted by new info. and NOOP (no operation) if candidate fact requires no modification.
- LLM's reasoning abilities are used rather than a seperate classifier.
- In essence when the relation generator from the extraction phase is fed to the update phase it goes through a conflict resolver which selects existing nodes and for the operation and in a update resolver component updates the memory graphs.
- generally working params are m = 10, s = 10 similar memories

**Memories stored in Mem0g**
- memories are represented as a directed labeled graph G = (V, E, L) 
- V is entities (Name, Places Nouns)
- E is Edges represnting relationships b/w entities (example `LIVES_IN`)
- L is Labels assign semantic types (ALICE -`Person` DELHI - `Place`)
- So each entity node v belonging ton V has 
 - an entity type classification.
 - embedding vector that captures entity's semantic meaning 
 - metadata such time of creation tv, Relationships are stored in the form of triplets (vs, r, vd) which mean how entity vs is related to vd like (AlICE, `LIVES_IN`, WONDERLAND).

 ---
- The extraction phase has an **entity extractor** module which processes input text to identify a set of entities along with corresponding types. 
- entity extractor in Mem0 analyzes semantic importance , uniqueness and persistence too.
- in a convo about travel plans entities might be the destinations, transportation modes, dates, activities etc
---
- **Relationship generator** is an LLM based module that capture semantically significant connections, it examines linguistic patterns, contextual cues and domain knowledge
- this module employs prompt engineering techniques that guide the LLM to reason about both explicit statements and implicit info in the Dialogue. 
- finally produces that relationship triplet {vs,R,vd} R might be LIVES_IN or BORN_ON.
- for every new relationship we compute embeddings for both vs and vd and then search for existing nodes with semantic similarity above a threshold 't'.
- there is also an **conflict detection** mechanism which identifies potentially conflicting existing relationships.
- another LLM based **update resolver** determines if certain relationships should be obsolete or invalid.
----

### Memory Retrieval Functionality
- it implements a dual approach for optimal info access
- entity centric method first identifies key entities within a query and leverages semantic similarity to locate corresponding nodes in the knowledge graph.
- this explore both incoming and ougoing relationships from anchor nodes constructing a comprehensive subgraph which captures relevant contextual info
- entire query is matched against textual encodings of each relationship triplet and it calculates fine grained similarity scores b/w query and all available triplets returning those above some relevance threshold ranked with decreasing similarity.    


the system uses Neo4j as underlying graph DB, LLM based extractors, update modules leveraged GPT-4o-Mini with tool calling capabilities 

### EXPERIMENTATION IMP POINTS
LOCOMO Dataset used for experiment setup for dialogue systems.comprises 600 dialogues and 26000 tokens on avg

### Evaluation Metrics
**Performance metrics** 
BLEU like scores would score high for Aice born in july with alice was born in march due to lexical overlap so we use LLM as a Judge (**J**), judge model asseses the question, ground truth answer and the generated answer, providing a nuanced evaluation aligning better with Human Judgemenet

**Deployment Metrics**
- Deployment metrics refer to real world constraints for agent applications so **Token Consumption** using `c100k_base` encoding measuring no. of tokens extracted during retrieval that serve as context for answering queries which in memory's context represent memories retrieval from knowledge base
- Latency is another metric:- **search latency**:  total time reqd to search memory, **total_latency**: time to generate appropriate responses consisting both retrieval time and answer generation.
- For Benchmarking k is used as 1 (only the single most relevant chunk is selected)
- Consistent Embedding model is used when any sorta similarity search is reqd and anything needs to be converted to embedding vectors.
- Performance metrics must include F1 score, B1(BLEU) and LLM as Judge (J).
- In Evaluation Single Hop Question means involve locating a single factual span contained within one dialogue turn
- Multi Hop Queries require synthesizing info dispersed across multiple conversation sessions