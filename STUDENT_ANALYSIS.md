# CS4603 PA4 — Document Analyst (Student Submission)

> This is your **submission file**. `README.md` is the assignment spec — this document is where you write up your work.
>
> - Document how to set up, run, and deploy your Document Analyst so a TA can reproduce your results.
> - **Answer every ANALYSIS QUESTION** from the assignment in the sections below.
> - Replace every `TODO` before submitting.
> - Keep it self-contained: a reader should be able to follow this file top-to-bottom —
>   setup → ingest → run → deploy → results — without opening the assignment spec.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your values
```

## Running locally

TODO: how to ingest the corpus, run the graph in `pa4.ipynb`, and test queries.

> **Example of the level of detail expected** (replace with your own steps/values):
>
> 1. **Ingest the corpus** (run once, from a Databricks notebook):
>    ```python
>    from rag.ingest import ingest
>    ingest(spark, volume_path="/Volumes/main/default/pa4/annual_report.pdf")
>    ```
>    This parses the PDF, chunks it into `main.default.ali_analyst_chunks`, and syncs the
>    Vector Search index `main.default.ali_analyst_index`. Wait until the index is `READY`.
>
> 2. **Build and run the graph** in `pa4.ipynb`:
>    ```python
>    from agent.graph import build_graph
>    graph = build_graph()          # uses config.py + rag/store.py + the MCP server
>    result = graph.invoke({"messages": [{"role": "user",
>              "content": "What was the net revenue in 2023?"}]})
>    print(result["messages"][-1].content)
>    ```
>
> 3. **Test queries I ran** (retrieval-only, computation-only, combined):
>    | Query | Answer produced |
>    |-------|-----------------|
>    | "What was the net income in 2023?" | ¥1.11 trillion [source: annual_report.pdf, p.4] |
>    | "What is 15% of 2.4 billion?" | 360 million |
>    | "What was 2023 revenue, and its value after 10% growth?" | ¥16.91T → ¥18.60T (16.91 × 1.10) |

## Deployment

TODO: how you logged, registered, and served the model; endpoint name; URL.

## Design decisions

TODO: graph architecture, routing, deployment choices.

### Ingestion (Part 0)
- Used ai_parse_document + ai_prep_search + Delta Sync index per the given pipeline.
- Debugged a silent schema mismatch: initial SQL guessed wrong VARIANT field paths (chunk:id, chunk:content, chunk:page_number), which resolved to NULL instead of erroring. Root-caused via DESCRIBE FUNCTION EXTENDED and inspecting raw ai_prep_search output before rewriting the explode/mapping logic against the actual nested schema (search_chunks:document.contents, chunk:chunk_id, chunk:chunk_to_retrieve, chunk:pages[0]:page_id).
- Lesson applied elsewhere: never assume a Variant/JSON schema, always inspect a sample row first.
- Verified indexed_row_count > 0 before trusting READY status, since READY only confirms the sync pipeline ran, not that any rows were embedded.

### State schema (1.1)
- Only `messages` uses a reducer (add_messages). All other fields (plan, step_results, etc.) are single-run scratch space, overwritten by whichever node owns them.
- step_results is NOT auto-appending; each node explicitly returns the extended list. Chosen for traceability over convenience.

### Planner (1.2)
- Forces JSON-only output, 2-5 steps, implicit categorization by wording (no explicit type label).
- Graceful fallback: unparseable output collapses to a single-step plan (the raw query) rather than crashing the run.
- Steps intentionally phrased so calculation steps reference prior facts by description, not by variable reference.

### Supervisor (1.3)
- Structured (single-word) LLM classification, not keyword-only, with keyword fallback if parsing fails.
- Default-to-rag_agent on ambiguous classification, since a bad RAG route degrades safely ("not found") while a bad mcp_tools route risks a hallucinated number.
- Routes to synthesizer once current_step_index >= len(plan).

### RAG agent (1.4)
- Single retriever factory (rag/store.py) reused identically by local runs and the deployed endpoint, no separate local vs. cloud retrieval path.
- Empty retrieval short-circuits before calling the LLM (cheaper, avoids hallucination around empty context).
- Citation format enforced in the extraction prompt; a lightweight guard checks for a citation tag in the output.

### MCP tools (1.5)
- MCP server loaded once at graph-build time, tool invocation is per-request but the subprocess is not respawned each call.
- Exactly one tool call enforced per step.
- Added an explicit guard: if any prior step_result is "not found" or an error, the calculation step is blocked rather than sent to the LLM, after observing a live hallucination (LLM invented a revenue figure once a prior RAG step failed).

### Synthesizer (1.6)
- Writes both final_answer and an AIMessage into messages, since the deployed endpoint's messages-in/messages-out contract only reads the last message.
- Prompted to explicitly surface partial failures rather than silently drop them.

### Graph wiring (1.7)
- Dependency injection (llm, retriever, tools all optional args) so the compiled graph can run fully offline in tests with fakes, with zero Databricks/network calls.
- Standard supervisor loop: rag_agent/mcp_tools always return to supervisor, only synthesizer exits to END.

---

## Analysis Questions

> Answer in your own words. Each question is copied from the assignment so you don't have to flip back.

### Task 1.2 — Planner
1. What happens when the planner produces steps that depend on each other (e.g., step 3 needs the result of step 1)? How does your architecture handle this?
   - The architecture handles this implicitly through step_results. Every completed step's output is appended to a shared list that stays in state for the whole run. Calculation steps are phrased by the planner to reference an earlier fact in plain language (e.g. "calculate the revenue after 3 years of 8% growth") rather than through a formal variable. When that step executes, the node reads the full step_results list as context to figure out which number to use. The weakness is there's no explicit link between a step and the specific prior step it depends on. If there were two retrieval steps, a calculation step could pull the wrong figure and nothing in the state schema would catch it.

2. Would a replanning step after each execution improve or hurt performance for this use case? Justify with an example.
   - It would help in one specific case: when a step fails or returns something unusable, like the RAG agent returning "not found in documents." Without replanning, the next step (a calculation using that missing figure) still gets attempted and likely fails or produces nonsense. A replan could catch that and insert a corrective step, like retrying retrieval with different phrasing.
   
   It would hurt in the common case. Most queries here are short, with independent facts plus one calculation, so there's little branching to adapt to. Replanning after every step adds an LLM call per iteration, increasing cost and latency, and introduces a new failure mode: the replanner itself could hallucinate a worse plan or keep adding steps indefinitely. For PA4's scope, a static plan is the better tradeoff, with conditional replanning (only on failure) as a reasonable middle ground.


### Task 1.3 — Supervisor
1. Your supervisor makes a routing decision per step. What is the failure mode if it misroutes? How would you detect and recover from a misroute?
   - If the supervisor sends a document-lookup step to mcp_tools, there's no real number to compute with, so the tool either errors out or the LLM feeding it invents a value that then gets treated as fact in step_results. If it sends a calculation step to rag_agent, the retriever searches the document for something that isn't there and usually returns "not found," silently skipping a step the plan needed. The errored case is easy to detect since it throws an exception, but the "not found on a step that should have been a calculation" case is subtler and needs a check that compares the step's wording against the outcome. Recovery would mean routing back to the supervisor to reclassify when this mismatch is detected, rather than the current design, which just appends whatever result it gets and moves on with no retry path.

2. Compare this supervisor pattern with a single ReAct agent that has access to all tools. When is the supervisor pattern worth the added complexity?
   - A single ReAct agent decides tool use turn by turn from one prompt with no separation between planning and execution, which is simpler and works fine for short single-hop tasks. The supervisor pattern earns its complexity when the task naturally splits into distinct phases like retrieval and computation that benefit from separate, narrower prompts, when you want an explicit auditable plan and per-step trace for debugging and citations, and when routing is cheap relative to letting a ReAct agent re-reason from scratch every turn. For a simple one-fact query it's not worth it, since the planning and routing overhead outweighs any reliability gain.

### Task 1.4 — RAG Agent
1. The RAG agent retrieves for a single decomposed step, not the full user query. How does this affect retrieval quality compared to retrieving for the original question?
   - Retrieving per decomposed step generally improves precision: each query is narrower and more specific (e.g. "Find Meridian's net revenue for FY2023" instead of a compound multi-part question), so the embedding is closer to the actual chunk that contains the answer, and the top-k results are less likely to be dominated by content matching only one part of a multi-part question. The tradeoff is that retrieval quality now depends entirely on how well the planner phrased that one step. If the plan decomposes a question awkwardly, retrieval for that step can do worse than retrieving for the original question would have, since the original question sometimes carries context (e.g. "revenue" alongside "growth") that a poorly split step loses.

2. If the planner produces a vague step like "find relevant financial data," how would you improve the retrieval query before sending it to the vector store?
   - A step like that is a bad retrieval query because it has no specific entity or metric embedded in it. Before sending it to the vector store, it should be rewritten into a specific query, either by re-prompting the LLM with the original user question plus the vague step and asking it to produce a concrete search query naming the actual metric or fact needed, or by having the planner itself be constrained to avoid vague steps in the first place (stricter prompt instructions, as done in Task 1.2). A lighter-weight fix is to fall back to the original full user query whenever a step's wording is judged too generic, since a vague step retrieving against the original question is likely better than retrieving against "find relevant financial data" verbatim.

### Task 2.1 — Model Definition
1. Why does `models-from-code` require a self-contained file? What breaks if you reference external state (e.g., a database running only on your laptop)?
   - MLflow's models-from-code serializes the file itself (plus whatever local packages you list in code_paths), not a pickled Python object graph. When the serving container starts, it re-executes this exact file from scratch, in a completely different machine with a fresh filesystem and no access to whatever was running on your laptop. If this file (or anything it imports) referenced a local database, a file at an absolute path on your machine, or an in-memory object built earlier in a notebook session, none of that exists inside the container. The import would either raise (file/connection not found) or silently produce broken behavior if it falls back to some default. Everything the model needs at inference time must be either bundled in the artifact (code_paths), reachable over the network with credentials passed as env vars, or a managed cloud service, which is exactly why the Vector Search index (a Databricks-managed service reachable via DATABRICKS_HOST/DATABRICKS_TOKEN) works here and a local pgvector container never could.

2. Your model calls a managed Vector Search index at inference time rather than embedding documents into the container image. What are the tradeoffs (freshness, cold-start size, latency, failure modes) of querying an external index vs. baking the corpus into the model artifact?
   - Querying the managed index keeps the model artifact small and fast to build, since no embeddings or document data ship inside the container, and it means retrieval always reflects the current state of the index. If the corpus is re-ingested or updated, every deployed model version picks up the change automatically with no redeploy needed. The cost is an extra network round trip per retrieval call (latency), and a new failure mode: if the Vector Search endpoint is down, misconfigured, or the model's env vars point at the wrong index, retrieval fails at request time even though the model itself loaded fine. Baking the corpus into the artifact (e.g. a local FAISS index shipped as a file) would remove that network dependency and failure mode, and could be faster per-query, but makes the artifact larger, ties the model version to a specific snapshot of the corpus (so updating the document requires re-logging and redeploying the model), and increases cold-start time since more data has to load into the container on startup. For PA4's scope, an external managed index is the better fit given the analyst is meant to always answer from the current document.


### Task 2.3 — Serving Endpoint
1. Why must you pass `DATABRICKS_TOKEN` as an environment variable to the endpoint, even though it's already authenticated to serve models?
   - The endpoint being "authenticated to serve models" only covers Databricks' own control plane, the mechanism that lets it load your registered model and respond to inference requests. It says nothing about what your model's code does once it's running. Your synthesizer/RAG/planner nodes make their own outbound calls, to the LLM serving endpoint (via ChatOpenAI) and to the Vector Search index (via DatabricksVectorSearch), both of which require a bearer token in the request itself. Those are calls your model code makes as a client, not something the serving infrastructure does on your model's behalf, so the token has to be available inside the container at runtime for your code to use.

2. What happens to in-flight requests when you deploy a new model version to the same endpoint? How does Databricks handle the transition?
   - Databricks Model Serving does a rolling update: it stands up a new serving container running the new model version, waits until that new version passes its own readiness checks, and only then routes new traffic to it while retiring the old version. Requests already in flight when the switch happens continue to be served by the container handling them (the old version) until they complete; new requests received after the cutover go to the new version. This is why the endpoint doesn't need a maintenance window for a redeploy, though behavior can differ briefly during the transition depending on how Databricks load-balances between old and new instances mid-swap.

### Task 3.2 — Client
1. Why is exponential backoff better than fixed-interval retries for a model serving endpoint?
   - A 429 or 503 usually means the endpoint is temporarily overloaded or mid-scale-up. Fixed-interval retries hit it again at the same short interval regardless of whether it's recovered, which piles more load onto a system that's already struggling and can make the outage worse, especially with many clients retrying in lockstep. Exponential backoff spaces retries out increasingly (1s, 2s, 4s...), giving the endpoint real time to finish scaling or clear its queue before the next attempt, and naturally staggers concurrent clients so they don't all hammer it at the same moment.

2. Your client has a `max_retries` parameter. What is the danger of setting it too high in a production system with many concurrent users?
   - If many clients each retry aggressively while an endpoint is struggling, the retries themselves become additional load, compounding the original problem instead of waiting it out. This can turn a brief scaling hiccup into a sustained overload, delay recovery, and burn through request quota or cost unnecessarily. It also increases the worst-case latency a user experiences, since a request that will ultimately fail still consumes the full backoff sequence before returning that failure.

3. When would you choose `ask_streaming()` over `ask()`? Give a concrete UX example.
   - ask_streaming() makes sense whenever the user is waiting on-screen for output and total latency for a full multi-step answer (planner, retrieval, calculation, synthesis) might take several seconds, since showing partial progress feels faster and keeps the interface visibly alive rather than looking frozen. A concrete example: a chat UI where the analyst's answer is being typed out live, the way ChatGPT-style interfaces render tokens as they arrive, so the user sees something moving within the first second instead of staring at a blank response bubble for 5-10 seconds while the whole graph runs. ask() is the better fit for backend/batch use, like a scheduled job that logs answers to a database, where nobody is watching it render in real time and the caller just needs the final string.

### Bonus A — CI/CD (if attempted)
1. Why should the deploy step only run on `main` and not on feature branches?
   - Feature branches are where work in progress lives, often incomplete, experimental, or actively broken. If every push to any branch could trigger a production deploy, a developer testing an unfinished change could accidentally overwrite the live, working endpoint that real users or the grading process depend on. Restricting deploys to main means that only code that's gone through review and merge (implicitly vetted) ever reaches production, giving you a single, predictable point of control over what's actually live.

2. What would you add to this pipeline to prevent deploying a model that performs worse than the current version? Describe the gate.
   - Add an evaluation gate between logging the model and updating the endpoint: after mlflow.langchain.log_model() registers a candidate version, run a fixed evaluation set of test queries (the same 3 from Task 1.7, plus more) against the newly logged model before it's live, and compare its answers against a stored baseline using some concrete metric, like citation presence, exact-match on known figures, an LLM-as-judge score comparing new vs. previous answers, or simply pass/fail on the offline smoke test's assertions applied to real queries instead of mocks. If the candidate scores below a set threshold relative to the current production version, the pipeline should fail the deploy step (exit non-zero) instead of calling create_or_update_endpoint(), leaving the old version serving traffic. This is essentially a CI-style regression gate, but for model quality instead of code correctness.


### Bonus B — `databricks-agents` SDK (if attempted)
1. Compare the `agents.deploy()` approach with the manual MLflow + CLI approach from Part 2. What control do you gain or lose with each?
   - The manual approach (Part 2) gives full visibility and control over every step: you choose the exact endpoint config (workload size, scale-to-zero, which env vars are secrets vs. plaintext), you control the secret scope and how credentials are injected, and you can customize the endpoint name, environment variables, and update strategy precisely. The cost is that you own every failure mode yourself, as this session's deployment debugging showed (MCP stdio subprocess issues, stderr handling, endpoint update races) — nothing is hidden, but nothing is handled for you either.
   
   agents.deploy() trades that control for convenience: one call provisions the endpoint and a Review App together, with authentication handled automatically (no secret scope to create or wire manually). This is faster to get running and adds a genuinely useful capability the manual path doesn't give you directly, structured human feedback collection via the Review App. The tradeoff is less visibility into what's actually happening under the hood; if something goes wrong, you have fewer knobs to adjust and less insight into the underlying endpoint configuration, since the SDK is making those decisions for you.

2. The Review App enables human feedback collection. How would you use this feedback to improve the agent over time? Describe a concrete feedback loop.
   - A concrete feedback loop: collect the ratings and any free-text comments from the Review App into the MLflow experiment as structured assessments, tagged by query and by which part of the pipeline (retrieval, calculation, synthesis) most plausibly caused a bad rating. Periodically export the lowest-rated interactions and manually inspect them for patterns, for example, if retrieval failures ("not found") are consistently rated poorly for queries about a topic the document actually covers, that signals the planner's step phrasing or the retriever's k value needs tuning, whereas if synthesis is rated poorly despite having correct step_results, that points to the synthesizer's prompt. These low-rated examples then become a held-out regression set: before deploying a new model version, run it against this set and require the new version's ratings (via automated LLM-as-judge scoring or a lightweight human spot-check) to meet or beat the previous version's average, feeding directly into the CI/CD quality gate described in Bonus A's second analysis question.

### Bonus C — Standalone MCP server (if attempted)
1. You moved the MCP server out of the model container. What did you gain (scaling, deployment, security, observability) and what new failure modes did you introduce (network, auth, latency, availability)?
   - TODO
2. The remote MCP server now needs its own authentication. How would you secure it so that only your serving endpoint — not the public internet — can call the tools?
   - TODO
3. When is bundling the tools in the container (Part 1) the *better* choice, and when is a separately deployed tool service (Bonus C) worth the extra moving parts?
   - TODO
