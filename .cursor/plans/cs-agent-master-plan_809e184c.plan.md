---
name: cs-agent-master-plan
status: completed
overview: "Master plan for the Bitext Customer Service Data Analyst Agent. Tasks 1, 2, 3, Bonus A, and Bonus B are all DONE. Covers architecture, repo layout, dependencies, model choices, and a per-task incremental roadmap (Tasks 1–3 + Bonuses A & B). Per-task sub-plans live alongside this file: task-1-initial-agent, task-2-memory, task-3-mcp, bonus-a-streamlit-ui, bonus-b-query-recommender."
todos:
  - id: scaffold
    content: Task 1 — Scaffold pyproject.toml, config, llm factory, and data loader (HF → parquet cache)
    status: completed
  - id: tools
    content: Task 1 — Implement tool set with Pydantic schemas (catalog / filter / summarize) + unit tests
    status: completed
  - id: router-graph
    content: Task 1 — Build router node, agent + tool nodes, decline branch, max-iterations guard, and the StateGraph
    status: completed
  - id: cli
    content: Task 1 — Build interactive CLI with rich-formatted reasoning trace and --session / --user args
    status: completed
  - id: episodic
    content: Task 2a — Add SqliteSaver checkpointer, wire thread_id from --session, verify follow-ups across restarts
    status: completed
  - id: profile
    content: Task 2b — Implement per-user JSON profile + profile_update_node + inject profile into system prompt
    status: completed
  - id: mcp
    content: Task 3 — Wrap 6 tools in FastMCP server (streamable-http) + README client example. See bonus/sub-plan task-3-mcp.
    status: completed
  - id: streamlit
    content: Bonus A — Streamlit chat UI with session-ID sidebar and reasoning-step expanders. See sub-plan bonus-a-streamlit-ui.
    status: completed
  - id: recommender
    content: Bonus B — Add recommend route + recommender_node with pending_query / confirm flow. See sub-plan bonus-b-query-recommender.
    status: completed
  - id: smoke
    content: "Final — Superseded by per-task verifiers: scripts/verify_task1.py (10 cases), verify_task2.py (2 cases), verify_task3.py (5 cases), verify_bonus_a.py (4 cases). Same intent (route + routing assertions on the assignment examples), better granularity."
    status: completed
  - id: readme
    content: Final — README ships 5-min setup, run CLI, run MCP, run Streamlit (Bonus A), run all four verifiers, model justification, mermaid architecture diagram, tools table, walkthrough flows.
    status: completed
isProject: false
---


# Customer Service Data Analyst Agent — Master Plan

This is the **master plan**. It gives you the architecture, repo layout, tech stack, and a high-level roadmap for each task. When we start a task, we'll create a focused sub-plan for it.

## 1. Big-picture architecture

```mermaid
flowchart TD
    user["User CLI / Streamlit"] --> cli["cli.py / streamlit_app.py"]
    cli -->|"thread_id = session"| graph["LangGraph StateGraph"]

    subgraph graph [Compiled graph w/ SqliteSaver checkpointer]
        router["router_node\n(small LLM,\nstructured output)"]
        decline["decline_node"]
        agent["agent_node\n(ReAct, large LLM)"]
        tools["tool_node\n(Pydantic-typed tools)"]
        recommender["recommender_node\n(Bonus B)"]
        profile["profile_update_node\n(post-turn)"]
    end

    router -->|"out_of_scope"| decline --> profile
    router -->|"recommend?"| recommender --> profile
    router -->|"structured / unstructured"| agent
    agent <--> tools
    agent --> profile
    profile --> endNode([END])

    subgraph storage [Persistence]
        sqlite["checkpoints.sqlite\n(thread state)"]
        profiles["profiles/<user>.json\n(distilled facts)"]
        parquet["data/bitext.parquet\n(HF cache)"]
    end

    graph --- sqlite
    graph --- profiles
    tools --- parquet

    subgraph mcp [MCP server (Task 3)]
        fastmcp["FastMCP server\n(streamable-http)"]
    end
    tools -. "shared impl" .- fastmcp
    fastmcp --> client["External MCP client"]
```

Three guiding principles:
- **One source of truth for tool implementations.** The CLI agent, the Streamlit UI, and the FastMCP server all import the *same* Python tool functions. MCP is just another transport.
- **Two memories, separate stores.** Episodic = LangGraph checkpoint per `thread_id`. Semantic = per-user JSON profile. Never conflate them.
- **Two LLMs by role.** Tiny model for routing/profile-distillation, larger model for reasoning + tool use + summarization.

## 2. Recommendations (with rationale)

### 2.1 Storage — pandas DataFrame in-memory, cached as parquet
The dataset is 19 MB / ~27K rows, so any storage works. I recommend in-memory pandas because:
- Tools become **plain Python functions with Pydantic schemas** — easiest to read, test, and pedagogically clearest as your first agent.
- One-time HF download cached to `data/bitext.parquet`; subsequent runs load in <100ms.
- Keeps SQLite reserved exclusively for the **checkpointer** (`checkpoints.sqlite`), so "memory state" and "domain data" never share a file — cleaner mental model.
- DuckDB would be a great alternative if we wanted SQL-feeling tools later; we can swap with minimal churn since tools are isolated behind a small `data/loader.py` module.

### 2.2 Models (Nebius Token Factory)
- **Router & profile-distillation:** `meta-llama/Meta-Llama-3.1-8B-Instruct-fast` — cheap, fast, more than enough for a 3-class classification with structured output.
- **Agent (ReAct + summarization + recommender):** `meta-llama/Llama-3.3-70B-Instruct` — strong tool-calling and reasoning, a sensible default on Nebius.
- Justification goes verbatim into the README.

### 2.3 Graph framework choice
Use a **custom `StateGraph`** rather than `create_react_agent` — the assignment requires an explicit router node, a decline branch, a profile-update step, and (Bonus B) a confirm-before-execute branch. A custom graph keeps all of those visible and gradeable.

## 3. Repository layout

```
assignment/
├── README.md                          # setup, run, model justification, MCP client snippet
├── pyproject.toml                     # deps + ruff config (uv-friendly)
├── .env / .env.example                # NEBIUS_API_KEY
├── .gitignore                         # data/, checkpoints.sqlite, profiles/
├── src/
│   └── cs_agent/
│       ├── __init__.py
│       ├── config.py                  # settings, paths, model ids
│       ├── llm.py                     # ChatOpenAI factory -> Nebius base_url
│       ├── data/
│       │   └── loader.py              # HF download + parquet cache, get_df()
│       ├── tools/
│       │   ├── schemas.py             # Pydantic input/return models
│       │   ├── catalog.py             # list_categories, list_intents, get_distribution
│       │   ├── filter.py              # count_rows, get_examples, search_by_keyword
│       │   ├── summarize.py           # summarize (LLM-backed)
│       │   └── registry.py            # exported list[Tool] used by graph + MCP
│       ├── agent/
│       │   ├── state.py               # GraphState (messages, route, pending_query, profile)
│       │   ├── prompts.py             # router prompt, agent system prompt, profile prompt
│       │   ├── router.py              # router_node + routing function
│       │   ├── nodes.py               # agent_node, decline_node, recommender_node, profile_update_node
│       │   └── graph.py               # build_graph(checkpointer)
│       ├── memory/
│       │   ├── checkpoint.py          # SqliteSaver factory
│       │   └── profile.py             # load/save/update per-user profile
│       ├── mcp_server/
│       │   └── server.py              # FastMCP wrapping >=3 tools
│       ├── cli.py                     # `python -m cs_agent.cli --session ...`
│       └── ui/
│           └── streamlit_app.py       # Bonus A
├── data/                              # gitignored (parquet cache)
├── checkpoints.sqlite                 # gitignored (LangGraph state)
├── profiles/                          # gitignored (per-user JSON)
└── tests/
    ├── test_tools.py
    └── test_router.py
```

## 4. Dependencies (pyproject.toml)

Core: `langgraph`, `langgraph-checkpoint-sqlite`, `langchain-core`, `langchain-openai` (Nebius is OpenAI-API compatible), `pydantic>=2`, `pandas`, `pyarrow`, `datasets` (HF), `python-dotenv`, `rich` (pretty CLI reasoning trace).

MCP: `fastmcp`, `langchain-mcp-adapters` (for verifying client wiring).

Bonus A: `streamlit`.

Dev: `pytest`, `ruff`.

Pin to current major versions (`langgraph>=0.2`, `pydantic>=2.7`, `fastmcp>=2`).

## 5. State shape

```python
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: Literal["structured", "unstructured", "out_of_scope", "recommend"]
    pending_query: Optional[str]   # Bonus B: a suggested-not-yet-executed query
    iterations: int                # for max-iterations guard
    user_id: str                   # for profile lookup (separate from thread_id)
```

## 6. Per-task roadmap

When you say "let's start Task X", I'll create a focused sub-plan with concrete code-level steps. Below is the high-level outline so you can see the shape.

### Task 1 — Initial agent (50 pts)
1. Scaffold `pyproject.toml`, `config.py`, `llm.py`, `data/loader.py` (one-time HF→parquet, expose `get_df()`).
2. Implement **tools** in `tools/` with Pydantic schemas. Target set:
   - `list_categories()` → `list[str]`
   - `list_intents(category: str | None)` → `list[str]`
   - `count_rows(category=None, intent=None, keyword=None)` → `int`
   - `get_examples(category=None, intent=None, keyword=None, n=5, columns=None)` → `list[dict]`
   - `get_distribution(group_by: Literal["category","intent"], scope_category=None)` → `dict[str, int]`
   - `search_by_keyword(keyword, n=10)` → `list[dict]` (handles "people wanting their money back")
   - `summarize(category=None, intent=None, role: Literal["instruction","response"], sample_size=20)` → `str`
   Tools must **not** hardcode categories/intents — read them from the dataframe so we're robust to dataset variations.
3. Build **router_node**: small LLM with `with_structured_output(RouterDecision)` → `{"route": ..., "reason": ...}`.
4. Build **agent_node** + **tool_node** (`ToolNode` from `langgraph.prebuilt`), wired in a ReAct loop with a max-iterations counter in state. On overflow → graceful fallback message.
5. Build **decline_node** for out-of-scope, with a fixed polite refusal.
6. **CLI** (`cli.py`) using `rich` to print every tool call name + args + observation, then the final answer. Interactive `while True` loop. Argparse `--session`, `--user`.
7. Smoke-test against the 8 example queries from the brief.

Deliverable: graph runs end-to-end without persistence yet.

### Task 2 — Memory (30 pts)
**2a — Episodic (20 pts)**
1. Add `memory/checkpoint.py` with `get_checkpointer(path) -> SqliteSaver` (using `SqliteSaver.from_conn_string`).
2. Compile graph with `checkpointer=checkpointer`.
3. CLI passes `config={"configurable": {"thread_id": session_id}}` to every `invoke`.
4. Verify follow-up queries: "Show me 3 examples of REFUND" → "Show me 3 more" works after a process restart.

**2b — User profile (10 pts)**
1. `memory/profile.py`: read/write `profiles/<user_id>.json` with fields like `name`, `topics_of_interest`, `preferences`, `notable_facts`.
2. Add `profile_update_node` that runs after the agent answers. It uses the small LLM with `with_structured_output(UserProfile)` to **merge** (add / update / remove) the current profile against the latest turn — explicitly *not* append-only. Atomic write back to disk.
   ```python
   class UserProfile(BaseModel):
       name: str | None = None
       role: str | None = None
       topics_of_interest: list[str] = []
       preferences: dict[str, str] = {}
       notable_facts: list[str] = []      # short, distilled, deduped
       last_updated: datetime
   ```
   The merge prompt instructs: "overwrite fields when new information refines them, remove items the user retracts, keep `notable_facts` deduped".
3. Inject the profile summary into the agent's system prompt at the start of each turn so questions like "What do you remember about me?" are answered naturally.

### Task 3 — MCP server (20 pts)
1. `mcp_server/server.py`: `FastMCP("cs-agent-tools")` with `@mcp.tool` wrappers around at minimum `list_categories`, `get_examples`, `search_by_keyword`. Reuse the same Pydantic schemas — FastMCP auto-generates the tool schema from type hints.
2. Run via `fastmcp run src/cs_agent/mcp_server/server.py:mcp --transport http --port 8765`.
3. README snippet: connect with the FastMCP `Client` and call one tool. Optionally show `MultiServerMCPClient` from `langchain-mcp-adapters` to prove they plug back into LangGraph.

### Bonus A — Streamlit UI (+10)
1. `ui/streamlit_app.py`: `st.chat_input`, `st.chat_message` rendering.
2. Sidebar `st.text_input("Session ID", value="default")` — passed as thread_id.
3. Stream graph events with `graph.stream(..., stream_mode="values")` and render tool calls / tool messages in `st.expander("Reasoning")` blocks above each assistant response.

### Bonus B — Query recommender (+10)
1. Add a router class `recommend` for prompts like "what should I query next?".
2. `recommender_node` reads `messages` + profile, asks the agent LLM for a single suggestion, writes it to `state["pending_query"]`, and returns a message asking the user to confirm or refine.
3. On the next turn, if `pending_query` is set and the user confirms, the graph routes directly to `agent_node` with the suggested query injected as the human turn; if they refine, regenerate; if they reject, clear `pending_query`.

## 7. Working sequence (recommended)

1. Scaffolding + data loader (Task 1, step 1).
2. Tools + tests (Task 1, step 2) — foundation everything else stands on.
3. Router + agent loop + CLI (Task 1, steps 3–6) — get a working agent end-to-end without memory.
4. Episodic memory (Task 2a) — small wrap, biggest UX win.
5. User profile (Task 2b).
6. MCP server (Task 3) — straightforward once tools are stable.
7. Streamlit UI (Bonus A).
8. Recommender (Bonus B).
9. README polish + final smoke test against all example queries.

## 8. Submission package

- **Solo submission.** Repo / zip name: `customer-support-agent_ofir_nesher`.
- Provide both options to the grader: a GitHub URL *and* a downloadable zip (just zip the cleaned-up working tree, excluding `data/`, `.venv/`, `checkpoints.sqlite`, `profiles/`).

### README structure (the grader's 5-minute path)

1. **TL;DR** — one paragraph on what the agent does.
2. **Setup** — `git clone`, `uv sync` (or `pip install -e .`), copy `.env.example` to `.env`, paste `NEBIUS_API_KEY`.
3. **Run the CLI** — `python -m cs_agent.cli --session demo --user ofir`, with a sample transcript showing the reasoning trace.
4. **Run the MCP server** — start command + a 10-line FastMCP client snippet calling `list_categories`.
5. **Run the Streamlit UI** — `streamlit run src/cs_agent/ui/streamlit_app.py`.
6. **Run the smoke test** — `bash scripts/smoke.sh` (runs the 8 example queries end-to-end against a fresh session and prints PASS/FAIL).
7. **Architecture overview** — the mermaid diagram from this plan plus a 3-bullet description.
8. **Model choices** — Llama 3.1 8B (router + profile) and Llama 3.3 70B (agent + summarization), with the *why*.
9. **Tools reference table** — name, signature, description.
10. **Memory model** — episodic vs profile, where each is stored.
11. **Project layout** — the file tree.
12. **Limitations & future work**.

### Smoke-test script

`scripts/smoke.sh` runs the 8 brief examples (e.g. "What categories exist?", "How many refund requests?", "Who is the president of France?") against a fresh `--session smoke_<timestamp>`, captures the agent's output, asserts each one routes correctly (structured / unstructured / out-of-scope) and that out-of-scope queries are politely declined. Useful for self-checking and grader-friendly.

## 9. Out of scope for this plan

- Vector search / embeddings (not needed; keyword `str.contains` over instructions is sufficient and grader-friendly).
- Postgres checkpointer (Sqlite is enough for the assignment's persistence requirement).
- Production deployment / Dockerization.
