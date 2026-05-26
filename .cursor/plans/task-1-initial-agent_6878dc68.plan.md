---
name: task-1-initial-agent
overview: "Focused, code-level sub-plan for Task 1 (50 pts) — scaffold, data loader, Pydantic-typed tool set, router, ReAct agent loop with max-iterations guard, decline branch, and a `rich`-formatted CLI. Target: end-to-end working agent without persistence (Task 2 adds memory)."
todos:
  - id: t1-scaffold
    content: Step 1 — pyproject.toml (uv), package skeleton, .env.example, .gitignore, config.py, llm.py with two Nebius LLM factories
    status: completed
  - id: t1-loader
    content: Step 2 — data/loader.py with HF download + parquet cache + lru_cached get_df() + dataset_summary()
    status: completed
  - id: t1-tools-schemas
    content: Step 3a — tools/schemas.py with one Pydantic input model per tool, with rich Field descriptions
    status: completed
  - id: t1-tools-impl
    content: Step 3b — tools/{catalog.py, filter.py, summarize.py} implementing 7 tools with @tool + thorough docstrings
    status: completed
  - id: t1-tools-registry
    content: Step 3c — tools/registry.py exporting DATA_TOOLS list
    status: completed
  - id: t1-tools-tests
    content: Step 3d — tests/test_tools.py covering catalog/filter/search happy paths on a fixture DataFrame
    status: completed
  - id: t1-router
    content: Step 4 — agent/state.py, agent/prompts.py, agent/router.py with RouterDecision structured output
    status: completed
  - id: t1-graph
    content: Step 5 — agent/nodes.py (agent_node, decline_node, should_continue) and agent/graph.py (build_graph) with max-iterations guard
    status: completed
  - id: t1-cli
    content: Step 6 — cli.py with rich-formatted reasoning trace, --session and --user args, interactive REPL
    status: completed
  - id: t1-verify
    content: Step 7 — Run the 8 example queries, confirm routing, multi-step chaining, decline behavior, and max-iterations fallback
    status: completed
isProject: false
---


# Task 1 — Initial Agent (sub-plan)

## Goal

A single command (`python -m cs_agent.cli`) drops the user into an interactive loop. They ask a question. The router classifies it. If out-of-scope → polite decline. Otherwise → the ReAct loop selects tools, executes them on the Bitext DataFrame, and returns an answer. Every reasoning step is printed via `rich`. Max 12 iterations, then graceful fallback. No persistence yet.

## Grading map (50 pts)

- Router (15) — `agent/router.py` + structured-output schema
- Tools w/ Pydantic schemas + clear descriptions (15) — `tools/`
- Multi-step reasoning (10) — emerges from ReAct loop + tool granularity choice below
- CLI w/ reasoning output (5) — `cli.py` with `rich`
- Max-iterations fallback (5) — `iterations` counter in `GraphState`

## Locked decisions (from previous turn)

- **Package manager:** `uv`. Single `pyproject.toml`, dev workflow is `uv sync` then `uv run python -m cs_agent.cli`.
- **Tool granularity:** atomic functions with optional filter args (e.g. `count_rows(category=, intent=, keyword=)`). Multi-step reasoning emerges naturally when the agent doesn't know the exact values yet (e.g. `list_intents("REFUND")` → `count_rows(intent="track_refund")`).

## Step-by-step (estimated 4–5 hrs total)

### Step 1 — Project scaffold (~30 min)

Create the package skeleton.

**`pyproject.toml`** (one canonical version, `uv` reads it natively):

```toml
[project]
name = "cs-agent"
version = "0.1.0"
description = "Customer Service Data Analyst Agent (Bitext) — LangGraph ReAct + FastMCP"
requires-python = ">=3.11"
dependencies = [
  "langgraph>=0.2.50",
  "langgraph-checkpoint-sqlite>=2.0",
  "langchain-core>=0.3",
  "langchain-openai>=0.2",
  "pydantic>=2.7",
  "pandas>=2.2",
  "pyarrow>=15",
  "datasets>=2.20",
  "python-dotenv>=1.0",
  "rich>=13.7",
]

[project.optional-dependencies]
mcp = ["fastmcp>=2.0", "langchain-mcp-adapters>=0.2"]
ui  = ["streamlit>=1.36"]
dev = ["pytest>=8", "ruff>=0.6"]

[project.scripts]
cs-agent = "cs_agent.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/cs_agent"]

[tool.ruff]
target-version = "py311"
line-length = 110
```

**Files to create (all skeletons):**
- `src/cs_agent/__init__.py`
- `src/cs_agent/config.py` — `BITEXT_HF_ID`, `DATA_DIR`, `PARQUET_PATH`, `CHECKPOINT_PATH`, `PROFILES_DIR`, `ROUTER_MODEL`, `AGENT_MODEL`, `MAX_ITERATIONS = 12`. Loads `.env` with `python-dotenv`.
- `src/cs_agent/llm.py` — two factories:
  ```python
  def get_router_llm() -> ChatOpenAI: ...
  def get_agent_llm() -> ChatOpenAI: ...
  ```
  Both point at `base_url="https://api.studio.nebius.com/v1/"` (Nebius is OpenAI-compatible) with `api_key=os.environ["NEBIUS_API_KEY"]`. Cached with `functools.lru_cache`.
- `.env.example` with `NEBIUS_API_KEY=` placeholder.
- `.gitignore` adding `data/`, `checkpoints.sqlite`, `profiles/`, `.venv/`, `__pycache__/`.

### Step 2 — Data loader (~20 min)

**`src/cs_agent/data/loader.py`**

```python
@lru_cache(maxsize=1)
def get_df() -> pd.DataFrame:
    """Load the Bitext dataset, downloading + caching to local parquet on first call."""
    if PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH)
    ds = load_dataset(BITEXT_HF_ID, split="train")
    df = ds.to_pandas()
    df["category"] = df["category"].str.upper()
    df["intent"] = df["intent"].str.lower()
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)
    return df
```

Also export `dataset_summary()` returning `{n_rows, categories, intents_per_category}` — used by the agent's system prompt so the LLM knows the schema.

### Step 3 — Tools (~1.5 hr)

The most important step. Tool docstrings are graded as much as the logic. **Every tool reads from `get_df()`; never hardcode categories/intents.**

**`src/cs_agent/tools/schemas.py`** — Pydantic input models, one per tool. Example:

```python
class CountRowsArgs(BaseModel):
    """Filters for counting rows in the Bitext customer-service dataset."""
    category: str | None = Field(
        None,
        description="Optional. High-level semantic category, e.g. 'REFUND', 'ACCOUNT'. Case-insensitive.",
    )
    intent: str | None = Field(
        None,
        description="Optional. Specific intent within a category, e.g. 'track_refund', 'create_account'.",
    )
    keyword: str | None = Field(
        None,
        description="Optional. Case-insensitive substring matched against the user 'instruction' column.",
    )
```

**`src/cs_agent/tools/catalog.py`** — three discovery tools:

```python
@tool("list_categories", args_schema=NoArgs)
def list_categories() -> list[str]:
    """Return all distinct high-level categories that exist in the dataset.
    Use this FIRST when the user asks 'what categories exist' or before filtering
    by category if you are not sure of the exact spelling."""
    return sorted(get_df()["category"].unique().tolist())

@tool("list_intents", args_schema=ListIntentsArgs)
def list_intents(category: str | None = None) -> list[str]:
    """Return all intents in the dataset, optionally scoped to a single category.
    Use this when the user asks about a category and you need to know which intents
    belong to it (e.g. before counting refund requests, list intents in 'REFUND')."""
    df = get_df()
    if category:
        df = df[df["category"] == category.upper()]
    return sorted(df["intent"].unique().tolist())

@tool("get_distribution", args_schema=DistributionArgs)
def get_distribution(group_by: Literal["category", "intent"],
                     scope_category: str | None = None) -> dict[str, int]:
    """Return the row-count distribution grouped by category OR by intent.
    Optionally scope to a single category (useful for 'distribution of intents
    in the ACCOUNT category')."""
    ...
```

**`src/cs_agent/tools/filter.py`** — three filter/aggregate tools:

- `count_rows(category=, intent=, keyword=) -> int`
- `get_examples(category=, intent=, keyword=, n=5, columns=None) -> list[dict]` — returns ≤ `n` rows with the requested columns (default: `instruction`, `intent`, `category`).
- `search_by_keyword(keyword: str, n: int = 10) -> list[dict]` — purely keyword-based search over `instruction`. This is what answers "people wanting their money back".

**`src/cs_agent/tools/summarize.py`** — the only LLM-backed tool:

```python
@tool("summarize", args_schema=SummarizeArgs)
def summarize(category: str | None = None, intent: str | None = None,
              role: Literal["instruction", "response"] = "response",
              sample_size: int = 20) -> str:
    """Summarize patterns in the dataset for a category and/or intent.
    Set role='response' to summarize how customer service representatives reply,
    or role='instruction' to summarize what users ask. Samples up to `sample_size`
    rows. Use this for open-ended/unstructured questions."""
    rows = _filtered(category, intent).sample(min(sample_size, ...))
    text = "\n---\n".join(rows[role].tolist())
    llm = get_agent_llm()
    return llm.invoke(SUMMARIZE_PROMPT.format(role=role, text=text)).content
```

**`src/cs_agent/tools/registry.py`**
```python
DATA_TOOLS: list[BaseTool] = [
    list_categories, list_intents, get_distribution,
    count_rows, get_examples, search_by_keyword, summarize,
]
```

**Tests — `tests/test_tools.py`** — happy-path tests on a tiny fixture DataFrame (monkey-patch `get_df`):
- categories returned sorted & unique
- `count_rows(category="REFUND")` matches expected count on fixture
- `search_by_keyword("money back")` returns rows whose instruction contains the phrase
- `get_examples(intent="track_refund", n=3)` returns ≤ 3 rows with the right intent

### Step 4 — Router (~45 min)

**`src/cs_agent/agent/state.py`**

```python
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: Literal["structured", "unstructured", "out_of_scope"] | None
    iterations: int
    user_id: str
```

**`src/cs_agent/agent/prompts.py`** — three string constants:
- `ROUTER_SYSTEM` — explains the three classes, gives 2 examples each from the brief, instructs to return JSON only.
- `AGENT_SYSTEM` — the ReAct system message. Includes:
  - Dataset description (column names + a snapshot of categories/intents from `dataset_summary()`).
  - Tool list with the encouragement to chain (e.g. `list_categories` → `get_examples`) when unsure of exact spellings.
  - Instruction to use `summarize` only for unstructured/open-ended questions.
  - **Scoped-fallback paragraph** (handles "on-topic but no tool fits"):
    ```
    If the user's question is on-topic for the Bitext customer-service dataset
    but cannot be answered with the tools you have, do NOT guess and do NOT
    invent values. Reply briefly with: (1) what you cannot answer and why
    ("I don't have a sentiment tool"), and (2) the closest thing you CAN do,
    framed as a concrete tool call you would run.
    ```
- `SUMMARIZE_PROMPT` — used inside `summarize` tool.

**`src/cs_agent/agent/router.py`**

```python
class RouterDecision(BaseModel):
    route: Literal["structured", "unstructured", "out_of_scope"]
    reason: str

def router_node(state: GraphState) -> dict:
    last = state["messages"][-1].content
    decision = (
        get_router_llm()
        .with_structured_output(RouterDecision)
        .invoke([SystemMessage(ROUTER_SYSTEM), HumanMessage(last)])
    )
    return {"route": decision.route}

def route_from_router(state: GraphState) -> Literal["agent", "decline"]:
    return "decline" if state["route"] == "out_of_scope" else "agent"
```

### Step 5 — Agent loop + graph (~1 hr)

**`src/cs_agent/agent/nodes.py`**

```python
def agent_node(state: GraphState) -> dict:
    if state["iterations"] >= MAX_ITERATIONS:
        msg = AIMessage(
            "I couldn't solve this within my reasoning budget. "
            "Could you rephrase or simplify?"
        )
        return {"messages": [msg], "iterations": state["iterations"] + 1}
    llm = get_agent_llm().bind_tools(DATA_TOOLS)
    sys = build_agent_system(state)   # injects dataset summary
    response = llm.invoke([SystemMessage(sys), *state["messages"]])
    return {"messages": [response], "iterations": state["iterations"] + 1}

def decline_node(state: GraphState) -> dict:
    return {"messages": [AIMessage(
        "That's outside the scope of this customer-service dataset agent. "
        "I can help with questions about the Bitext dataset (categories, intents, "
        "examples, distributions, summaries). Try one of those?"
    )]}

def should_continue(state: GraphState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if state["iterations"] >= MAX_ITERATIONS:
        return "end"
    return "tools" if (isinstance(last, AIMessage) and last.tool_calls) else "end"
```

**`src/cs_agent/agent/graph.py`**

```python
def build_graph(checkpointer=None):
    g = StateGraph(GraphState)
    g.add_node("router", router_node)
    g.add_node("agent",  agent_node)
    g.add_node("tools",  ToolNode(DATA_TOOLS))
    g.add_node("decline", decline_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_from_router,
                            {"agent": "agent", "decline": "decline"})
    g.add_conditional_edges("agent", should_continue,
                            {"tools": "tools", "end": END})
    g.add_edge("tools", "agent")
    g.add_edge("decline", END)

    return g.compile(checkpointer=checkpointer)
```

Note: `iterations` is reset *per turn* in Task 1 (no checkpointer). Task 2a will require resetting on each new human input — we'll handle that then.

### Step 6 — CLI with `rich` reasoning trace (~30 min)

**`src/cs_agent/cli.py`**

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="default")
    parser.add_argument("--user", default="anon")
    args = parser.parse_args()

    console = Console()
    graph = build_graph()
    console.print(Panel.fit("Customer Service Data Analyst — type 'exit' to quit",
                            title="cs-agent"))

    while True:
        q = Prompt.ask("[bold cyan]you[/]")
        if q.strip().lower() in {"exit", "quit"}: break

        initial = {"messages": [HumanMessage(q)], "iterations": 0,
                   "user_id": args.user, "route": None}
        for chunk in graph.stream(initial, stream_mode="updates"):
            _render(chunk, console)   # pretty-print router decision, tool calls, observations
```

`_render` handles the four interesting cases:
1. `router` update → print routing decision in dim text.
2. `agent` update with `tool_calls` → print `→ tool_name(args)` in yellow.
3. `tools` update → print `← observation: <truncated>` in green inside an expander-like Panel.
4. `agent` update with final content → print as the assistant's answer in a bold Panel.

### Step 7 — Verification (~30 min)

Run the agent against the 8 example queries from the brief:

```
What categories exist in the dataset?                       → structured, list_categories
How many refund requests did we get?                        → structured, list_intents("REFUND")→count_rows
Show me 5 examples of the SHIPPING category.                → structured, get_examples (note: real category is SHIPPING_ADDRESS — agent should discover via list_categories)
Summarize how agents respond to complaint intents.          → unstructured, summarize(intent="complaint", role="response")
Show me examples of people wanting their money back.        → structured/keyword, search_by_keyword("money back")
What is the distribution of intents in the ACCOUNT category? → structured, get_distribution(group_by="intent", scope_category="ACCOUNT")
What's the best CRM software for handling complaints?       → out_of_scope, decline
Who is the president of France?                             → out_of_scope, decline
```

Confirm reasoning trace shows multi-step chaining for at least the refund and SHIPPING examples. Capture any failures into a notes section we'll polish in the README.

## Three failure modes (handled separately on purpose)

| Situation | Where it's caught | Response |
|---|---|---|
| Off-topic ("Who won the 2024 UCL?") | `router_node` → `decline_node` | Polite decline, suggest in-scope queries |
| On-topic but no tool fits ("Average instruction length?") | `AGENT_SYSTEM` scoped-fallback paragraph | Honest "I can't do X, but I can do Y" |
| On-topic, agent kept trying, ran out | `iterations >= MAX_ITERATIONS` in `agent_node` | "Couldn't solve in budget, please rephrase" |

We deliberately do **not** add an `execute_pandas_query` escape hatch — it would dilute the "tools with Pydantic schemas" grading focus and add prompt-injection / fabrication risk.

## Files this task creates

```
assignment/
├── pyproject.toml
├── .env.example
├── .gitignore
└── src/cs_agent/
    ├── __init__.py
    ├── config.py
    ├── llm.py
    ├── data/{__init__.py, loader.py}
    ├── tools/{__init__.py, schemas.py, catalog.py, filter.py, summarize.py, registry.py}
    ├── agent/{__init__.py, state.py, prompts.py, router.py, nodes.py, graph.py}
    └── cli.py
tests/
└── test_tools.py
```

## Acceptance criteria (Task 1 done)

- `uv sync` then `uv run cs-agent` opens the REPL.
- All 8 example queries return reasonable answers.
- At least one query (refund or SHIPPING) shows ≥ 2 tool calls in the trace.
- Out-of-scope queries are declined politely without making up answers.
- The 13th iteration on a hostile prompt yields the fallback message, not an infinite loop (will verify with a deliberately ambiguous query like "tell me everything").
- `pytest` passes on `tests/test_tools.py`.

## Out of scope (Task 1 only)

- Persistent state (Task 2a).
- User profile (Task 2b).
- MCP server (Task 3).
- Streamlit, recommender (Bonuses).
