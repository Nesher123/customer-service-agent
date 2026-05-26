---
name: task-2-memory
overview: Focused sub-plan for Task 2 (30 pts total) — episodic memory via a persistent SqliteSaver checkpointer (2a, 20 pts) and a per-user JSON profile updated on personal-info-bearing turns (2b, 10 pts). Builds on the Task 1 graph; no breaking changes to existing tools, router, or CLI flags.
todos:
  - id: t2a-checkpointer
    content: Step 1 — memory/checkpoint.py with a SqliteSaver factory (sync, file-backed, idempotent setup)
    status: completed
  - id: t2a-cli
    content: Step 2 — Wire checkpointer into CLI; switch graph.invoke/stream to use thread_id; drop the manual messages list
    status: completed
  - id: t2a-iter-reset
    content: Step 3 — Reset per-turn fields (iterations/route) explicitly on each invoke; fresh-vs-resumed banner
    status: completed
  - id: t2a-test-unit
    content: Step 4a — tests/test_checkpoint.py — fast unit tests for the SqliteSaver factory
    status: completed
  - id: t2a-test-int
    content: Step 4b — tests/test_episodic_integration.py — live integration tests for follow-ups across a simulated restart
    status: completed
  - id: t2b-schema
    content: Step 5 — memory/profile.py with UserProfile pydantic schema + load_profile/save_profile + _should_update_profile gate
    status: completed
  - id: t2b-update-node
    content: Step 6 — profile_update_node (router-LLM with structured output) + graph wiring (post-agent only on personal-info turns)
    status: completed
  - id: t2b-prompt-injection
    content: Step 7 — Render profile summary into AGENT_SYSTEM_TEMPLATE; load profile inside agent_node lazily
    status: completed
  - id: t2b-test-unit
    content: Step 8a — tests/test_profile.py — fast unit tests for the gate, schema roundtrip, and merge logic (mocked LLM)
    status: completed
  - id: t2b-test-int
    content: Step 8b — tests/test_profile_integration.py — live integration test for cross-session "What do you remember about me?" recall
    status: completed
  - id: t2-verify
    content: Step 9 — scripts/verify_task2.py with episodic + profile cases; final pytest + verifier run
    status: completed
isProject: false
---

# Task 2 — Memory (sub-plan)

## Goal

Two persistent memories, two storage models, one user-facing experience:

- **2a Episodic** — `--session demo` makes conversation history survive process
  restarts. Follow-up queries like "show me 3 more" work the next day, not just
  this minute.
- **2b User profile** — `--user ofir` keeps distilled facts (name, interests,
  preferences) across sessions. The agent can answer *"What do you remember
  about me?"* by quoting the profile.

## Grading map (30 pts)

- Episodic memory (20) — `memory/checkpoint.py`, CLI thread-id wiring, follow-up correctness
- User profile (10) — `memory/profile.py`, `profile_update_node`, system-prompt injection

## Locked decisions

- **Checkpointer:** `SqliteSaver` (sync, single-file). The graph is invoked
  synchronously today; sync saver keeps the wiring simple. `AsyncSqliteSaver`
  is documented as a future option in the README.
- **Connection lifetime:** opened once at CLI start, kept alive for the whole
  REPL. We pass the underlying `sqlite3.Connection` to a manually-constructed
  `SqliteSaver` (rather than using `from_conn_string` as a context manager),
  because the context-manager pattern would force the entire CLI loop inside
  a `with` block.
- **Storage path:** `checkpoints.sqlite` at the repo root (already gitignored).
- **Per-user profile storage:** plain JSON at `profiles/<user_id>.json`
  (gitignored). Transparent, easy for a grader to inspect, easy to delete to
  reset state. We deliberately do NOT use LangGraph's `BaseStore` for the
  profile — we want it to be a separate, inspectable artefact.
- **Profile is NOT in graph state.** The profile is per-user; checkpointed
  state is per-thread (a user can have multiple sessions). Loading from disk
  per turn keeps the model clean and avoids state-merge headaches.
- **Profile-update gate:** "smart" — regex over the latest user message for
  high-signal markers ("my name is", "i prefer", "remember that", etc.). On a
  miss we skip the LLM call entirely. This trades a small amount of recall
  (we'll miss subtle cues) for substantial cost savings on the dataset-Q&A
  turns that dominate normal use.
- **Profile injection** is unconditional: every agent turn renders the current
  profile into the system prompt, so questions like *"What do you remember about
  me?"* are answered without any tool call. If the profile is empty, we render
  *"No prior facts about this user yet."*.

## Architecture delta

```mermaid
flowchart TD
    user["User CLI"] --> cli["cli.py\n(opens SqliteSaver,\nthread_id = --session)"]
    cli --> graph["LangGraph StateGraph\ncompiled with checkpointer"]

    subgraph graph [Compiled graph]
        router["router_node"]
        decline["decline_node"]
        agent["agent_node\n(loads profile,\ninjects into prompt)"]
        tools["tool_node"]
        fallback["fallback_node"]
        profile["profile_update_node\n(only on personal-info-bearing turns)"]
    end

    router -->|"out_of_scope"| decline --> endNode([END])
    router -->|"in-scope"| agent
    agent <--> tools
    agent --> profile --> endNode
    agent --> endNode
    fallback --> profile
    fallback --> endNode

    subgraph storage [Persistent stores]
        ckpt["checkpoints.sqlite\n(per thread_id)"]
        prof["profiles/&lt;user_id&gt;.json\n(per user_id)"]
    end

    graph --- ckpt
    profile --- prof
    agent --- prof
```

## State delta

`agent/state.py` is unchanged. `pending_query` stays reserved for Bonus B.
`messages` is now mutated by the checkpointer rather than the CLI.

```python
# unchanged
class GraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    route: Route | None
    iterations: int
    user_id: str
    pending_query: str | None
```

Per-turn invoke pattern changes from passing the whole `messages` list to
passing only the new `HumanMessage`. The reducer (`add_messages`) appends to
the checkpointed history.

## Step-by-step (estimated 3–4 hrs total)

### Step 1 — `memory/checkpoint.py` (~20 min)

```python
# src/cs_agent/memory/checkpoint.py
from __future__ import annotations
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from cs_agent.config import CHECKPOINT_PATH


def get_checkpointer(path: Path = CHECKPOINT_PATH) -> SqliteSaver:
    """Return a SqliteSaver backed by a persistent file at ``path``.

    The connection is created with ``check_same_thread=False`` because the
    LangGraph saver has its own lock; SqliteSaver is documented to handle
    single-process serialisation correctly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # idempotent — creates checkpoint tables on first run
    return saver
```

### Step 2 — CLI wiring (~30 min)

In `cli.py`:
- Open the checkpointer at the top of `main`.
- `build_graph(checkpointer=cp)`.
- Drop the manual `messages: list[BaseMessage] = []` accumulator entirely —
  the checkpointer is the source of truth.
- Each turn invokes with **only the new HumanMessage** plus a per-turn reset of
  `iterations` / `route`:
  ```python
  config = {"configurable": {"thread_id": args.session}}
  initial = {
      "messages": [HumanMessage(question)],
      "iterations": 0,
      "route": None,
      "user_id": args.user,
  }
  for chunk in graph.stream(initial, config=config, stream_mode="updates"):
      _render_chunk(chunk, console)   # no longer needs a messages list
  ```
- Print a small banner: when `cp.get_tuple({"configurable": {"thread_id": s}})`
  returns a tuple, say `resumed session 'demo' (N prior turns)`; otherwise
  `starting new session 'demo'`.

### Step 3 — Iteration / route reset semantics (~15 min)

`iterations` has no custom reducer, so the dict update from each invoke
overwrites the prior value. Same for `route`. We just need to *always* pass
them in `initial` — already in Step 2 above. A short comment in `agent_node`
will document the contract.

Add a defensive guard in `agent_node`: if `iterations` is not present, default
to 0 (in case some external caller invokes without resetting).

### Step 4 — Episodic tests (~40 min, two files)

**Step 4a — `tests/test_checkpoint.py`** (fast, no LLM)

Keeps the unit-level surface for the saver factory in its own file so the
fast suite stays clearly Task-1+Task-2-unit only:

```python
# tests/test_checkpoint.py — fast, deterministic, no Nebius
from cs_agent.memory.checkpoint import get_checkpointer

def test_factory_creates_file_and_tables(tmp_path):
    p = tmp_path / "ckpt.sqlite"
    cp = get_checkpointer(p)
    assert p.exists()
    # `setup()` should be idempotent — calling get_checkpointer twice on the
    # same path must not raise.
    get_checkpointer(p)

def test_factory_returns_a_sqlitesaver(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver
    cp = get_checkpointer(tmp_path / "x.sqlite")
    assert isinstance(cp, SqliteSaver)
```

**Step 4b — `tests/test_episodic_integration.py`** (`@pytest.mark.integration`)

This file isolates the live "process-restart" behaviour. **Does NOT extend
`tests/test_agent_integration.py`** — that file is the Task-1 verifier wrapper
and stays focused on per-query routing/tool-use checks.

```python
# tests/test_episodic_integration.py
import pytest

pytestmark = pytest.mark.integration

def test_followup_within_session_uses_history(tmp_path):
    """Build the graph twice with the SAME checkpointer file. Turn 2 must
    reach the agent with turn-1's messages already in state."""
    cp_path = tmp_path / "ckpt.sqlite"

    # Round 1
    cp1 = get_checkpointer(cp_path)
    g1 = build_graph(checkpointer=cp1)
    g1.invoke(
        {"messages": [HumanMessage("Show me 3 examples of REFUND")], "iterations": 0,
         "user_id": "test", "route": None},
        config={"configurable": {"thread_id": "demo"}},
    )

    # Round 2 — fresh process simulation
    cp2 = get_checkpointer(cp_path)
    g2 = build_graph(checkpointer=cp2)
    state = g2.get_state({"configurable": {"thread_id": "demo"}}).values
    assert any("REFUND" in str(m.content) for m in state["messages"])

    # Follow-up should produce DIFFERENT examples than turn 1.
    g2.invoke(
        {"messages": [HumanMessage("Show me 3 more")], "iterations": 0,
         "user_id": "test", "route": None},
        config={"configurable": {"thread_id": "demo"}},
    )
    final = g2.get_state({"configurable": {"thread_id": "demo"}}).values
    # ≥ 4 messages: 2 human + ≥ 2 AI (likely with tool messages too)
    assert sum(isinstance(m, HumanMessage) for m in final["messages"]) == 2
```

### Step 5 — `memory/profile.py` (~45 min)

```python
# src/cs_agent/memory/profile.py
from datetime import datetime
import json
import re
from pathlib import Path
from pydantic import BaseModel, Field

from cs_agent.config import PROFILES_DIR


class UserProfile(BaseModel):
    user_id: str
    name: str | None = None
    role: str | None = None
    topics_of_interest: list[str] = Field(default_factory=list)
    preferences: dict[str, str] = Field(default_factory=dict)
    notable_facts: list[str] = Field(default_factory=list)
    last_updated: datetime | None = None

    def render_for_prompt(self) -> str:
        """Produce a short, agent-readable summary."""
        ...


def profile_path(user_id: str) -> Path: ...

def load_profile(user_id: str) -> UserProfile: ...   # empty if missing

def save_profile(p: UserProfile) -> None: ...        # atomic write


# Personal-info-bearing markers. Conservative on purpose to avoid false hits.
PROFILE_MARKERS = re.compile(
    r"\b("
    r"my name is|i'm called|call me|"
    r"i prefer|i like|i love|i hate|i don't like|"
    r"remember (?:that|me)|note that|for next time|"
    r"i work as|my role is|my job is"
    r")\b",
    re.IGNORECASE,
)


def is_personal_info_bearing(message: str) -> bool:
    return bool(PROFILE_MARKERS.search(message))
```

### Step 6 — `profile_update_node` + graph wiring (~45 min)

`agent/nodes.py`:

```python
PROFILE_UPDATE_SYSTEM = """You maintain a small structured profile for one user.
Given the current profile JSON and the latest user/assistant turn, return the
updated profile. Add new facts, refine existing ones if contradicted, drop
items the user retracts. Keep notable_facts short and deduped.
Return ONLY a JSON object matching the UserProfile schema."""


def profile_update_node(state: GraphState) -> dict:
    user_id = state.get("user_id") or "anon"
    last_human = next(...HumanMessage...)
    if not is_personal_info_bearing(last_human.content):
        return {}
    current = load_profile(user_id)
    updated = (
        get_router_llm()
        .with_structured_output(UserProfile)
        .invoke([SystemMessage(PROFILE_UPDATE_SYSTEM),
                 HumanMessage(json.dumps({
                     "current": current.model_dump(mode="json"),
                     "latest_turn": ...,
                 }))])
    )
    updated.user_id = user_id
    updated.last_updated = datetime.utcnow()
    save_profile(updated)
    return {}  # no state change
```

Graph topology change in `agent/graph.py`:
- Add `profile` node between agent (final answer) → END.
- Add `profile` node between fallback → END.
- `decline` skips profile (an off-topic decline carries no user-relevant info).

The conditional edge from `should_continue` becomes:
```
{"tools": "tools", "fallback": "fallback", "end": "profile"}
```
And `profile → END`.

`fallback → profile → END`.

### Step 7 — Prompt injection (~20 min)

In `agent/prompts.py`:

```python
def build_agent_system(route, user_id="anon"):
    profile = load_profile(user_id)
    profile_block = profile.render_for_prompt() if profile.has_facts() \
                    else "No prior facts about this user yet."
    base = AGENT_SYSTEM_TEMPLATE.format(...)
    hint = ROUTE_HINTS[route]
    return f"{base}\n\nUSER PROFILE\n{profile_block}\n\n{hint}"
```

`agent_node` already calls `build_agent_system(route=...)`; we extend the call
with `user_id=state.get("user_id", "anon")`.

### Step 8 — Profile tests (~40 min, two files)

**Step 8a — `tests/test_profile.py`** (fast, mocked LLM, no Nebius)

```python
# tests/test_profile.py
def test_gate_fires_on_personal_info():
    assert is_personal_info_bearing("Hi, my name is Ofir")
    assert is_personal_info_bearing("I prefer concise answers")
    assert is_personal_info_bearing("Remember that I work as a DE")

def test_gate_does_not_fire_on_dataset_questions():
    assert not is_personal_info_bearing("How many refund requests?")
    assert not is_personal_info_bearing("Summarize the FEEDBACK category")
    assert not is_personal_info_bearing("Show me 3 examples of REFUND")

def test_load_profile_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_mod, "PROFILES_DIR", tmp_path)
    p = load_profile("nobody")
    assert p.user_id == "nobody"
    assert p.name is None and p.notable_facts == []

def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_mod, "PROFILES_DIR", tmp_path)
    save_profile(UserProfile(user_id="u1", name="Ofir", notable_facts=["a"]))
    p = load_profile("u1")
    assert p.name == "Ofir" and "a" in p.notable_facts

def test_update_node_skips_when_gate_misses(monkeypatch):
    """Mocked router LLM must NOT be invoked for non-personal messages."""
    fake_llm = _FakeLLM(...)  # raises if .with_structured_output is called
    monkeypatch.setattr(profile_mod, "get_router_llm", lambda: fake_llm)
    state = {"messages": [HumanMessage("How many refund requests?")],
             "user_id": "u1"}
    assert profile_update_node(state) == {}

def test_update_node_invokes_llm_and_saves_when_gate_fires(monkeypatch):
    """Mock the router LLM to return a UserProfile; assert save_profile called."""
    ...
```

**Step 8b — `tests/test_profile_integration.py`** (`@pytest.mark.integration`)

Live cross-session recall test in its own file — does not mix with the Task-1
agent verifier or the episodic checkpoint integration tests.

```python
# tests/test_profile_integration.py
import pytest
pytestmark = pytest.mark.integration

def test_profile_recall_across_sessions(tmp_path, monkeypatch):
    """Turn 1 (session A): 'Hi, my name is Ofir'.
    Turn 2 (session B, same user): 'What do you remember about me?'
    The answer must contain 'Ofir'."""
    ...
```

### Step 9 — `scripts/verify_task2.py` (~25 min)

Create a **separate** verifier so Task 1 and Task 2 stay independently
runnable. `verify_task1.py` stays untouched (it's the documented Task 1
deliverable).

`scripts/verify_task2.py` contains its own `MultiTurnCase` dataclass (a list
of (user_id, session_id, query) tuples per case) and runs them serially
through a single graph compiled with a tmp-dir checkpointer. Two cases:

| Case | Setup | Assertion |
|---|---|---|
| `episodic-followup` | "Show me 3 examples of REFUND" → "show me 3 more" | 2 distinct sets of REFUND examples in the trace |
| `profile-recall` | "Hi, my name is Ofir" → process restart → "What do you remember about me?" | Final answer contains "Ofir" |

The script uses a `tmp_path`-style temp dir for both `checkpoints.sqlite` and
`profiles/` so it doesn't pollute the user's real working state. README's
"Run" section will gain a `uv run python scripts/verify_task2.py` line next
to the existing Task 1 verifier.

The wrapper integration tests in `tests/test_agent_integration.py` are
**not** modified — they remain a Task-1-flavoured per-query verifier and
import only `scripts/verify_task1.py`'s `CASES`.

## Files this task creates/modifies

```
src/cs_agent/memory/
├── __init__.py                  # already exists
├── checkpoint.py                # NEW (Step 1)
└── profile.py                   # NEW (Step 5)

src/cs_agent/cli.py              # MODIFIED (Step 2): checkpointer wiring + drop messages list
src/cs_agent/agent/
├── nodes.py                     # MODIFIED (Step 6): + profile_update_node
├── graph.py                     # MODIFIED (Step 6): + profile node and edges
└── prompts.py                   # MODIFIED (Step 7): build_agent_system takes user_id

# Tests — every new behaviour gets its own file. Existing Task-1 test files are not touched.
tests/
├── test_checkpoint.py           # NEW (Step 4a) — fast unit tests for the saver factory
├── test_episodic_integration.py # NEW (Step 4b) — live, marked integration
├── test_profile.py              # NEW (Step 8a) — fast unit tests, mocked LLM
└── test_profile_integration.py  # NEW (Step 8b) — live cross-session recall

scripts/verify_task2.py          # NEW (Step 9) — multi-turn cases for episodic + profile

# UNCHANGED on purpose:
#  - tests/test_tools.py
#  - tests/test_tools_integration.py
#  - tests/test_router.py
#  - tests/test_agent_integration.py
#  - scripts/verify_task1.py
```

No new dependencies. `langgraph-checkpoint-sqlite` is already in the lock file.

## Acceptance criteria

- `uv run cs-agent --session demo` shows `starting new session 'demo'` on first
  run; on a second run says `resumed session 'demo' (N prior turns)`.
- A 2-turn dialog "Show me 3 examples of REFUND" → "Show me 3 more" produces
  examples that don't repeat (verifies follow-up reasoning works).
- `profiles/ofir.json` exists after a turn that says "my name is Ofir";
  `cat profiles/ofir.json` shows `{"name": "Ofir", ...}`.
- `uv run cs-agent --user ofir` after a process restart, asking "What do you
  remember about me?", returns an answer containing "Ofir".
- All existing pytest cases still pass; new tests pass.
- `fon check` and `ruff format/check` clean.

## Out of scope

- LangGraph cross-thread `BaseStore` (we use plain JSON for the profile).
- Async checkpointing (`AsyncSqliteSaver`) — sync SqliteSaver is sufficient.
- PostgresSaver — overkill for an assignment.
- Profile-driven personalisation of tool behaviour (we only personalise the
  agent's responses, not the tools).
- Encryption of the profile file (synthetic data; not needed).
