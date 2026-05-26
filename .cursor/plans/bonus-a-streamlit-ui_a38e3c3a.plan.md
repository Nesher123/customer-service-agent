---
name: bonus-a-streamlit-ui
status: completed
overview: "DONE. Wrap the existing LangGraph CLI agent in a Streamlit chat UI that shows live reasoning steps and supports per-session memory via a sidebar session-id input — reusing the compiled graph and SqliteSaver checkpointer, leaving cli.py untouched. Shipped as src/cs_agent/ui/{rendering,streamlit_app}.py + scripts/verify_bonus_a.py + cs-agent-ui console script. Verifier: 4/4 passed (incl. live turn). fon check: 16/16 passed. Pytest -m 'not integration': 85 passed (no regression)."
todos:
  - id: ui_package
    content: Create src/cs_agent/ui/__init__.py and src/cs_agent/ui/streamlit_app.py with cached graph factory, sidebar (session_id, user_id, switch button), history replay from graph.get_state, and per-turn streaming renderer (st.status for reasoning, st.chat_message for final text)
    status: completed
  - id: console_entry
    content: Add 'cs-agent-ui = cs_agent.ui.streamlit_app:main' to [project.scripts] in pyproject.toml; implement main() that delegates to streamlit.web.cli.main with the app path
    status: completed
  - id: verify_script
    content: "Add scripts/verify_bonus_a.py matching the verify_task1/2/3.py pattern: import smoke test, helper-function smoke test on synthetic chunks, and an optional one-turn end-to-end run gated on NEBIUS_API_KEY"
    status: completed
  - id: readme_update
    content: "Update README.md: flip Bonus A status to 'done' and add a Streamlit UI subsection under Running with both entrypoints and a short feature description"
    status: completed
  - id: fon_check
    content: Run `fon check` and address any rule violations before finishing
    status: completed
isProject: false
---

## Bonus A — Streamlit UI

### Design

```mermaid
flowchart LR
    user["User browser"] --> st["streamlit_app.py"]
    st -->|"st.cache_resource"| graph["build_graph(checkpointer)"]
    st -->|"st.cache_resource"| ckpt["get_checkpointer()"]
    graph --> sqlite[("checkpoints.sqlite")]
    st -->|"per-turn invoke (HumanMessage only)"| graph
    graph -->|"stream_mode='updates'"| st
    st -->|"chat history on session switch"| graphState["graph.get_state(config).values['messages']"]
```

Key design choices, mirroring the CLI contract in [src/cs_agent/cli.py](src/cs_agent/cli.py):

- **Per-turn contract**: invoke the graph with `{"messages": [HumanMessage(q)], "iterations": 0, "user_id": user_id, "route": None}` — the checkpointer owns history. Same payload as `cli.py:225-230`.
- **Self-contained renderer**: walk the `chunk` dict from `stream_mode="updates"` directly inside the Streamlit file. Map `router` → small caption, `agent` with `tool_calls` → bullet inside an `st.status("reasoning…")` expander, `tools` → expander row, final `agent` text / `decline` / `fallback` → `st.chat_message("assistant")`.
- **Resumed view on session switch**: pull `graph.get_state(config).values["messages"]` and re-render in chronological order. Only the chat bubbles are available (no router/tool chunks in the checkpoint) — that's fine; live turns still get the full reasoning view.
- **Resource caching**: `@st.cache_resource` for both the `SqliteSaver` (kept alive across reruns via the existing `check_same_thread=False` connection in [src/cs_agent/memory/checkpoint.py](src/cs_agent/memory/checkpoint.py)) and the compiled `graph`.

### Files to create

- `src/cs_agent/ui/__init__.py` — empty package marker.
- `src/cs_agent/ui/streamlit_app.py` (<250 lines) with:
  - `@st.cache_resource def _get_graph() -> CompiledGraph` — calls `get_checkpointer()` then `build_graph(checkpointer)`.
  - Sidebar block: `st.text_input("session id", value="default")`, `st.text_input("user id", value="anon")`, a "Switch / reload" button, and a `st.caption` showing `resumed N prior turns` (computed from `graph.get_state(config)`).
  - History rendering: on first load or session-id change, replay `graph.get_state(config).values["messages"]` as chat bubbles (pair `ToolMessage`s under the `AIMessage` that issued them by `tool_call_id`, inside a collapsed expander).
  - `prompt = st.chat_input(...)`. On submit, render user bubble, then stream the graph inside `with st.chat_message("assistant")` using an `st.status("reasoning…", expanded=False)` container for tool-call/result lines and writing the final text below. Catch exceptions like the CLI does and surface via `st.error`.
  - A `main()` function used by the console-script entry, which `sys.argv`-substitutes and invokes `streamlit.web.cli.main()` so `uv run cs-agent-ui` works.
- `scripts/verify_bonus_a.py` — matches the `verify_task1/2/3.py` pattern. Smoke test that:
  - Imports `cs_agent.ui.streamlit_app` (catches import-time errors).
  - Asserts the small chunk-to-renderable helpers (router/agent/tools/decline/fallback) handle a representative chunk shape without raising (no Streamlit runtime needed — they should be plain functions returning lightweight payloads).
  - Builds the graph against a throw-away temp `SqliteSaver`, runs one canned turn (`"What categories exist?"` only if `NEBIUS_API_KEY` is set; otherwise just exits 0 with a skip note, mirroring how `verify_task3.py` guards live calls).

### Files to modify

- [pyproject.toml](pyproject.toml): add `cs-agent-ui = "cs_agent.ui.streamlit_app:main"` under `[project.scripts]`. (`streamlit>=1.36` is already declared under the `ui` optional extra — no dep work.)
- [README.md](README.md):
  - Flip the Bonus A row in the status table from `up next` to `done`.
  - Add a "Streamlit UI (Bonus A)" subsection under "Running" showing both entrypoints — `uv run cs-agent-ui` and the fallback `uv run streamlit run src/cs_agent/ui/streamlit_app.py` — plus a short screenshot-placeholder description of the sidebar and the live reasoning expander.

### Files explicitly NOT touched

- [src/cs_agent/cli.py](src/cs_agent/cli.py), [src/cs_agent/agent/graph.py](src/cs_agent/agent/graph.py), [src/cs_agent/agent/nodes.py](src/cs_agent/agent/nodes.py), [src/cs_agent/memory/checkpoint.py](src/cs_agent/memory/checkpoint.py) — no behavior changes; the Streamlit app is a pure new consumer of the same graph contract.

### fon / quality checks

- New `src/cs_agent/ui/` directory will have 2 files (`__init__.py`, `streamlit_app.py`) — well under the 12-files-per-dir cap.
- Streamlit file target is ~200 lines; below the 300-line cap. If it grows past the cap, split into `ui/streamlit_app.py` + `ui/rendering.py` (preferred over flat `streamlit_*` siblings to satisfy the `prefix_grouping` rule).
- Run `fon check` after the edits as required by repo rules.