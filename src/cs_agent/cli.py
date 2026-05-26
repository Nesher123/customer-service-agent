"""Interactive CLI for the customer-service data-analyst agent.

Run with::

    uv run cs-agent --session demo --user ofir
    # or
    uv run python -m cs_agent.cli --session demo --user ofir

The REPL prints every reasoning step (router decision, tool calls,
observations, fallbacks, final answer) in a colour-coded ``rich`` trace, so the
grader can see *how* the agent arrived at its answer — not just the answer.

Episodic memory (Task 2a) is on by default: the graph is compiled with a
``SqliteSaver`` checkpointer keyed by ``make_thread_id(--user, --session)``.
Each turn invokes the graph with only the *new* HumanMessage plus a per-turn
reset of ``iterations`` / ``route``; the checkpointer owns the running history.
So "show me 3 more" works the next day, not just this minute.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from cs_agent.agent.graph import build_graph
from cs_agent.agent.state import GraphState
from cs_agent.memory.checkpoint import get_checkpointer, make_thread_id

# Suppress library log spam from inside agent_node ("loop detected — short-circuiting")
# unless the user explicitly raises the level.
logging.basicConfig(level=logging.WARNING, format="%(message)s")

EXIT_WORDS = {"exit", "quit", ":q"}
TOOL_RESULT_PREVIEW_CHARS = 240


def _format_args(args: dict[str, Any]) -> str:
    """Render a tool-call's argument dict compactly for the trace."""
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        if v is None:
            continue
        s = repr(v) if isinstance(v, str) else str(v)
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _truncate(text: str, limit: int = TOOL_RESULT_PREVIEW_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …"


def _render_router(update: dict[str, Any], console: Console) -> None:
    route = update.get("route")
    console.print(f"  [dim]router → [bold]{route}[/][/]")


def _render_agent(update: dict[str, Any], console: Console) -> None:
    for msg in update.get("messages", []):
        if not isinstance(msg, AIMessage):
            continue
        if msg.tool_calls:
            for tc in msg.tool_calls:
                console.print(f"  [yellow]→ {tc['name']}([dim]{_format_args(tc.get('args') or {})}[/])[/]")
        elif msg.content:
            console.print(
                Panel(
                    Text(str(msg.content)),
                    title="agent",
                    border_style="green",
                    padding=(0, 1),
                )
            )


def _render_tools(update: dict[str, Any], console: Console) -> None:
    for msg in update.get("messages", []):
        if not isinstance(msg, ToolMessage):
            continue
        preview = _truncate(str(msg.content))
        console.print(f"  [green]← {msg.name}[/] [dim]→[/] {preview}")


def _render_terminal(node: str, update: dict[str, Any], console: Console) -> None:
    """Render decline_node / fallback_node output."""
    border_style = "red" if node == "decline" else "yellow" if node == "fallback" else "green"
    title = {
        "decline": "out-of-scope decline",
        "fallback": "fallback (loop / max-iter)",
        "profile_recall": "profile recall",
    }[node]
    for msg in update.get("messages", []):
        if isinstance(msg, AIMessage) and msg.content:
            console.print(
                Panel(
                    Text(str(msg.content)),
                    title=title,
                    border_style=border_style,
                    padding=(0, 1),
                )
            )


def _render_recommender(update: dict[str, Any], console: Console) -> None:
    """Render recommender output.

    Three flavours, all distinguishable from the update dict alone:

    - **suggest / refine** (``pending_query`` set, last msg = AIMessage):
      a cyan panel titled "recommender (awaiting confirmation)".
    - **reject** (``pending_query`` cleared, last msg = AIMessage): same
      cyan panel but titled "recommender (suggestion dropped)".
    - **confirm** (``pending_query`` cleared, last msg = HumanMessage):
      a dim grey line showing the dispatched query — the agent panel that
      follows will render the actual answer.
    """
    pending = update.get("pending_query")
    messages = update.get("messages") or []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            console.print(f"  [dim]recommender → executing:[/] [italic]{msg.content}[/]")
        elif isinstance(msg, AIMessage) and msg.content:
            title = "recommender (awaiting confirmation)" if pending else "recommender (suggestion dropped)"
            console.print(
                Panel(
                    Text(str(msg.content)),
                    title=title,
                    border_style="cyan",
                    padding=(0, 1),
                )
            )


def _render_chunk(
    chunk: dict[str, dict[str, Any]],
    console: Console,
) -> None:
    """Print a single LangGraph stream update.

    The checkpointer owns conversation history now; the renderer is purely a
    visualiser of the per-turn delta and does not accumulate state.
    """
    for node_name, update in chunk.items():
        if node_name == "router":
            _render_router(update, console)
        elif node_name == "agent":
            _render_agent(update, console)
        elif node_name == "tools":
            _render_tools(update, console)
        elif node_name in {"decline", "fallback", "profile_recall"}:
            _render_terminal(node_name, update, console)
        elif node_name == "recommender":
            _render_recommender(update, console)


def _count_human_turns(graph, config: RunnableConfig) -> int:
    """Best-effort count of prior HumanMessages persisted under ``thread_id``.

    Returns 0 for a fresh thread (no checkpoint yet) or if anything goes
    wrong while reading state. Used purely for the welcome banner; never
    raises into the REPL.
    """
    try:
        snapshot = graph.get_state(config)
    except Exception:  # noqa: BLE001 — banner is informational; never crash on it
        return 0
    if snapshot is None:
        return 0
    messages = (snapshot.values or {}).get("messages") or []
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def _print_banner(console: Console, session: str, user: str, prior_turns: int) -> None:
    if prior_turns:
        status = f"resumed session [cyan]{session}[/] ([bold]{prior_turns}[/] prior turns)"
    else:
        status = f"starting new session [cyan]{session}[/]"
    body = (
        "[bold]Customer Service Data Analyst[/] — Bitext dataset agent\n"
        f"{status}   user: [cyan]{user}[/]\n"
        f"type [yellow]{', '.join(sorted(EXIT_WORDS))}[/] (or Ctrl-C / Ctrl-D) to quit"
    )
    console.print(Panel.fit(body, title="cs-agent", border_style="blue"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="cs-agent",
        description="Interactive ReAct agent for the Bitext customer-support dataset.",
    )
    p.add_argument(
        "--session",
        default="default",
        help="Session id. With --user, scopes episodic memory to this "
        "conversation (same user + session resumes; different users do not "
        "share history). Default: 'default'.",
    )
    p.add_argument(
        "--user",
        default="anon",
        help="User id. In Task 2b this keys the per-user profile. Default: 'anon'.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show INFO-level logs (router fallbacks, loop-detection, etc.).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    console = Console()
    # Open the checkpointer for the lifetime of the REPL. Compile the graph
    # against it so every invoke/stream call is automatically persisted under
    # a composite thread keyed by user + session.
    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer=checkpointer)

    thread_id = make_thread_id(args.user, args.session)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    prior_turns = _count_human_turns(graph, config)
    _print_banner(console, args.session, args.user, prior_turns)

    while True:
        try:
            question = Prompt.ask("\n[bold cyan]you[/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye[/]")
            return 0

        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            console.print("[dim]bye[/]")
            return 0

        # Pass ONLY the new HumanMessage. The checkpointer's ``add_messages``
        # reducer appends it to the persisted history. Per-turn fields are
        # explicitly reset so leftover ``iterations`` / ``route`` from the
        # previous turn never leak into this one (Step 3 contract).
        initial: GraphState = {
            "messages": [HumanMessage(question)],
            "iterations": 0,
            "user_id": args.user,
            "route": None,
        }

        try:
            for chunk in graph.stream(initial, config=config, stream_mode="updates"):
                _render_chunk(chunk, console)
        except Exception as exc:  # noqa: BLE001 — interactive REPL: never crash on a single bad turn
            # The checkpointer hasn't committed the failed turn's tail (the
            # graph errored before completion), but the HumanMessage we passed
            # in may already be persisted. That's fine — surfacing it as the
            # last user turn is honest and keeps the thread coherent.
            console.print(f"[red]error during turn:[/] {exc!r}")


if __name__ == "__main__":
    sys.exit(main())
