"""Verify Bonus B (Query Recommender) plumbing without spinning up Nebius.

Seven cases — six offline (mocked LLMs / hand-built state), one live
(guarded by ``NEBIUS_API_KEY``) that drives the real graph through
suggest → confirm end-to-end:

1. ``module-import``           — public surface of
                                 ``cs_agent.agent.recommender`` and the new
                                 ``"recommend"`` literal on ``Route``.
2. ``router-short-circuit``    — ``router_node`` returns
                                 ``{"route": "recommend"}`` when
                                 ``pending_query`` is set, without ever
                                 touching the router LLM.
3. ``suggest-shape``           — ``recommender_node`` on an empty state
                                 calls the suggestion LLM, emits an
                                 AIMessage, and stores the suggested query
                                 in ``pending_query``.
4. ``confirm-routes-to-agent`` — confirmation clears ``pending_query``,
                                 appends a synthetic HumanMessage, and
                                 ``route_from_recommender`` returns
                                 ``"agent"``.
5. ``refine-regenerates``      — refinement updates ``pending_query`` to a
                                 new suggestion; edge → ``"profile"``.
6. ``reject-clears``           — rejection clears ``pending_query`` and
                                 emits a polite acknowledgement; edge →
                                 ``"profile"``.
7. ``live-flow`` (optional)    — real graph: send "what should I query
                                 next?", then "yes", and verify a tool
                                 call fired and ``pending_query`` ended as
                                 ``None``.

Per-case logic lives in ``scripts/_bonus_b_checks.py``; the shared
PASS/FAIL runner lives in ``scripts/_verifier_runner.py``.

Run with::

    uv run python scripts/verify_bonus_b.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bonus_b_checks import (  # noqa: E402
    check_confirm_routes_to_agent,
    check_live_flow,
    check_recommender_module_import,
    check_refine_regenerates,
    check_reject_clears,
    check_router_short_circuit,
    check_suggest_shape,
)
from _verifier_runner import Case, run_main  # noqa: E402

CASES: list[Case] = [
    Case(
        label="module-import",
        description="recommender module + Route literal expose Bonus B surface.",
        fn=check_recommender_module_import,
        notes="Suggestion, RecommenderIntent, recommender_node, route_from_recommender.",
    ),
    Case(
        label="router-short-circuit",
        description="router_node returns 'recommend' without calling the LLM when pending_query is set.",
        fn=check_router_short_circuit,
        notes="Avoids misclassifying 'yes' / 'no' replies as structured/out-of-scope.",
    ),
    Case(
        label="suggest-shape",
        description="recommender first-call sets pending_query and emits an AIMessage.",
        fn=check_suggest_shape,
        notes="Mocked Suggestion LLM; asserts pending_query == suggested query.",
    ),
    Case(
        label="confirm-routes-to-agent",
        description="confirmation appends synthetic HumanMessage and routes to agent.",
        fn=check_confirm_routes_to_agent,
        notes="iterations=0 reset gives the agent its full ReAct budget.",
    ),
    Case(
        label="refine-regenerates",
        description="refinement updates pending_query to a new suggestion.",
        fn=check_refine_regenerates,
        notes="route_from_recommender returns 'profile' (not 'agent') on refine.",
    ),
    Case(
        label="reject-clears",
        description="rejection clears pending_query and acknowledges.",
        fn=check_reject_clears,
        notes="Next turn routes normally through the router (pending_query=None).",
    ),
    Case(
        label="live-flow",
        description="real graph: suggest → confirm → agent executes a tool.",
        fn=check_live_flow,
        notes="Skipped without NEBIUS_API_KEY; otherwise drives 2 turns against tmp SqliteSaver.",
    ),
]


def main() -> int:
    """Entry point so callers can ``import scripts.verify_bonus_b as v; v.main()``."""
    return run_main(CASES, banner="Verifying Bonus B (Query Recommender)")


if __name__ == "__main__":
    sys.exit(main())
