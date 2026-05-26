"""Shared scaffolding for the ``scripts/verify_bonus_*.py`` runners.

Each bonus verifier defines a list of named cases (each a callable that
returns a short detail string or raises on failure). The runner here owns
the boring bits: timing, success/failure capture, per-case progress lines,
and the final PASS/FAIL summary table.

Task verifiers (``verify_task{1,2,3}.py``) have different per-case shapes
(e.g. Task 1 cases also carry an expected route and a tool-name allow-list)
so they keep their own runners.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Case:
    """One verifier case: a callable that either returns a detail string or raises."""

    label: str
    description: str
    fn: Callable[[], str]
    notes: str = ""


@dataclass
class CaseResult:
    """Outcome of running one ``Case``."""

    case: Case
    passed: bool = False
    error: str | None = None
    elapsed_s: float = 0.0
    detail: str = ""


def run_case(case: Case) -> CaseResult:
    """Execute one ``Case`` and capture its outcome.

    ``AssertionError`` is treated as an expected failure (short message);
    any other exception is treated as a defect (full ``repr``). Either way
    the runner keeps going so a single broken case can't mask the rest.
    """
    res = CaseResult(case=case)
    t0 = time.time()
    try:
        res.detail = case.fn() or ""
        res.passed = True
    except AssertionError as exc:
        res.error = str(exc)
    except Exception as exc:  # noqa: BLE001 — verifier keeps going on any failure
        res.error = repr(exc)
    res.elapsed_s = time.time() - t0
    return res


def print_summary(results: list[CaseResult], label_width: int = 24) -> None:
    """Print the per-case PASS/FAIL summary table."""
    print("\n" + "=" * 100)
    print(f"{'#':>2}  {'PASS':<5} {'LABEL':<{label_width}} {'TIME':>7}  DETAIL")
    print("=" * 100)
    for i, r in enumerate(results, 1):
        mark = "✓" if r.passed else "✗"
        detail = r.detail if r.passed else (r.error or "")
        print(f"{i:>2}  {mark:<5} {r.case.label:<{label_width}} {r.elapsed_s:>6.2f}s  {detail}")
    print("=" * 100)
    n_pass = sum(1 for r in results if r.passed)
    print(f"\nResult: {n_pass}/{len(results)} passed, {len(results) - n_pass} failed")


def run_main(cases: list[Case], banner: str, label_width: int = 24) -> int:
    """Run ``cases`` in order, print progress + summary, return exit code.

    Returns ``0`` iff every case passed, otherwise ``1``. Suitable to use
    directly from ``if __name__ == "__main__": sys.exit(run_main(...))``.
    """
    print(f"{banner} — {len(cases)} cases.\n", flush=True)
    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case.label:<{label_width}} ", end="", flush=True)
        r = run_case(case)
        results.append(r)
        mark = "✓" if r.passed else "✗"
        suffix = f"  ({r.error})" if r.error else f"  ({r.detail})"
        print(f"{mark}  ({r.elapsed_s:.2f}s){suffix}", flush=True)
    print_summary(results, label_width=label_width)
    return 0 if all(r.passed for r in results) else 1
