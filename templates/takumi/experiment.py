"""Inner Loop orchestrator for the Takumi template.

This file is the *Aider-editable* entry point. The outer loop (AI Scientist v1)
proposes Skills format hypotheses by mutating the ``FORMAT_HYPOTHESIS`` dict
below and rerunning the script with a new ``--out_dir``.

The ``FORMAT_HYPOTHESIS`` dict has two AI-Scientist controlled axes:

1. ``components``: subset of S3..S12 (S1, S2 are always added).
2. ``show_genome`` (bool): whether the Articulator can see the Expert's
   network (topology + weights). ``False`` = trajectory only (旧 Stage 1);
   ``True`` = trajectory + genome (旧 Stage 2). This 1-bit axis tests the
   Polanyi sub-hypothesis: does seeing the tacit-layer network help the
   conscious teaching layer?

Every operational detail (network forward, env wrapper, articulator, mutator,
evaluation, model selection, temperatures, budget thresholds) lives under
``lib/`` and ``config.yaml`` and is intentionally NOT meant to be edited by
Aider. Only this file and ``plot.py`` are within the editable scope.

Usage::

    python experiment.py --out_dir=run_0          # full run with real API
    python experiment.py --out_dir=run_test --mock  # offline dry-run
    python experiment.py --out_dir=run_0 --config=alt_config.yaml  # alt config

================================================================================
Context for the AI Scientist (read carefully before proposing ideas)
--------------------------------------------------------------------------------
The two sections below are *summaries* meant to give the AI Scientist outer
loop (and any human reader of this file) enough vocabulary to reason about
SlimeVolley and the Skills section catalogue when choosing values for
``FORMAT_HYPOTHESIS``. The **source of truth** is
``lib/articulator.py`` — specifically ``SECTION_FIXED_TEXTS`` (for S1/S2)
and ``SECTION_DESCRIPTIONS`` (for S3..S12). If the two diverge, trust
``lib/articulator.py``; the docstring is informational only.

SlimeVolley game rules
^^^^^^^^^^^^^^^^^^^^^^
* Episode: up to 3000 steps against a built-in opponent.
* Reward: +1 when the agent scores a point, -1 when the opponent scores.
* Observation (12 floats, all scaled by 1/10):
    indices 0..3  = agent    (x, y, vx, vy)
    indices 4..7  = ball     (x, y, vx, vy)
    indices 8..11 = opponent (x, y, vx, vy)
* Action (Novice network outputs 3 floats, mapped to 3 binary controls):
    out_0 = left, out_1 = right, out_2 = jump.
    A control fires when its output is > 0.
* Fitness = mean total reward over ``eval_episodes`` (default 100) test-mode
  episodes (5-life rule); ``learning_curve_scores`` records per-cycle training
  rewards (no life-limit) for diagnostic purposes.

Skills section catalogue (subset selected via ``components``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
S1: Goal (fixed)   — task definition; always emitted verbatim.
S2: Rules (fixed)  — action/observation vocabulary; always emitted verbatim.

S3: Phase markers  — decompose play into 4-6 rally phases with onset patterns
                     and dominant action class.
S4: Key states    — 6-10 IF-THEN production rules over the 12-dim obs.
S5: Trajectory exemplars — 3-5 worked (t, obs, action) excerpts with rationale.
S6: Counter-examples — 5-8 NEVER-WHEN anti-patterns the novice should avoid.
S7: External focus — 4-6 imperatives directing attention to ball/opponent/effect.
S8: Internal focus — 4-6 directives on body/posture mechanics
                     (**negative control**, expected to underperform S7 per Wulf 2013).
S9: Metaphor       — 1-2 unifying analogies setting the global stance.
S10: Onomatopoeia  — re-narrate play using Japanese 擬音語/擬態語 per phase.
S11: Statistics    — quantitative profile (action frequency, per-phase action
                     distribution, ball.y at jump moment, opponent distance, ...).
S12: Stage-conditional — 3 advice sets indexed by novice's rolling proficiency
                         (守 / 破 / 離).
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make `lib.*` importable when invoked from any CWD.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# テンプレート同梱の `.env` から ANTHROPIC_API_KEY 等を自動読込.
# (既に shell で export されている値は override しない = `override=False`)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(_HERE / ".env", override=False)
except ImportError:
    # python-dotenv が無くても shell export 経由で動かす運用は壊さない
    pass

from lib.budget import BudgetTracker  # noqa: E402
from lib.config import default_config_path, load_config  # noqa: E402
from lib.eval import compute_final_info, run_inner_loop  # noqa: E402

# ---------------------------------------------------------------------------
# Skills format hypothesis (Aider edits THIS dict to test new ideas)
#
# Two experimental axes Aider iterates:
#   * `components`: subset of {"S3", ..., "S12"} (S1, S2 are always added).
#   * `show_genome` (bool): False = trajectory-only Articulator input
#     (旧 Stage 1); True = trajectory + Expert network YAML (旧 Stage 2).
#
# Everything else (config.yaml, lib/, data/) is fixed infrastructure.
# ---------------------------------------------------------------------------

FORMAT_HYPOTHESIS = {
    "components": [],
    "show_genome": False,
    "instructions": "Goal and Rules",
}


def _resolve_config(arg_path: str | None) -> Path:
    if arg_path:
        return Path(arg_path).resolve()
    env_path = os.environ.get("TAKUMI_CONFIG")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = _HERE / p
        return p.resolve()
    return default_config_path(_HERE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Takumi inner loop")
    parser.add_argument(
        "--out_dir", type=str, default="run_0",
        help="Output directory (AI Scientist convention)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: <template>/config.yaml)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Skip real LLM calls; use a deterministic mock for sanity tests.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = _resolve_config(args.config)
    config = load_config(config_path)

    use_mock = args.mock
    if not use_mock and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "[experiment] ANTHROPIC_API_KEY is not set; falling back to --mock mode. "
            "Set the env var and rerun for a real evaluation.",
            file=sys.stderr,
        )
        use_mock = True

    budget = BudgetTracker(
        out_dir=out_dir,
        warning_threshold=config.budget.warning_threshold_usd,
        hard_limit=config.budget.hard_limit_usd,
    )

    means = run_inner_loop(
        FORMAT_HYPOTHESIS,
        config,
        out_dir=out_dir,
        use_mock=use_mock,
        budget=budget,
    )

    final_info = compute_final_info(means)
    out_path = out_dir / "final_info.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_info, f, ensure_ascii=False, indent=2)
    print(f"[experiment] wrote {out_path}")
    print(
        f"[experiment] config={config_path.name} "
        f"label={config.experiment.run_label!r} "
        f"show_genome={FORMAT_HYPOTHESIS.get('show_genome', False)} "
        f"K={config.inner_loop.K} "
        f"eval_ep={config.inner_loop.eval_episodes}"
    )
    print(
        f"[experiment] fitness_mean_100ep = "
        f"{means['fitness_mean_100ep']:.3f}, "
        f"win_rate = {means['win_rate_at_best_cycle']:.2f}, "
        f"components = {means['components_used']}, "
        f"cumulative_cost = ${budget.total_usd:.3f}"
    )


if __name__ == "__main__":
    main()
