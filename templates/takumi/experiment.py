"""Inner Loop orchestrator for the Takumi template.

This file is the *Aider-editable* entry point. The outer loop (AI Scientist v1)
proposes Skills format hypotheses by mutating the ``FORMAT_HYPOTHESIS`` dict
below and rerunning the script with a new ``--out_dir``.

Stage (1 = trajectory-only, 2 = + genome) is **NOT** an Aider-controlled
variable: it is set by the experimenter in ``config.yaml`` so that Stage 1 vs
Stage 2 ablations are clean experimental conditions, not entangled with the
component-selection axis. To compare Stage 1 vs Stage 2 for the same set of
ideas, run the outer loop twice with different ``config.yaml::experiment.stage``.

Every other operational detail (network forward, env wrapper, articulator,
mutator, evaluation, model selection, temperatures, budget thresholds) lives
under ``lib/`` and ``config.yaml`` and is intentionally NOT meant to be edited
by Aider. Only this file and ``plot.py`` are within the editable scope.

Usage::

    python experiment.py --out_dir=run_0          # full run with real API
    python experiment.py --out_dir=run_test --mock  # offline dry-run
    python experiment.py --out_dir=run_0 --config=alt_config.yaml  # alt config
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
# `components` is the only experimental axis Aider iterates. Allowed values:
# the strings "S3" .. "S12" (S1, S2 are fixed and always added). The Stage
# setting is fixed by the experimenter via config.yaml::experiment.stage and
# is intentionally NOT in this dict.
# ---------------------------------------------------------------------------

FORMAT_HYPOTHESIS = {
    "components": ["S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"],
    "instructions": "Upper-bound reference: all variable sections.",
}


def _resolve_config(arg_path: str | None) -> Path:
    if arg_path:
        return Path(arg_path).resolve()
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
        f"stage={config.experiment.stage} "
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
