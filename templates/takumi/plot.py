"""Visualization for the Takumi template.

Reads ``run_*/final_info.json`` files in the current working directory and
produces three PNGs::

    learning_curves.png        — per-cycle score for each run
    ablation_comparison.png    — fitness_mean_100ep bar chart per run
    genome_comparison.png      — show_genome=False vs True fitness when both
                                  are present for the same Skills components

This file is editable by Aider; rewrites should keep the three output paths
because perform_experiments.py expects ``plot.py`` to be re-runnable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_runs(root: Path) -> List[Tuple[str, Dict[str, object]]]:
    runs: List[Tuple[str, Dict[str, object]]] = []
    for d in sorted(root.glob("run_*")):
        info = d / "final_info.json"
        if not info.exists():
            continue
        try:
            payload = json.loads(info.read_text(encoding="utf-8"))
            means = payload.get("skills_format_eval", {}).get("means", {})
            if means:
                runs.append((d.name, means))
        except json.JSONDecodeError:
            continue
    return runs


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_learning_curves(runs: List[Tuple[str, Dict[str, object]]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, means in runs:
        curve = means.get("learning_curve_scores") or []
        if not curve:
            continue
        ax.plot(range(1, len(curve) + 1), curve, marker="o", label=name)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Cycle reward")
    ax.set_title("Inner-loop learning curves")
    ax.grid(True, alpha=0.3)
    if runs:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_ablation_bars(runs: List[Tuple[str, Dict[str, object]]], out: Path) -> None:
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [n for n, _ in runs]
    fitness = [float(m.get("fitness_mean_100ep", 0.0)) for _, m in runs]
    err = [float(m.get("fitness_std_100ep", 0.0)) for _, m in runs]
    ax.bar(names, fitness, yerr=err, capsize=4, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Run")
    ax.set_ylabel("fitness_mean_100ep")
    ax.set_title("Skills format ablation comparison")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_genome_comparison(runs: List[Tuple[str, Dict[str, object]]], out: Path) -> None:
    """show_genome=False vs True を同一 components で並べる paired bar chart."""
    by_components: Dict[str, Dict[bool, float]] = {}
    for _name, m in runs:
        comps = str(m.get("components_used", "S1+S2"))
        if "show_genome" not in m:
            continue  # legacy run (no show_genome key) — skip
        flag = bool(m["show_genome"])
        by_components.setdefault(comps, {})[flag] = float(
            m.get("fitness_mean_100ep", 0.0)
        )
    paired = {k: v for k, v in by_components.items() if False in v and True in v}
    if not paired:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(paired.keys())
    off = [paired[k][False] for k in labels]
    on = [paired[k][True] for k in labels]
    x = range(len(labels))
    width = 0.4
    ax.bar([i - width / 2 for i in x], off, width=width, label="show_genome=False")
    ax.bar([i + width / 2 for i in x], on, width=width, label="show_genome=True")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("fitness_mean_100ep")
    ax.set_title("Genome visibility effect (same Skills components)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    here = Path(".")
    runs = _load_runs(here)
    plot_learning_curves(runs, here / "learning_curves.png")
    plot_ablation_bars(runs, here / "ablation_comparison.png")
    plot_genome_comparison(runs, here / "genome_comparison.png")
    print(f"plot.py: wrote 3 figures from {len(runs)} runs")


if __name__ == "__main__":
    main()
