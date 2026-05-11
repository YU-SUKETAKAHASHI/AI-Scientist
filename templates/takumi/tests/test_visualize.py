"""lib/visualize.py — PNG が生成されるかの最小テスト."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.network import (  # noqa: E402
    apply_mutations,
    init_minimal_network,
    load_network,
)
from lib.visualize import (  # noqa: E402
    draw_takumi_network,
    plot_inner_loop_history,
)


@dataclass
class _CR:
    index: int
    score: float
    n_mutations: int
    network_nodes: int
    network_edges: int
    stage_label: str


def test_draw_minimal_network(tmp_path: Path) -> None:
    net = init_minimal_network(seed=0)
    out = tmp_path / "net.png"
    fig = draw_takumi_network(net, out, title="test")
    assert out.exists() and out.stat().st_size > 1000
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_draw_with_hidden_node(tmp_path: Path) -> None:
    net = init_minimal_network(seed=0)
    net2 = apply_mutations(
        net,
        [
            {
                "op": "add_node",
                "new_id": "h_test",
                "activation": "tanh",
                "initial_edges": [
                    {"from": "in_5", "to": "h_test", "weight": 1.0},
                    {"from": "h_test", "to": "out_2", "weight": 0.5},
                ],
            }
        ],
    )
    out = tmp_path / "with_hidden.png"
    fig = draw_takumi_network(net2, out)
    assert out.exists()
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_draw_expert_with_bias(tmp_path: Path) -> None:
    expert = load_network(ROOT / "data" / "expert_genome.yaml")
    out = tmp_path / "expert.png"
    fig = draw_takumi_network(expert, out, title="expert")
    assert out.exists()
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_inner_loop_history(tmp_path: Path) -> None:
    records = [
        _CR(i, -5.0 + i, 1, 15 + i // 3, 36 + i, "novice" if i < 3 else "intermediate")
        for i in range(5)
    ]
    out = tmp_path / "curve.png"
    fig = plot_inner_loop_history(
        records, out, title="test run",
        eval_mean=-2.5, eval_std=0.8,
    )
    assert out.exists()
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_handles_empty_records(tmp_path: Path) -> None:
    out = tmp_path / "empty.png"
    fig = plot_inner_loop_history([], out)
    assert out.exists()
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_create_gameplay_gif_smoke(tmp_path: Path) -> None:
    """Expert Network で短い GIF が生成されるかの smoke test."""
    EVOJAX_PATH = ROOT.parents[2] / "ref_implementations" / "evojax"
    if not EVOJAX_PATH.exists():
        import pytest
        pytest.skip(f"EvoJAX clone not found at {EVOJAX_PATH}")
    from lib.network import load_network
    from lib.visualize import create_gameplay_gif

    expert = load_network(ROOT / "data" / "expert_genome.yaml")
    out = tmp_path / "play.gif"
    result = create_gameplay_gif(
        expert, out, max_steps=60, n_trials=2, seed=0,
    )
    assert out.exists() and out.stat().st_size > 1000
    assert result["save_path"] == str(out)
    assert result["frames"] >= 1
    assert len(result["all_scores"]) == 2


def test_inner_loop_creates_topology_dir(tmp_path: Path) -> None:
    """lib/eval.run_inner_loop の hook が PNG を出すか (mock)."""
    from lib.config import load_config
    from lib.eval import run_inner_loop

    cfg = load_config(ROOT / "config.yaml")
    out_dir = tmp_path / "viz_run"
    means = run_inner_loop(
        {"components": [], "instructions": "test"},
        cfg.with_overrides(inner_loop={"K": 2, "eval_episodes": 3}),
        out_dir=out_dir,
        use_mock=True,
    )
    assert (out_dir / "learning_curve.png").exists()
    topo = out_dir / "topology"
    assert topo.is_dir()
    pngs = sorted(topo.glob("*.png"))
    # init + 2 cycles
    assert len(pngs) >= 3
    assert means["n_cycles"] == 2
