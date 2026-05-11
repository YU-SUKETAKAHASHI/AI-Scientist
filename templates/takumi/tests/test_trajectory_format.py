"""lib/trajectory_format.py — flow-style + 低精度 trajectory レンダラのテスト."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.trajectory_format import (  # noqa: E402
    _round_strip,
    render_trajectory_bundle,
    render_trajectory_entries,
)


def test_round_strip_basic() -> None:
    assert _round_strip(1.234, 1) == "1.2"
    assert _round_strip(1.0, 1) == "1"
    assert _round_strip(0.0, 1) == "0"
    assert _round_strip(0, 1) == "0"
    assert _round_strip(1, 1) == "1"
    assert _round_strip(1.5, 1) == "1.5"
    assert _round_strip(-1.234, 1) == "-1.2"


def test_render_entries_single_line_per_entry() -> None:
    traj = [
        {"t": 0, "o": [1.234, 0.150, 0.0] + [0.0] * 9, "a": [0, 0, 0]},
        {"t": 10, "o": [0.5] * 12, "a": [1, 0, 1]},
    ]
    out = render_trajectory_entries(traj, precision=1)
    lines = out.split("\n")
    # 1 line per entry
    assert len(lines) == 2
    # flow-style braces
    assert "{t: 0" in lines[0]
    assert "o: [" in lines[0] and "a: [" in lines[0]
    # precision applied
    assert "1.2" in lines[0]
    assert "1.234" not in lines[0]


def test_render_entries_with_indent() -> None:
    traj = [{"t": 0, "o": [0.0] * 12, "a": [0, 0, 0]}]
    out = render_trajectory_entries(traj, indent="    ")
    assert out.startswith("    - {")


def test_render_bundle_has_summary_and_episode_blocks() -> None:
    bundle = {
        "summary": {"num_episodes": 1, "mean_reward": 1.5},
        "episodes": [
            {
                "episode": 0,
                "total_reward": 2.0,
                "episode_length": 3000,
                "trajectory": [
                    {"t": 0, "o": [0.0] * 12, "a": [0, 0, 0]},
                    {"t": 10, "o": [1.234] * 12, "a": [1, 1, 1]},
                ],
            },
        ],
    }
    out = render_trajectory_bundle(bundle, max_episodes=1, precision=1)
    assert "summary:" in out
    assert "episodes:" in out
    assert "trajectory:" in out
    # block-style for outer fields (one key per line)
    assert "  num_episodes: 1" in out
    # flow-style for trajectory entries
    assert "{t: 0" in out
    assert "{t: 10" in out
    # precision 1 dropped 1.234 → 1.2
    assert "1.2" in out
    assert "1.234" not in out


def test_render_bundle_caps_max_episodes() -> None:
    bundle = {
        "summary": {},
        "episodes": [
            {"episode": i, "total_reward": 0.0, "episode_length": 1,
             "trajectory": [{"t": 0, "o": [0.0] * 12, "a": [0, 0, 0]}]}
            for i in range(5)
        ],
    }
    out = render_trajectory_bundle(bundle, max_episodes=2, precision=1)
    assert out.count("- episode:") == 2


def test_render_bundle_compresses_real_data() -> None:
    """実 expert_trajectory.yaml で 50% 以上の圧縮を確認する."""
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    bundle = yaml.load(open(ROOT / "data" / "expert_trajectory.yaml"))
    new_size = len(render_trajectory_bundle(bundle, max_episodes=1, precision=1))
    # block-style baseline (旧フォーマット相当) は ~87,000 chars
    # 新フォーマットは 30,000 chars 前後
    assert new_size < 35_000, f"compressed render unexpectedly large: {new_size}"
