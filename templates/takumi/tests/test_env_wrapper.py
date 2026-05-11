"""lib/env_wrapper.py — random policy で 1 episode が走ることを確認."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.env_wrapper import play_episode  # noqa: E402


def _random_policy(obs: np.ndarray) -> np.ndarray:
    return (np.random.RandomState(0).rand(3) > 0.5).astype(np.int_)


def test_play_episode_runs() -> None:
    result = play_episode(_random_policy, seed=0, max_steps=100, subsample_interval=10)
    assert "trajectory" in result
    assert isinstance(result["total_reward"], float)
    assert result["episode_length"] >= 1
    if result["trajectory"]:
        entry = result["trajectory"][0]
        assert set(entry.keys()) >= {"t", "o", "a"}
        assert len(entry["o"]) == 12
        assert len(entry["a"]) == 3
