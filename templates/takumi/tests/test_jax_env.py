"""lib/jax_env.py — EvoJAX SlimeVolley wrapper の sanity tests.

ref_implementations/evojax/ が見つからないと skip する.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# evojax のローカル clone があるかをまず調べる (無ければ全テスト skip)
# ROOT = templates/takumi/, parents: 0=templates 1=AI-Scientist 2=Part5
EVOJAX_PATH = (
    ROOT.parents[2]
    / "ref_implementations"
    / "evojax"
)
pytestmark = pytest.mark.skipif(
    not EVOJAX_PATH.exists(),
    reason=f"EvoJAX clone not found at {EVOJAX_PATH}",
)


def _random_policy(obs: np.ndarray) -> np.ndarray:
    return (np.random.RandomState(0).rand(3) > 0.5).astype(np.int_)


def test_play_episode_jax_runs() -> None:
    from lib.jax_env import play_episode  # noqa: WPS433

    result = play_episode(
        _random_policy, seed=0, max_steps=200, subsample_interval=10, test=False
    )
    assert "trajectory" in result
    assert isinstance(result["total_reward"], float)
    assert result["episode_length"] >= 1
    if result["trajectory"]:
        entry = result["trajectory"][0]
        assert set(entry.keys()) >= {"t", "o", "a"}
        assert len(entry["o"]) == 12
        assert len(entry["a"]) == 3


def test_play_n_episodes_jax_returns_stats() -> None:
    from lib.jax_env import play_n_episodes  # noqa: WPS433

    result = play_n_episodes(
        _random_policy, n=3, base_seed=0, max_steps=200, test=True, warmup=True
    )
    assert result["n"] == 3
    assert len(result["rewards"]) == 3
    assert isinstance(result["mean"], float)
    assert 0.0 <= result["win_rate"] <= 1.0


def test_eval_dispatch_selects_jax_backend() -> None:
    from lib.config import load_config  # noqa: WPS433
    from lib.eval import _select_env  # noqa: WPS433
    from lib import jax_env  # noqa: WPS433

    cfg = load_config(ROOT / "config.yaml")
    if cfg.env.backend != "jax":
        pytest.skip("default config does not select jax backend")
    play_ep, play_n = _select_env("jax")
    assert play_ep is jax_env.play_episode
    assert play_n is jax_env.play_n_episodes


def test_eval_dispatch_selects_gym_backend() -> None:
    from lib.eval import _select_env  # noqa: WPS433
    from lib import env_wrapper  # noqa: WPS433

    play_ep, play_n = _select_env("gym")
    assert play_ep is env_wrapper.play_episode
    assert play_n is env_wrapper.play_n_episodes
