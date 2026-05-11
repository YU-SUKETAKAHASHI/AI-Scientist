"""SlimeVolleyGym 環境の薄い adapter.

* :func:`play_episode`  — policy で 1 episode を走らせ trajectory を収集
* :func:`play_n_episodes` — 100-episode 評価で reward 統計を返す

Trajectory format (compact YAML 想定)::

    - {t: 0, o: [...12 floats...], a: [0, 0, 0]}
    - {t: 10, o: [...], a: [...]}

Subsample (default N=10) が effective なのは ``trajectory_record`` の中身のみで、
sim 自体は毎 step 走る. 仕様書 §4.4 参照.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# slimevolleygym registers SlimeVolley-v0 on import
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gym  # noqa: E402
import slimevolleygym  # noqa: E402, F401


DEFAULT_SUBSAMPLE_INTERVAL = 10
DEFAULT_MAX_STEPS = 3000


def _make_env(opponent: str = "builtin", *, test: bool = False) -> gym.Env:
    """SlimeVolleyGym 環境を作る.

    Parameters
    ----------
    opponent : str
        現状 "builtin" (built-in baseline policy) のみ対応.
        将来 "heuristic" 等を追加する場合の hook を残してある.
    test : bool
        True で 5-life evaluation mode、False で 3000-step training mode.

    Returns
    -------
    gym.Env
        SlimeVolley の Env instance.
    """
    if opponent != "builtin":
        # Phase 1 では builtin opponent のみ. Fallback heuristic は後で別途追加可能.
        raise NotImplementedError(
            f"opponent={opponent!r} はまだ未実装 (builtin のみ対応)"
        )
    env_id = "SlimeVolley-v0"
    env = gym.make(env_id)
    # SlimeVolleyGym の env オブジェクトに直接 test 属性を渡せる場合は反映
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    if hasattr(base, "test"):
        base.test = test
    return env


def _reset(env: gym.Env, seed: Optional[int] = None) -> np.ndarray:
    """gym 0.21 / 0.26 両対応の reset."""
    try:
        out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _step(env: gym.Env, action: np.ndarray):
    """gym 0.21 (4-tuple) / 0.26 (5-tuple) 両対応の step."""
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, bool(done), info
    obs, reward, terminated, truncated, info = out
    return obs, reward, bool(terminated or truncated), info


def play_episode(
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    opponent: str = "builtin",
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: Optional[int] = None,
    subsample_interval: int = DEFAULT_SUBSAMPLE_INTERVAL,
    test: bool = False,
) -> Dict[str, Any]:
    """Policy で 1 episode を走らせる.

    Parameters
    ----------
    policy : Callable[[np.ndarray], np.ndarray]
        12 次元 obs から 3 次元 action を返す callable
        (:func:`lib.network.network_to_policy` の出力).
    opponent : str
        対戦相手. 現状 "builtin" のみ.
    max_steps : int, default 3000
        最大 step 数.
    seed : int or None
        env reset seed.
    subsample_interval : int, default 10
        trajectory に記録する step 間隔. ``1`` を指定すると全 step 記録される
        (token 消費が爆発するので注意、§4.4).
    test : bool, default False
        True で 5-life evaluation mode.

    Returns
    -------
    dict
        ``{"trajectory": [...], "total_reward": float, "episode_length": int}``.
    """
    env = _make_env(opponent=opponent, test=test)
    obs = _reset(env, seed=seed)
    obs = np.asarray(obs, dtype=np.float64)

    trajectory: List[Dict[str, Any]] = []
    total_reward = 0.0
    step = 0
    while step < max_steps:
        action = np.asarray(policy(obs), dtype=np.int_).reshape(-1)
        if action.shape[0] != 3:
            raise ValueError(f"policy must return 3-dim action, got {action.shape}")
        if step % subsample_interval == 0:
            trajectory.append(
                {
                    "t": int(step),
                    "o": [round(float(x), 3) for x in obs.tolist()],
                    "a": [int(x) for x in action.tolist()],
                }
            )
        obs, reward, done, _info = _step(env, action)
        obs = np.asarray(obs, dtype=np.float64)
        total_reward += float(reward)
        step += 1
        if done:
            break

    env.close()
    return {
        "trajectory": trajectory,
        "total_reward": float(total_reward),
        "episode_length": int(step),
    }


def play_n_episodes(
    policy: Callable[[np.ndarray], np.ndarray],
    n: int,
    *,
    opponent: str = "builtin",
    max_steps: int = DEFAULT_MAX_STEPS,
    base_seed: Optional[int] = 0,
    test: bool = True,
) -> Dict[str, Any]:
    """Policy で n episodes を play して reward 統計を返す (100-episode 評価用).

    Parameters
    ----------
    policy : Callable
        評価対象 policy.
    n : int
        episode 数.
    opponent : str
        対戦相手 ("builtin" のみ).
    max_steps : int, default 3000
        episode あたりの最大 step.
    base_seed : int or None
        base seed. episode i は ``base_seed + i`` で初期化される.
    test : bool, default True
        評価時は True (5-life mode).

    Returns
    -------
    dict
        ``{"rewards": List[float], "mean": float, "std": float, "n": int,
        "win_rate": float}``. win_rate は reward > 0 の episode 比率.
    """
    rewards: List[float] = []
    wins = 0
    for i in range(n):
        seed = None if base_seed is None else int(base_seed) + i
        result = play_episode(
            policy,
            opponent=opponent,
            max_steps=max_steps,
            seed=seed,
            subsample_interval=max_steps + 1,  # disable trajectory record (eval only)
            test=test,
        )
        r = float(result["total_reward"])
        rewards.append(r)
        if r > 0:
            wins += 1
    arr = np.asarray(rewards, dtype=np.float64)
    return {
        "rewards": rewards,
        "mean": float(arr.mean()) if rewards else 0.0,
        "std": float(arr.std()) if rewards else 0.0,
        "n": len(rewards),
        "win_rate": float(wins) / len(rewards) if rewards else 0.0,
    }
