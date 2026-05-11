"""SlimeVolley の JAX 実装に対する薄い adapter (Part 1 EvoJAX 版を流用).

設計方針
~~~~~~~~

* 既存の :mod:`lib.env_wrapper` (gym 版) と **同じ I/O 契約** を提供する.
* Env 自体は JAX (jax.lax + jax.jit + jax.vmap) で動く.
* Novice/Expert の policy callable は今のところ ``numpy`` のまま (graph 構造が
  cycle ごとに変わるため JIT 化は将来 work). 1 step ごとに jax→numpy→jax の
  境界変換が入るが、batch_size を増やせば env 側の vmap で速度メリットが出る.
* Test mode (5-life): 100-episode 評価で使う仕様書 §「ドメイン知識」.
* Training mode (3000-step 全プレイ): novice の cycle 内 trajectory 収集で使う.

EvoJAX のソースは ``ref_implementations/evojax/`` に同梱されているものを
``sys.path`` に追加して import する (`evojax` を pip install しなくても動く).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# EvoJAX SlimeVolley の lazy import
# ---------------------------------------------------------------------------

_EVOJAX_PATH_ENV = "TAKUMI_EVOJAX_PATH"
_DEFAULT_EVOJAX_PATH = (
    Path(__file__).resolve().parents[4]
    / "ref_implementations"
    / "evojax"
)


def _ensure_evojax_on_path() -> None:
    """EvoJAX のローカル clone を sys.path に積む (idempotent)."""
    custom = os.environ.get(_EVOJAX_PATH_ENV)
    candidate = Path(custom) if custom else _DEFAULT_EVOJAX_PATH
    candidate = candidate.resolve()
    if not candidate.exists():
        raise RuntimeError(
            f"EvoJAX path not found at {candidate}. "
            f"Set {_EVOJAX_PATH_ENV} env var to override."
        )
    p = str(candidate)
    if p not in sys.path:
        sys.path.insert(0, p)


def _import_jax_env():
    """``SlimeVolley``, ``jax``, ``jnp`` を返す (lazy import).

    Returns
    -------
    tuple
        ``(SlimeVolley, jax_module, jnp_module)``.
    """
    _ensure_evojax_on_path()
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    from evojax.task.slimevolley import SlimeVolley  # noqa: WPS433
    return SlimeVolley, jax, jnp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_task(*, max_steps: int, test: bool):
    SlimeVolley, _jax, _jnp = _import_jax_env()
    return SlimeVolley(max_steps=max_steps, test=test)


def _stack_keys(jax_mod, base_seed: int, n: int):
    keys = [jax_mod.random.PRNGKey(int(base_seed) + i) for i in range(n)]
    return jax_mod.numpy.stack(keys)


def _action_to_jax(action: np.ndarray, jnp_mod) -> Any:
    arr = np.asarray(action, dtype=np.float32).reshape(-1, 3)
    return jnp_mod.asarray(arr)


# ---------------------------------------------------------------------------
# Public API: matches lib.env_wrapper signatures where it makes sense
# ---------------------------------------------------------------------------

DEFAULT_SUBSAMPLE_INTERVAL = 10
DEFAULT_MAX_STEPS = 3000


def play_episode(
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    opponent: str = "builtin",
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: Optional[int] = None,
    subsample_interval: int = DEFAULT_SUBSAMPLE_INTERVAL,
    test: bool = False,
) -> Dict[str, Any]:
    """JAX SlimeVolley で 1 episode を走らせる (gym 版と同じ I/O).

    Parameters
    ----------
    policy : Callable[[np.ndarray], np.ndarray]
        12 次元 obs から 3 次元 binary action を返す callable.
    opponent : str
        EvoJAX 版は built-in baseline policy のみ. 互換のため引数だけ残す.
    max_steps : int, default 3000
        episode 最大 step.
    seed : int or None
        ``jax.random.PRNGKey`` の base. None なら 0.
    subsample_interval : int, default 10
        trajectory に記録する step 間隔.
    test : bool, default False
        True なら 5-life mode (life=0 で done), False なら 3000-step 全プレイ.

    Returns
    -------
    dict
        ``{"trajectory": [...], "total_reward": float, "episode_length": int}``.
    """
    if opponent != "builtin":
        raise NotImplementedError(
            f"opponent={opponent!r} はまだ未実装 (builtin baseline のみ)"
        )

    SlimeVolley, jax_mod, jnp = _import_jax_env()
    task = _make_task(max_steps=max_steps, test=test)

    base_seed = 0 if seed is None else int(seed)
    keys = _stack_keys(jax_mod, base_seed, 1)
    state = task.reset(keys)

    trajectory: List[Dict[str, Any]] = []
    total_reward = 0.0
    step = 0
    done_flag = False
    while step < max_steps and not done_flag:
        obs_np = np.asarray(state.obs[0], dtype=np.float64)
        action_np = np.asarray(policy(obs_np), dtype=np.int_).reshape(-1)
        if action_np.shape[0] != 3:
            raise ValueError(
                f"policy must return 3-dim action, got {action_np.shape}"
            )
        if step % subsample_interval == 0:
            trajectory.append(
                {
                    "t": int(step),
                    "o": [round(float(x), 3) for x in obs_np.tolist()],
                    "a": [int(x) for x in action_np.tolist()],
                }
            )
        action_jax = _action_to_jax(action_np[None, :], jnp)
        state, reward, done = task.step(state, action_jax)
        total_reward += float(np.asarray(reward[0]))
        done_flag = bool(np.asarray(done[0]))
        step += 1

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
    warmup: bool = True,
) -> Dict[str, Any]:
    """JAX SlimeVolley で n episodes を ``vmap`` 並列 評価する.

    100-episode 評価 (§5 Step 4) の主目的に最適化されている. policy は依然
    Python (numpy) なので各 step ごとに n 回 Python loop で呼ばれるが、env step
    自体は JAX 側でまとめて vmap される.

    Parameters
    ----------
    policy : Callable
        評価対象 policy.
    n : int
        並列に評価する episode 数.
    opponent : str
        現状 "builtin" のみ.
    max_steps : int, default 3000
        episode あたりの最大 step.
    base_seed : int or None
        episode i は ``jax.random.PRNGKey(base_seed + i)`` で初期化される.
    test : bool, default True
        True で 5-life mode (評価用).
    warmup : bool, default True
        最初に小さな batch で JIT compile を済ませてから時計を止める. eval の
        計測に使うときに有用.

    Returns
    -------
    dict
        ``{"rewards": list, "mean": float, "std": float, "n": int,
        "win_rate": float}``. 各 episode の reward は最初に done になった
        時点で凍結される (subsequent rewards を 0 として加算しない).
    """
    if opponent != "builtin":
        raise NotImplementedError(
            f"opponent={opponent!r} はまだ未実装 (builtin baseline のみ)"
        )
    if n <= 0:
        return {"rewards": [], "mean": 0.0, "std": 0.0, "n": 0, "win_rate": 0.0}

    SlimeVolley, jax_mod, jnp = _import_jax_env()
    task = _make_task(max_steps=max_steps, test=test)

    if warmup:
        # 1-batch の dummy step で JIT compile を済ませる
        dummy_keys = _stack_keys(jax_mod, 0, 1)
        s = task.reset(dummy_keys)
        s, _, _ = task.step(s, jnp.zeros((1, 3), dtype=jnp.float32))
        del s

    seed_base = 0 if base_seed is None else int(base_seed)
    keys = _stack_keys(jax_mod, seed_base, n)
    state = task.reset(keys)

    total_rewards = jnp.zeros(n, dtype=jnp.float32)
    done_mask = jnp.zeros(n, dtype=bool)

    for _step in range(max_steps):
        obs_batch = np.asarray(state.obs)  # (n, 12)
        actions_np = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            a = np.asarray(policy(obs_batch[i]), dtype=np.float32).reshape(-1)
            if a.shape[0] != 3:
                raise ValueError(
                    f"policy returned {a.shape}, expected (3,)"
                )
            actions_np[i] = a
        action_jax = jnp.asarray(actions_np)
        state, reward, done = task.step(state, action_jax)
        # 既に done になった env の reward は無視 (環境は auto-reset するが
        # 統計的に "1 episode の合計" を保つため)
        total_rewards = total_rewards + jnp.where(done_mask, 0.0, reward)
        done_mask = jnp.logical_or(done_mask, done)
        if bool(np.asarray(jnp.all(done_mask))):
            break

    rewards = [float(x) for x in np.asarray(total_rewards)]
    arr = np.asarray(rewards, dtype=np.float64)
    wins = int(np.sum(arr > 0))
    return {
        "rewards": rewards,
        "mean": float(arr.mean()) if rewards else 0.0,
        "std": float(arr.std()) if rewards else 0.0,
        "n": len(rewards),
        "win_rate": float(wins) / len(rewards) if rewards else 0.0,
    }
