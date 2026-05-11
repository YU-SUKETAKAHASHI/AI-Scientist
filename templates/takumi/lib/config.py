"""``config.yaml`` の型付き読込.

実験者が触る正本は ``templates/takumi/config.yaml``. このモジュールは:

* :class:`TakumiConfig` — frozen dataclass で全設定を保持
* :func:`load_config` — YAML → :class:`TakumiConfig`
* バリデーション (K > 0, env backend, budget 整合性, ...)

を提供する. Aider 編集対象外 (Day 3 以降の outer loop で書き換わらない).

Note (2026-05-11): Stage 概念 (`experiment.stage`) は廃止された.
Articulator の genome 可視性は `FORMAT_HYPOTHESIS["show_genome"]: bool`
で AI Scientist が制御する.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

from ruamel.yaml import YAML


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentSection:
    """``experiment:`` ブロック (実験者が固定する run-level label).

    Stage 概念は廃止. Articulator の genome 可視性は
    `FORMAT_HYPOTHESIS["show_genome"]` で AI Scientist が選ぶ.
    """

    run_label: str = "default"


@dataclass(frozen=True)
class EnvSection:
    """``env:`` ブロック (env backend の選択)."""

    backend: str  # "jax" or "gym"
    training_max_steps: int
    test_max_steps: int


@dataclass(frozen=True)
class InnerLoopSection:
    """``inner_loop:`` ブロック (§5 のパラメータ)."""

    K: int
    eval_episodes: int
    seed: int
    subsample_interval: int
    max_steps_per_episode: int


@dataclass(frozen=True)
class ModelsSection:
    """``models:`` ブロック."""

    articulator: str
    mutator: str


@dataclass(frozen=True)
class ArticulatorParams:
    """``articulator_params:`` ブロック."""

    temperature: float
    max_tokens: int
    max_retries: int
    trajectory_episodes: int


@dataclass(frozen=True)
class MutatorParams:
    """``mutator_params:`` ブロック."""

    temperature: float
    max_tokens: int
    max_retries: int
    max_mutations_per_cycle: int


@dataclass(frozen=True)
class StageLabelParams:
    """``stage_label:`` ブロック (S12 用 dynamic 判定)."""

    novice_below: float
    expert_above: float
    window: int


@dataclass(frozen=True)
class VisualizationSection:
    """``visualization:`` ブロック."""

    enabled: bool
    per_cycle_topology: bool
    include_init_topology: bool
    end_of_loop_curve: bool
    gameplay_gif: bool
    gif_max_steps: int
    gif_n_trials: int
    gif_seed: int
    topology_dpi: int
    curve_dpi: int


@dataclass(frozen=True)
class BudgetSection:
    """``budget:`` ブロック."""

    warning_threshold_usd: float
    hard_limit_usd: float


@dataclass(frozen=True)
class DataPathsSection:
    """``data:`` ブロック (相対パス文字列を保持)."""

    expert_trajectory: str
    expert_genome: str


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TakumiConfig:
    """templates/takumi/config.yaml のメモリ上表現.

    実験者が変えるのはこの dataclass の field. Aider が触るのは
    `experiment.py` の `FORMAT_HYPOTHESIS` のみで、本 config には触らない.
    """

    experiment: ExperimentSection
    env: EnvSection
    inner_loop: InnerLoopSection
    models: ModelsSection
    articulator_params: ArticulatorParams
    mutator_params: MutatorParams
    stage_label: StageLabelParams
    visualization: VisualizationSection
    budget: BudgetSection
    data: DataPathsSection
    source_path: Optional[Path] = field(default=None, compare=False)

    # ------------------------------------------------------------------
    # Convenience accessors (副作用なし)
    # ------------------------------------------------------------------

    def expert_trajectory_path(self) -> Path:
        """templates/takumi/ 起点の絶対パスを返す."""
        return self._resolve(self.data.expert_trajectory)

    def expert_genome_path(self) -> Path:
        return self._resolve(self.data.expert_genome)

    def _resolve(self, rel: str) -> Path:
        if self.source_path is None:
            return Path(rel)
        return (self.source_path.parent / rel).resolve()

    def with_overrides(self, **kwargs: Any) -> "TakumiConfig":
        """テスト用: 一部 field を上書きしたコピーを返す.

        受け付ける形式は ``experiment={"run_label": "alt"}`` のように section 単位.
        """
        out = self
        for section_name, override in kwargs.items():
            if not hasattr(out, section_name):
                raise ValueError(f"unknown section: {section_name}")
            current = getattr(out, section_name)
            if not isinstance(override, dict):
                raise ValueError(
                    f"override for {section_name!r} must be a dict, got {override!r}"
                )
            new_section = replace(current, **override)
            out = replace(out, **{section_name: new_section})
        return out


# ---------------------------------------------------------------------------
# Loader / validator
# ---------------------------------------------------------------------------


_REQUIRED_SECTIONS = (
    "experiment",
    "env",
    "inner_loop",
    "models",
    "articulator_params",
    "mutator_params",
    "stage_label",
    "visualization",
    "budget",
    "data",
)


_ALLOWED_ENV_BACKENDS = ("jax", "gym")


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ValueError(f"config: missing field {key!r} in section {ctx!r}")
    return d[key]


def _build(payload: Dict[str, Any], source_path: Optional[Path]) -> TakumiConfig:
    for sec in _REQUIRED_SECTIONS:
        if sec not in payload:
            raise ValueError(f"config: missing section {sec!r}")

    exp_d = payload["experiment"]
    experiment = ExperimentSection(
        run_label=str(exp_d.get("run_label", "default")),
    )

    env_d = payload["env"]
    backend = str(_require(env_d, "backend", "env")).lower()
    if backend not in _ALLOWED_ENV_BACKENDS:
        raise ValueError(
            f"config.env.backend must be one of {_ALLOWED_ENV_BACKENDS}, "
            f"got {backend!r}"
        )
    env_section = EnvSection(
        backend=backend,
        training_max_steps=int(_require(env_d, "training_max_steps", "env")),
        test_max_steps=int(_require(env_d, "test_max_steps", "env")),
    )

    il_d = payload["inner_loop"]
    K = int(_require(il_d, "K", "inner_loop"))
    eval_eps = int(_require(il_d, "eval_episodes", "inner_loop"))
    if K <= 0:
        raise ValueError("config.inner_loop.K must be > 0")
    if eval_eps <= 0:
        raise ValueError("config.inner_loop.eval_episodes must be > 0")
    sub = int(_require(il_d, "subsample_interval", "inner_loop"))
    if sub < 1:
        raise ValueError("config.inner_loop.subsample_interval must be >= 1")
    inner_loop = InnerLoopSection(
        K=K,
        eval_episodes=eval_eps,
        seed=int(_require(il_d, "seed", "inner_loop")),
        subsample_interval=sub,
        max_steps_per_episode=int(_require(il_d, "max_steps_per_episode", "inner_loop")),
    )

    md = payload["models"]
    models = ModelsSection(
        articulator=str(_require(md, "articulator", "models")),
        mutator=str(_require(md, "mutator", "models")),
    )

    ap = payload["articulator_params"]
    articulator_params = ArticulatorParams(
        temperature=float(_require(ap, "temperature", "articulator_params")),
        max_tokens=int(_require(ap, "max_tokens", "articulator_params")),
        max_retries=int(_require(ap, "max_retries", "articulator_params")),
        trajectory_episodes=int(_require(ap, "trajectory_episodes", "articulator_params")),
    )

    mp = payload["mutator_params"]
    mutator_params = MutatorParams(
        temperature=float(_require(mp, "temperature", "mutator_params")),
        max_tokens=int(_require(mp, "max_tokens", "mutator_params")),
        max_retries=int(_require(mp, "max_retries", "mutator_params")),
        max_mutations_per_cycle=int(
            _require(mp, "max_mutations_per_cycle", "mutator_params")
        ),
    )

    sl = payload["stage_label"]
    stage_label = StageLabelParams(
        novice_below=float(_require(sl, "novice_below", "stage_label")),
        expert_above=float(_require(sl, "expert_above", "stage_label")),
        window=int(_require(sl, "window", "stage_label")),
    )
    if stage_label.novice_below >= stage_label.expert_above:
        raise ValueError(
            "config.stage_label: novice_below must be < expert_above"
        )

    vd = payload["visualization"]
    visualization = VisualizationSection(
        enabled=bool(_require(vd, "enabled", "visualization")),
        per_cycle_topology=bool(_require(vd, "per_cycle_topology", "visualization")),
        include_init_topology=bool(
            _require(vd, "include_init_topology", "visualization")
        ),
        end_of_loop_curve=bool(_require(vd, "end_of_loop_curve", "visualization")),
        # gameplay_gif の各 field は backward-compat のため default 値を持たせる
        gameplay_gif=bool(vd.get("gameplay_gif", False)),
        gif_max_steps=int(vd.get("gif_max_steps", 1000)),
        gif_n_trials=int(vd.get("gif_n_trials", 5)),
        gif_seed=int(vd.get("gif_seed", 0)),
        topology_dpi=int(_require(vd, "topology_dpi", "visualization")),
        curve_dpi=int(_require(vd, "curve_dpi", "visualization")),
    )

    bd = payload["budget"]
    budget = BudgetSection(
        warning_threshold_usd=float(_require(bd, "warning_threshold_usd", "budget")),
        hard_limit_usd=float(_require(bd, "hard_limit_usd", "budget")),
    )
    if budget.warning_threshold_usd > budget.hard_limit_usd:
        raise ValueError(
            "config.budget: warning_threshold_usd must be <= hard_limit_usd"
        )

    dt = payload["data"]
    data = DataPathsSection(
        expert_trajectory=str(_require(dt, "expert_trajectory", "data")),
        expert_genome=str(_require(dt, "expert_genome", "data")),
    )

    return TakumiConfig(
        experiment=experiment,
        env=env_section,
        inner_loop=inner_loop,
        models=models,
        articulator_params=articulator_params,
        mutator_params=mutator_params,
        stage_label=stage_label,
        visualization=visualization,
        budget=budget,
        data=data,
        source_path=source_path,
    )


def load_config(path: str | Path) -> TakumiConfig:
    """``config.yaml`` を読み込んで :class:`TakumiConfig` を返す.

    Parameters
    ----------
    path : str or Path
        config ファイルパス. 相対パス (data: section) はこのファイルの
        ディレクトリを起点に解決される.

    Returns
    -------
    TakumiConfig
        バリデーション済み frozen dataclass.

    Raises
    ------
    ValueError
        必須 section/field の欠落、型エラー、整合性違反.
    """
    p = Path(path).resolve()
    yaml = YAML(typ="safe")
    with open(p, "r", encoding="utf-8") as f:
        payload = yaml.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"config: top-level YAML must be a mapping, got {type(payload).__name__}")
    return _build(payload, source_path=p)


def default_config_path(template_dir: Path) -> Path:
    """templates/takumi/config.yaml への canonical なパス."""
    return Path(template_dir) / "config.yaml"
