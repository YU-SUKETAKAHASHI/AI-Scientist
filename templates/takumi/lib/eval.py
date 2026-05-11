"""Inner Loop (§5) を 1 つ実行する高位 helper.

* :func:`run_inner_loop` — Skills format hypothesis 1 つを評価して
  ``final_info`` の means dict を返す.
* :func:`compute_final_info` — AI Scientist 規約の ``final_info.json``
  ペイロードに整形する.

Articulator の genome 可視性は ``format_hypothesis["show_genome"]: bool`` で
AI Scientist が制御する (旧 Stage 1/2 を 1 bit に統合).

baseline (A0 = run_0) との比較は AI Scientist 自身が chat history 上の
``run_0/final_info.json`` を参照して行う設計. eval.py 側で差分を pre-compute
しない (`delta_score_vs_a0` 等は提供しない).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .articulator import (
    ArticulationResult,
    articulate_skills,
    render_minimal_skills,
)
from .budget import BudgetTracker
from .config import TakumiConfig
from .mutator import (
    MutatorResult,
    propose_mutations,
    stage_label_from_history,
)


def _select_env(backend: str):
    """``config.env.backend`` に従って ``play_episode`` / ``play_n_episodes`` を返す.

    JAX 版は遅延 import (jax の重い初期化を gym 利用時に避ける).
    """
    backend = (backend or "jax").lower()
    if backend == "jax":
        from . import jax_env as _impl  # noqa: WPS433
    elif backend == "gym":
        from . import env_wrapper as _impl  # noqa: WPS433
    else:
        raise ValueError(f"unknown env backend: {backend!r}")
    return _impl.play_episode, _impl.play_n_episodes
from .network import (
    Network,
    init_minimal_network,
    network_to_policy,
    serialize_network,
)
from .visualize import (
    create_gameplay_gif,
    draw_takumi_network,
    plot_inner_loop_history,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: str | Path) -> Any:
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f)


def _count_active(network: Network) -> Tuple[int, int]:
    return len(network.nodes), len(network.edges)


@dataclass
class CycleRecord:
    """1 cycle の中間結果."""

    index: int
    score: float
    n_mutations: int
    network_nodes: int
    network_edges: int
    stage_label: str
    mutator_input_tokens: int = 0
    mutator_output_tokens: int = 0
    mutator_retries: int = 0
    reasoning: str = ""  # Mutator の reasoning を全文保持 (truncate しない)


# ---------------------------------------------------------------------------
# Mock mode (no LLM call) — for sanity/integration tests
# ---------------------------------------------------------------------------


def _mock_propose_mutations(
    network: Network,
    cycle_index: int,
    rng: np.random.Generator,
) -> Tuple[Network, List[Dict[str, Any]]]:
    """LLM 不在時の fallback: 簡素な weight 摂動で最低限の挙動を作る."""
    from .network import apply_mutations

    mutations: List[Dict[str, Any]] = []
    if cycle_index == 0:
        mutations.append(
            {"op": "change_weight", "from": "in_5", "to": "out_2",
             "new_weight": float(rng.uniform(0.4, 0.9))}
        )
        mutations.append(
            {"op": "change_weight", "from": "in_4", "to": "out_1",
             "new_weight": float(rng.uniform(-0.5, -0.1))}
        )
    else:
        if network.edges:
            e = network.edges[int(rng.integers(0, len(network.edges)))]
            mutations.append(
                {"op": "change_weight", "from": e["from"], "to": e["to"],
                 "new_weight": float(np.clip(e["weight"] + rng.normal(0, 0.3), -3.0, 3.0))}
            )
    return apply_mutations(network, mutations), mutations


# ---------------------------------------------------------------------------
# Inner loop
# ---------------------------------------------------------------------------


def run_inner_loop(
    format_hypothesis: Dict[str, Any],
    config: TakumiConfig,
    *,
    out_dir: str | Path,
    use_mock: bool = False,
    budget: Optional[BudgetTracker] = None,
) -> Dict[str, Any]:
    """1 つの Skills format hypothesis を評価し means dict を返す.

    Parameters
    ----------
    format_hypothesis : dict
        ``{"components": [...], "show_genome": <bool>, "instructions": <opt>}``.
        ``show_genome`` は AI Scientist が選ぶ 1 bit 軸 (default False).
    config : TakumiConfig
        実験者が編集する正本 (`templates/takumi/config.yaml`).
    out_dir : path
        skills.md / mutator log / cost_log.json などを書き出すディレクトリ.
    use_mock : bool, default False
        True で LLM call 抜きのドライラン (CI / 接続なし環境).
    budget : BudgetTracker or None
        コスト追跡. None なら config から閾値を読んで out_dir 配下に新規作成.

    Returns
    -------
    dict
        ``final_info`` の ``means`` の中身に相当する dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if budget is None:
        budget = BudgetTracker(
            out_dir=out_dir,
            warning_threshold=config.budget.warning_threshold_usd,
            hard_limit=config.budget.hard_limit_usd,
        )

    play_episode, play_n_episodes = _select_env(config.env.backend)

    expert_trajectory = _load_yaml(config.expert_trajectory_path())
    show_genome = bool(format_hypothesis.get("show_genome", False))
    expert_genome_dict: Optional[Dict[str, Any]] = None
    if show_genome:
        expert_genome_dict = _load_yaml(config.expert_genome_path())

    # ----- Step 1: Articulator (show_genome on/off) -----
    articulator_input_tokens = 0
    articulator_output_tokens = 0
    articulator_model = ""
    if use_mock:
        skills_md = render_minimal_skills(format_hypothesis, extra_note="mock-mode")
        components_used = [
            s.upper() for s in (format_hypothesis.get("components") or [])
        ]
        components_used = ["S1", "S2"] + [
            c for c in components_used if c not in {"S1", "S2"}
        ]
        articulation: Optional[ArticulationResult] = None
    else:
        articulation = articulate_skills(
            format_hypothesis,
            expert_trajectory,
            show_genome=show_genome,
            expert_genome=expert_genome_dict,
            model=config.models.articulator,
            budget=budget,
            max_tokens=config.articulator_params.max_tokens,
            temperature=config.articulator_params.temperature,
            max_retries=config.articulator_params.max_retries,
            trajectory_episodes=config.articulator_params.trajectory_episodes,
        )
        skills_md = articulation.skills_markdown
        components_used = articulation.components_used
        articulator_input_tokens = articulation.input_tokens
        articulator_output_tokens = articulation.output_tokens
        articulator_model = articulation.model

    skills_path = out_dir / "skills.md"
    skills_path.write_text(skills_md, encoding="utf-8")

    # ----- Step 2: Novice 暗黙層 init -----
    novice = init_minimal_network(seed=config.inner_loop.seed)
    serialize_network(novice, out_dir / "novice_init.yaml")
    init_n_nodes, init_n_edges = _count_active(novice)

    # ----- Visualization helpers (closure over out_dir + config) -----
    viz = config.visualization
    topology_dir = out_dir / "topology"
    if viz.enabled and viz.per_cycle_topology:
        topology_dir.mkdir(parents=True, exist_ok=True)

    def _maybe_save_topology(net: Network, *, cycle_idx: int, tag: str) -> None:
        """cycle ごとのネットワーク図を 1 枚保存する (closure)."""
        import matplotlib.pyplot as _plt  # noqa: WPS433
        if not (viz.enabled and viz.per_cycle_topology):
            return
        # cycle_idx == -1 は init を意味する
        fname = (
            "cycle_init.png" if cycle_idx < 0
            else f"cycle_{cycle_idx:03d}.png"
        )
        title = (
            f"Init network ({tag})" if cycle_idx < 0
            else f"Cycle {cycle_idx} — {tag}"
        )
        try:
            fig = draw_takumi_network(
                net,
                topology_dir / fname,
                title=title,
                dpi=viz.topology_dpi,
            )
            _plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            print(f"[viz] topology save failed for {fname}: {exc}")

    if viz.enabled and viz.include_init_topology:
        _maybe_save_topology(novice, cycle_idx=-1, tag="initial minimal network")

    # ----- Step 3: K cycles -----
    rng = np.random.default_rng(config.inner_loop.seed)
    cycle_scores: List[float] = []
    cycle_records: List[CycleRecord] = []
    best_cycle = 0
    best_network = novice
    best_score = float("-inf")
    mutator_calls = 0
    mutator_input_tokens = 0
    mutator_output_tokens = 0
    mutator_model = ""
    # mutator_strategy_summary 用に mutation を集計
    mutation_type_counter: Counter = Counter()
    edge_target_counter: Counter = Counter()

    K = config.inner_loop.K
    for cycle_idx in range(K):
        # a. Novice play 1 episode
        policy = network_to_policy(novice)
        ep_seed = config.inner_loop.seed * 1000 + cycle_idx
        # Inner cycle = training mode (3000-step 全プレイで多様な trajectory 収集)
        result = play_episode(
            policy,
            seed=ep_seed,
            subsample_interval=config.inner_loop.subsample_interval,
            max_steps=config.env.training_max_steps,
            test=False,
        )
        score = float(result["total_reward"])
        cycle_scores.append(score)
        if score > best_score:
            best_score = score
            best_cycle = cycle_idx
            best_network = novice  # snapshot BEFORE mutating

        # b. Stage label 判定 (S12 用)
        stage_label = stage_label_from_history(
            cycle_scores,
            novice_below=config.stage_label.novice_below,
            expert_above=config.stage_label.expert_above,
            window=config.stage_label.window,
        )

        # c. Mutator
        if use_mock:
            new_network, mutations = _mock_propose_mutations(novice, cycle_idx, rng)
            reasoning_full = "(mock)"
            in_tok = out_tok = retries = 0
        else:
            mres: MutatorResult = propose_mutations(
                novice,
                result["trajectory"],
                skills_md,
                cycle_score=score,
                cycle_index=cycle_idx,
                current_stage_label=stage_label,
                model=config.models.mutator,
                budget=budget,
                max_tokens=config.mutator_params.max_tokens,
                temperature=config.mutator_params.temperature,
                max_retries=config.mutator_params.max_retries,
                max_mutations_per_cycle=config.mutator_params.max_mutations_per_cycle,
            )
            new_network = mres.network
            mutations = mres.mutations
            reasoning_full = mres.reasoning or ""
            in_tok = mres.input_tokens
            out_tok = mres.output_tokens
            retries = mres.retries
            mutator_input_tokens += in_tok
            mutator_output_tokens += out_tok
            mutator_model = mres.model

        mutator_calls += 1
        # mutator_strategy_summary 用に各 op を集計
        for _m in mutations:
            op = _m.get("op", "unknown")
            mutation_type_counter[op] += 1
            if op == "change_weight":
                edge_str = f"{_m.get('from', '?')}->{_m.get('to', '?')}"
                edge_target_counter[edge_str] += 1
        n_nodes, n_edges = _count_active(new_network)
        # 各 cycle 後 (mutation 適用直後) のネットワークを可視化保存
        _maybe_save_topology(
            new_network,
            cycle_idx=cycle_idx,
            tag=f"score={score:+.1f} stage={stage_label}",
        )
        cycle_records.append(
            CycleRecord(
                index=cycle_idx,
                score=score,
                n_mutations=len(mutations),
                network_nodes=n_nodes,
                network_edges=n_edges,
                stage_label=stage_label,
                mutator_input_tokens=in_tok,
                mutator_output_tokens=out_tok,
                mutator_retries=retries,
                reasoning=reasoning_full,
            )
        )

        novice = new_network

    serialize_network(best_network, out_dir / "novice_best.yaml")
    serialize_network(novice, out_dir / "novice_final.yaml")

    # ----- Step 4: best_network で eval_episodes 評価 (test mode = 5-life) -----
    best_policy = network_to_policy(best_network)
    eval_result = play_n_episodes(
        best_policy,
        config.inner_loop.eval_episodes,
        base_seed=10_000,
        max_steps=config.env.test_max_steps,
        test=True,
    )

    # ----- Best Network のプレイ動画 (best_play.gif) -----
    if viz.enabled and viz.gameplay_gif:
        try:
            gif_result = create_gameplay_gif(
                best_network,
                save_path=out_dir / "best_play.gif",
                max_steps=viz.gif_max_steps,
                n_trials=viz.gif_n_trials,
                seed=viz.gif_seed,
            )
            print(
                f"[viz] best_play.gif: best_score={gif_result['best_score']:+.2f} "
                f"frames={gif_result['frames']}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[viz] gameplay_gif failed: {exc}")

    # ----- End-of-loop visualization -----
    if viz.enabled and viz.end_of_loop_curve:
        import matplotlib.pyplot as _plt  # noqa: WPS433
        try:
            curve_title = (
                f"{config.experiment.run_label} · "
                f"show_genome={show_genome} · "
                f"components={'+'.join(components_used) if components_used else 'S1+S2'}"
            )
            fig = plot_inner_loop_history(
                cycle_records,
                save_path=out_dir / "learning_curve.png",
                title=curve_title,
                dpi=viz.curve_dpi,
                eval_mean=float(eval_result["mean"]),
                eval_std=float(eval_result["std"]),
            )
            _plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            print(f"[viz] learning_curve save failed: {exc}")

    # ----- compute means -----
    final_n_nodes, final_n_edges = _count_active(novice)

    # mutator_strategy_summary (Option A: 構造化 raw)
    best_reasoning = cycle_records[best_cycle].reasoning if cycle_records else ""
    mutator_strategy_summary = {
        "mutation_counts": dict(mutation_type_counter),
        "topology_change": {
            "nodes_delta": int(final_n_nodes - init_n_nodes),
            "edges_delta": int(final_n_edges - init_n_edges),
        },
        "top_3_edge_targets": [e for e, _ in edge_target_counter.most_common(3)],
        "best_cycle_reasoning_excerpt": best_reasoning,
    }

    means: Dict[str, Any] = {
        "fitness_mean_100ep": float(eval_result["mean"]),
        "fitness_std_100ep": float(eval_result["std"]),
        "win_rate_at_best_cycle": float(eval_result["win_rate"]),
        "best_cycle_index": int(best_cycle),
        "learning_curve_scores": [float(s) for s in cycle_scores],
        "components_used": "+".join(components_used) if components_used else "S1+S2",
        "show_genome": bool(show_genome),
        "final_network_nodes": int(final_n_nodes),
        "final_network_links": int(final_n_edges),
        "mutator_strategy_summary": mutator_strategy_summary,
    }

    # 詳細ログ (final_info.json には含めない)
    debug_log = {
        "config_source": str(config.source_path) if config.source_path else None,
        "run_label": config.experiment.run_label,
        "seed": int(config.inner_loop.seed),
        "format_hypothesis": format_hypothesis,
        "components_used": components_used,
        "cycle_records": [vars(r) for r in cycle_records],
        "eval": {
            "mean": eval_result["mean"],
            "std": eval_result["std"],
            "win_rate": eval_result["win_rate"],
            "n": eval_result["n"],
        },
        "articulator": {
            "input_tokens": articulator_input_tokens,
            "output_tokens": articulator_output_tokens,
            "model": articulator_model,
        },
        "mutator": {
            "input_tokens": mutator_input_tokens,
            "output_tokens": mutator_output_tokens,
            "calls": mutator_calls,
            "model": mutator_model,
        },
        "budget_snapshot": budget.snapshot(),
        "use_mock": use_mock,
    }
    (out_dir / "inner_loop_debug.json").write_text(
        json.dumps(debug_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return means


def compute_final_info(means: Dict[str, Any]) -> Dict[str, Any]:
    """``final_info.json`` ペイロードに整形する."""
    return {"skills_format_eval": {"means": means}}
