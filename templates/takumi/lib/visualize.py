"""Takumi network topology + inner loop fitness の可視化.

``ref_implementations/neat/visualize.py`` の :func:`draw_network` /
:func:`plot_fitness_history` / :func:`create_gameplay_gif` を Takumi の
:class:`Network` (YAML schema) に移植したもの. NEAT の int node-id ではなく、
本研究の文字列 id (``in_0`` .. ``in_11`` / ``out_0`` .. ``out_2`` / ``h_*`` /
``bias``) を直接扱う.

Public API
~~~~~~~~~~

* :func:`draw_takumi_network` — 1 つの :class:`Network` を PNG に描画
* :func:`plot_inner_loop_history` — K cycles の学習曲線を 1 枚で出力
* :func:`create_gameplay_gif` — Best Network のプレイ動画を GIF に保存
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from .network import (  # noqa: E402
    BIAS_ID,
    INPUT_IDS,
    NUM_INPUTS,
    NUM_OUTPUTS,
    Network,
    OUTPUT_IDS,
    _topological_order,
)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

# SlimeVolley input order (slimevolleygym / EvoJAX 共通): agent → ball → opp
_DEFAULT_INPUT_LABELS: Sequence[str] = (
    "x", "y", "vx", "vy",
    "bx", "by", "bvx", "bvy",
    "ox", "oy", "ovx", "ovy",
)
_DEFAULT_OUTPUT_LABELS: Sequence[str] = ("left", "right", "jump")


def _node_label(
    node: Dict[str, Any],
    *,
    input_labels: Sequence[str],
    output_labels: Sequence[str],
) -> str:
    """ノードの表示ラベルを返す."""
    nid = node["id"]
    ntype = node.get("type")
    if ntype == "bias":
        return "B"
    if ntype == "input":
        if nid in INPUT_IDS:
            idx = INPUT_IDS.index(nid)
            if idx < len(input_labels):
                return input_labels[idx]
        return nid
    if ntype == "output":
        if nid in OUTPUT_IDS:
            idx = OUTPUT_IDS.index(nid)
            if idx < len(output_labels):
                return output_labels[idx]
        return nid
    # hidden node — show its activation
    return str(node.get("activation", "h"))


# ---------------------------------------------------------------------------
# Layer assignment
# ---------------------------------------------------------------------------


def _hidden_layers(
    network: Network,
) -> Tuple[Dict[str, int], int]:
    """Topological 順序を使って hidden node のレイヤー番号を計算する.

    入力/bias は layer=0、出力は最後のレイヤー、hidden は接続の深さで決定.

    Returns
    -------
    tuple
        ``(layer_of_node, n_hidden_layers)``. ``layer_of_node`` は dict[id,int].
        ``n_hidden_layers`` は >=1 (hidden が無くても出力レイヤー用に >=1).
    """
    nodes_by_id = {n["id"]: n for n in network.nodes}
    in_edges: Dict[str, List[str]] = {n["id"]: [] for n in network.nodes}
    for e in network.edges:
        in_edges[e["to"]].append(e["from"])

    order = _topological_order(network)
    layer: Dict[str, int] = {}
    for nid in order:
        ntype = nodes_by_id[nid]["type"]
        if ntype in ("input", "bias"):
            layer[nid] = 0
        elif ntype == "hidden":
            preds = in_edges[nid]
            if not preds:
                layer[nid] = 1
            else:
                layer[nid] = max(layer[p] for p in preds) + 1
        # output は後で一律に max+1 にする
    hidden_layers = [
        layer[nid] for nid in layer
        if nodes_by_id[nid]["type"] == "hidden"
    ]
    max_h = max(hidden_layers) if hidden_layers else 0
    out_layer = max_h + 1
    for nid in order:
        if nodes_by_id[nid]["type"] == "output":
            layer[nid] = out_layer
    return layer, out_layer


# ---------------------------------------------------------------------------
# Position assignment
# ---------------------------------------------------------------------------


def _node_positions(
    network: Network,
    layer: Dict[str, int],
    out_layer: int,
) -> Dict[str, Tuple[float, float]]:
    """各 node の (x, y) 座標を返す.

    Inputs は左で 2 列の千鳥配置、bias は inputs の上、出力は右、hidden は
    レイヤーごとに縦並び.
    """
    pos: Dict[str, Tuple[float, float]] = {}
    nodes_by_id = {n["id"]: n for n in network.nodes}

    # Layer x scale
    x_scale = 10.0 / max(out_layer, 1)

    # ---- Input column (2-col stagger) ----
    n_input_nodes = NUM_INPUTS
    n_rows = (n_input_nodes + 1) // 2  # = 6
    input_spacing_y = 1.2
    input_col_offset = 0.6
    stagger = input_spacing_y * 0.4
    for j, iid in enumerate(INPUT_IDS):
        if iid not in nodes_by_id:
            continue
        row = j // 2
        col = j % 2
        x = -input_col_offset / 2 + col * input_col_offset
        y = row * input_spacing_y - (n_rows - 1) * input_spacing_y / 2
        if col == 1:
            y += stagger
        pos[iid] = (x, -y)

    # ---- Bias above the input column ----
    if BIAS_ID in nodes_by_id:
        bias_y = -(n_rows + 1) * input_spacing_y / 2
        pos[BIAS_ID] = (0.0, bias_y)

    # ---- Hidden + Output by layer ----
    layer_buckets: Dict[int, List[str]] = {}
    for nid, n in nodes_by_id.items():
        if n["type"] in ("input", "bias"):
            continue
        layer_buckets.setdefault(layer[nid], []).append(nid)

    node_spacing_y = 1.5
    for lid, members in layer_buckets.items():
        members_sorted = sorted(members)
        total = len(members_sorted)
        for i, nid in enumerate(members_sorted):
            x = lid * x_scale
            y = (i - (total - 1) / 2) * node_spacing_y
            pos[nid] = (x, -y)
    return pos


# ---------------------------------------------------------------------------
# Public: draw a single network
# ---------------------------------------------------------------------------


def draw_takumi_network(
    network: Network,
    save_path: Optional[str | Path] = None,
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 7),
    input_labels: Sequence[str] = _DEFAULT_INPUT_LABELS,
    output_labels: Sequence[str] = _DEFAULT_OUTPUT_LABELS,
    dpi: int = 120,
) -> plt.Figure:
    """Takumi YAML network を 1 枚の PNG に描画する.

    Parameters
    ----------
    network : Network
        対象 network.
    save_path : str or Path or None
        出力先. None なら保存せず Figure のみ返す.
    title : str or None
        図のタイトル.
    figsize : tuple
        サイズ.
    input_labels : sequence of str
        12 input の表示ラベル. デフォルトは ``[x, y, vx, vy, bx, by, bvx, bvy,
        ox, oy, ovx, ovy]`` で SlimeVolley 12 obs に対応.
    output_labels : sequence of str
        3 output の表示ラベル.
    dpi : int
        出力 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    layer, out_layer = _hidden_layers(network)
    pos = _node_positions(network, layer, out_layer)
    nodes_by_id = {n["id"]: n for n in network.nodes}

    g = nx.DiGraph()
    for nid in nodes_by_id:
        g.add_node(nid)
    edges_with_w: List[Tuple[str, str, float]] = []
    for e in network.edges:
        w = float(e["weight"])
        edges_with_w.append((e["from"], e["to"], w))
        g.add_edge(e["from"], e["to"], weight=w)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Edges (color by sign, width/alpha by magnitude relative to max)
    if edges_with_w:
        max_w = max(abs(w) for _, _, w in edges_with_w) or 1.0
        for src, dst, w in edges_with_w:
            if src not in pos or dst not in pos:
                continue
            color = "salmon" if w > 0 else "steelblue"
            width = max(0.4, abs(w) / max_w * 3.0)
            alpha = max(0.25, min(1.0, abs(w) / max_w))
            ax.annotate(
                "",
                xy=pos[dst], xytext=pos[src],
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=width,
                    alpha=alpha,
                    connectionstyle="arc3,rad=0.1",
                    shrinkA=10, shrinkB=10,
                ),
            )

    # Nodes (color by type)
    node_colors: List[str] = []
    node_labels_disp: Dict[str, str] = {}
    for nid in g.nodes():
        n = nodes_by_id[nid]
        ntype = n["type"]
        if ntype == "bias":
            node_colors.append("lightgray")
        elif ntype == "input":
            node_colors.append("lightblue")
        elif ntype == "output":
            node_colors.append("lightcoral")
        else:
            node_colors.append("lightgreen")
        node_labels_disp[nid] = _node_label(
            n, input_labels=input_labels, output_labels=output_labels
        )

    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=node_colors,
        node_size=1800, edgecolors="black", linewidths=1.0,
    )
    nx.draw_networkx_labels(
        g, pos, ax=ax, labels=node_labels_disp,
        font_size=11, font_weight="bold",
    )

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color="salmon", lw=2, label="Positive weight"),
        Line2D([0], [0], color="steelblue", lw=2, label="Negative weight"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)

    n_nodes = len(network.nodes)
    n_edges = len(network.edges)
    n_hidden = sum(1 for n in network.nodes if n.get("type") == "hidden")
    subtitle = f"{n_nodes} nodes ({n_hidden} hidden) · {n_edges} edges"
    if title:
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    else:
        ax.set_title(subtitle, fontsize=11)
    ax.axis("off")
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Public: end-of-inner-loop history plot
# ---------------------------------------------------------------------------


def plot_inner_loop_history(
    cycle_records: Iterable[Any],
    save_path: Optional[str | Path] = None,
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    dpi: int = 140,
    eval_mean: Optional[float] = None,
    eval_std: Optional[float] = None,
) -> plt.Figure:
    """Inner loop 1 回分 (K cycles) の学習曲線 + ネットワーク複雑度を描画する.

    Parameters
    ----------
    cycle_records : iterable
        :class:`lib.eval.CycleRecord` のリスト (もしくは互換 dict). 各要素は
        ``index, score, n_mutations, network_nodes, network_edges, stage_label``
        のフィールドを持つ.
    save_path : str or Path or None
        保存先.
    title : str or None
        タイトル.
    figsize : tuple
        図サイズ.
    dpi : int
        出力 dpi.
    eval_mean / eval_std : float or None
        最終 100-episode 評価の mean / std. 渡すと右側パネルに水平線として描画.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cycles: List[int] = []
    scores: List[float] = []
    n_nodes_list: List[int] = []
    n_edges_list: List[int] = []
    n_mut_list: List[int] = []
    stage_labels: List[str] = []
    for r in cycle_records:
        if hasattr(r, "index"):
            cycles.append(int(r.index))
            scores.append(float(r.score))
            n_nodes_list.append(int(r.network_nodes))
            n_edges_list.append(int(r.network_edges))
            n_mut_list.append(int(r.n_mutations))
            stage_labels.append(str(r.stage_label))
        else:
            cycles.append(int(r["index"]))
            scores.append(float(r["score"]))
            n_nodes_list.append(int(r["network_nodes"]))
            n_edges_list.append(int(r["network_edges"]))
            n_mut_list.append(int(r["n_mutations"]))
            stage_labels.append(str(r["stage_label"]))

    if not cycles:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No cycle records", ha="center", va="center")
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)

    # ---- Left: cycle reward + best running ----
    best_running: List[float] = []
    cur = float("-inf")
    for s in scores:
        cur = max(cur, s)
        best_running.append(cur)

    ax_left.plot(cycles, scores, "o-", label="cycle reward", color="tab:blue", alpha=0.75)
    ax_left.plot(cycles, best_running, "-", label="best so far",
                 color="tab:red", linewidth=2)

    # stage_label を色で重ねる
    cmap = {"novice": "tab:gray", "intermediate": "tab:orange", "expert": "tab:green"}
    prev_label = None
    for i, lbl in enumerate(stage_labels):
        c = cmap.get(lbl, "tab:gray")
        ax_left.scatter(cycles[i], scores[i], c=[c], s=40, zorder=5,
                        edgecolors="black", linewidths=0.5)
        if lbl != prev_label:
            ax_left.annotate(
                lbl, (cycles[i], scores[i]), fontsize=7, ha="center",
                va="bottom", xytext=(0, 8), textcoords="offset points",
                color=c,
            )
            prev_label = lbl

    if eval_mean is not None:
        ax_left.axhline(eval_mean, color="purple", linestyle="--",
                        linewidth=1.5, alpha=0.7,
                        label=f"100-ep eval mean = {eval_mean:+.2f}")
        if eval_std is not None:
            ax_left.fill_between(
                [min(cycles), max(cycles)],
                eval_mean - eval_std, eval_mean + eval_std,
                color="purple", alpha=0.1, label="±1σ"
            )

    ax_left.set_xlabel("Cycle")
    ax_left.set_ylabel("Reward")
    ax_left.set_title("Inner-loop fitness trajectory")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(fontsize=9, loc="best")

    # ---- Right: network complexity + mutations / cycle ----
    ax_right.plot(cycles, n_nodes_list, "o-", color="tab:green", label="nodes")
    ax_right.plot(cycles, n_edges_list, "s-", color="tab:olive", label="edges")
    ax_right.set_xlabel("Cycle")
    ax_right.set_ylabel("Network size")
    ax_right.set_title("Complexity & mutator activity")
    ax_right.grid(True, alpha=0.3)

    ax_right2 = ax_right.twinx()
    ax_right2.bar(cycles, n_mut_list, alpha=0.25, color="tab:purple",
                  label="mutations applied")
    ax_right2.set_ylabel("# mutations / cycle", color="tab:purple")
    ax_right2.tick_params(axis="y", labelcolor="tab:purple")

    lines1, labels1 = ax_right.get_legend_handles_labels()
    lines2, labels2 = ax_right2.get_legend_handles_labels()
    ax_right.legend(lines1 + lines2, labels1 + labels2,
                    loc="upper left", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Public: gameplay GIF (best Network のプレイ動画)
# ---------------------------------------------------------------------------


def create_gameplay_gif(
    network: Network,
    save_path: str | Path,
    *,
    max_steps: int = 1000,
    seed: int = 0,
    n_trials: int = 5,
    duration_ms: int = 33,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Best Network のプレイ動画を GIF として保存する.

    実装は ``ref_implementations/neat/visualize.py:create_gameplay_gif`` を
    Takumi YAML schema に移植したもの. 内部で 2 phase に分けて実行する:

    1. **Phase 1 — score 集計**: ``n_trials`` 試合を vmap で並列に走らせ、
       各 trial の総得点を集計する. 最高スコアの trial idx を best_idx とする.
    2. **Phase 2 — 録画**: 同じ ``trial_keys`` でもう一度 reset し、毎 step で
       ``best_idx`` の game state だけを ``Game.display()`` でレンダリングする.

    Phase 1 と Phase 2 で **同じ ``n_trials`` バッチサイズで走らせる** 必要が
    ある (XLA の浮動小数点丸めで軌道が発散する事があるため、ref 実装の
    コメント参照).

    Parameters
    ----------
    network : Network
        対象 network (典型的には ``run_*/novice_best.yaml`` の load 結果).
    save_path : str or Path
        GIF の出力先.
    max_steps : int, default 1000
        episode 最大 step (短いほど GIF サイズ小).
    seed : int, default 0
        ``jax.random.PRNGKey`` の base. 試合 key は ``jax.random.split`` で
        ``n_trials`` 個派生.
    n_trials : int, default 5
        並列に走らせる試合数. 最高 score の試合を録画する.
    duration_ms : int, default 33
        1 frame あたりの表示時間 (33ms = ~30fps、SlimeVolley の TIMESTEP に近い).
    verbose : bool, default False
        True で Phase 1/2 の途中経過を print.

    Returns
    -------
    dict
        ``{"save_path": str, "best_idx": int, "best_score": float,
        "all_scores": list[float], "frames": int}``.
    """
    # 遅延 import (重い依存を visualize 単体利用時に排除)
    from .jax_env import _import_jax_env  # noqa: WPS433
    from .network import network_to_policy  # noqa: WPS433

    SlimeVolley, jax, jnp = _import_jax_env()
    from evojax.task.slimevolley import Game  # noqa: WPS433
    from PIL import Image as PILImage  # noqa: WPS433

    policy = network_to_policy(network)
    task = SlimeVolley(max_steps=max_steps, test=True)

    # --- Phase 1: スコア集計 --------------------------------------------
    base_key = jax.random.PRNGKey(int(seed))
    trial_keys = jax.random.split(base_key, n_trials)
    state = task.reset(trial_keys)

    total_rewards = jnp.zeros(n_trials, dtype=jnp.float32)
    alive = jnp.ones(n_trials, dtype=bool)
    step_count = jnp.zeros(n_trials, dtype=jnp.int32)

    for _t in range(max_steps):
        obs_np = np.asarray(state.obs)  # (n_trials, 12)
        actions_np = np.zeros((n_trials, 3), dtype=np.float32)
        for i in range(n_trials):
            a = np.asarray(policy(obs_np[i]), dtype=np.float32).reshape(-1)
            if a.shape[0] != 3:
                raise ValueError(f"policy returned {a.shape}, expected (3,)")
            actions_np[i] = a
        state, reward, done = task.step(state, jnp.asarray(actions_np))
        total_rewards = total_rewards + jnp.where(alive, reward, 0.0)
        step_count = step_count + jnp.where(alive, 1, 0)
        alive = alive & ~done
        if not bool(np.asarray(alive.any())):
            break

    total_rewards_np = np.asarray(total_rewards)
    step_count_np = np.asarray(step_count)
    best_idx = int(np.argmax(total_rewards_np))
    best_score = float(total_rewards_np[best_idx])
    best_steps = int(step_count_np[best_idx])
    if verbose:
        print(
            f"[gif] Phase 1: scores={total_rewards_np.tolist()}, "
            f"best={best_idx} score={best_score:+.2f} steps={best_steps}"
        )

    # --- Phase 2: 同一バッチで再走、best_idx のみ render ------------------
    state = task.reset(trial_keys)

    def _render_frame(state_obj, task_id: int):
        """Batch 次元の task_id を抽出して 1 frame をレンダリング."""
        gs = jax.tree.map(lambda x: x[task_id], state_obj.game_state)
        game = Game(gs)
        canvas = game.display()
        return PILImage.fromarray(np.asarray(canvas))

    frames: List[Any] = []
    cumul = 0.0
    for _t in range(max_steps):
        frames.append(_render_frame(state, best_idx))

        obs_np = np.asarray(state.obs)
        actions_np = np.zeros((n_trials, 3), dtype=np.float32)
        for i in range(n_trials):
            a = np.asarray(policy(obs_np[i]), dtype=np.float32).reshape(-1)
            actions_np[i] = a
        state, reward, done = task.step(state, jnp.asarray(actions_np))
        cumul += float(np.asarray(reward[best_idx]))
        if bool(np.asarray(done[best_idx])):
            frames.append(_render_frame(state, best_idx))
            break

    if verbose and abs(cumul - best_score) > 1e-3:
        print(
            f"[gif] WARNING: Phase 2 cumul={cumul:+.2f} differs from "
            f"Phase 1 best_score={best_score:+.2f} (trajectory drift)"
        )

    # --- GIF として保存 -------------------------------------------------
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        frames[0].save(
            str(save_path),
            save_all=True,
            append_images=frames[1:],
            duration=int(duration_ms),
            loop=0,
        )

    if verbose:
        print(f"[gif] saved {save_path} ({len(frames)} frames, score={best_score:+.2f})")

    return {
        "save_path": str(save_path),
        "best_idx": best_idx,
        "best_score": best_score,
        "best_steps": best_steps,
        "all_scores": total_rewards_np.tolist(),
        "frames": len(frames),
    }
