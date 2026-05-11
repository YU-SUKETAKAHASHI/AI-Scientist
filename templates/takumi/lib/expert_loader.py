"""Part 1 で訓練した NEAT champion (npz) を本研究の Network YAML に変換する.

NEAT champion の npz は ``conn`` (5 × n_conn) と ``node`` (3 × n_node) を持つ
(``ref_implementations/neat/genome.py`` 参照)::

    node[0,:]: node id (int)
    node[1,:]: type (1=input, 2=output, 3=hidden, 4=bias)
    node[2,:]: activation function id (1=Linear, 5=Tanh, 6=Sigmoid, 9=ReLU, ...)

    conn[0,:]: innovation number
    conn[1,:]: source node id
    conn[2,:]: dest node id
    conn[3,:]: weight
    conn[4,:]: enabled flag (0/1)

Mapping convention (Part 1 train.py):

* node id 0  → bias (type 4)  → Takumi YAML id ``bias`` (type ``bias``)
* node id 1..12 → inputs (type 1) → ``in_0`` .. ``in_11``
* node id 13..15 → outputs (type 2) → ``out_0`` .. ``out_2``
* node id 16+ → hidden (type 3) → ``h_<original_id>``

NEAT activation id を YAML schema の文字列に変換する map も以下に持つ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .network import (
    BIAS_ID,
    INPUT_IDS,
    OUTPUT_IDS,
    Network,
    network_to_policy,
    serialize_network,
    validate_network,
)


# NEAT activation id → Takumi YAML activation 名 の map
NEAT_ACT_TO_NAME: Dict[int, str] = {
    1: "linear",
    3: "linear",      # Sin は YAML schema に無いため近似 (input 側のみで稀)
    4: "linear",      # Gaussian も同様
    5: "tanh",
    6: "sigmoid",
    7: "linear",      # Inverse (= -x) も近似
    9: "relu",
    10: "linear",     # Cosine 近似
}


def _act_to_name(act_id: int) -> str:
    return NEAT_ACT_TO_NAME.get(int(act_id), "tanh")


def load_neat_champion(npz_path: str | Path) -> Dict[str, np.ndarray]:
    """NEAT champion の npz を dict として読み込む.

    Parameters
    ----------
    npz_path : str or Path
        ``best.npz`` 形式のファイル ( keys: ``conn``, ``node`` ).

    Returns
    -------
    dict
        ``{"conn": np.ndarray (5, n_conn), "node": np.ndarray (3, n_node)}``.
    """
    data = np.load(npz_path)
    return {"conn": np.array(data["conn"]), "node": np.array(data["node"])}


def convert_neat_to_yaml(
    neat_genome: Dict[str, np.ndarray],
    *,
    label_inputs: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> Network:
    """NEAT genome dict を本研究の :class:`Network` に変換する.

    Parameters
    ----------
    neat_genome : dict
        ``load_neat_champion`` の出力.
    label_inputs : bool, default True
        True なら SlimeVolley の 12 次元入力に物理的意味の label を付ける.
    metadata : dict or None
        Network.metadata に格納する追加情報. ``source: "neat_champion"`` 等を
        付けるとデバッグしやすい.

    Returns
    -------
    Network
        Takumi YAML schema に変換済みの network.
    """
    node_arr = neat_genome["node"]
    conn_arr = neat_genome["conn"]

    input_labels = [
        "agent_x", "agent_y", "agent_vx", "agent_vy",
        "ball_x", "ball_y", "ball_vx", "ball_vy",
        "opp_x", "opp_y", "opp_vx", "opp_vy",
    ]

    # NEAT node id → Takumi YAML node id の lookup を作る
    id_map: Dict[int, str] = {}
    nodes: List[Dict[str, Any]] = []
    for col in range(node_arr.shape[1]):
        nid = int(node_arr[0, col])
        ntype = int(node_arr[1, col])
        act_id = int(node_arr[2, col])
        act_name = _act_to_name(act_id)
        if ntype == 4:  # bias
            id_map[nid] = BIAS_ID
            nodes.append({
                "id": BIAS_ID,
                "type": "bias",
                "activation": "linear",
                "label": "bias_const_1",
            })
        elif ntype == 1:  # input
            # NEAT convention: input ids 1..12 → in_0..in_11
            idx = nid - 1
            if not (0 <= idx < len(INPUT_IDS)):
                raise ValueError(
                    f"NEAT input node id {nid} maps to invalid in_{idx} "
                    f"(expected nid in 1..12)"
                )
            tid = INPUT_IDS[idx]
            id_map[nid] = tid
            n: Dict[str, Any] = {"id": tid, "type": "input", "activation": "linear"}
            if label_inputs:
                n["label"] = input_labels[idx]
            nodes.append(n)
        elif ntype == 2:  # output
            idx = nid - 13
            if not (0 <= idx < len(OUTPUT_IDS)):
                raise ValueError(
                    f"NEAT output node id {nid} maps to invalid out_{idx} "
                    f"(expected nid in 13..15)"
                )
            tid = OUTPUT_IDS[idx]
            id_map[nid] = tid
            output_labels = ["left", "right", "jump"]
            nodes.append({
                "id": tid,
                "type": "output",
                "activation": act_name,
                "label": output_labels[idx],
            })
        elif ntype == 3:  # hidden
            tid = f"h_{nid}"
            id_map[nid] = tid
            nodes.append({"id": tid, "type": "hidden", "activation": act_name})
        else:
            raise ValueError(f"unknown NEAT node type {ntype} for node id {nid}")

    edges: List[Dict[str, Any]] = []
    n_conn = conn_arr.shape[1]
    for col in range(n_conn):
        if int(conn_arr[4, col]) != 1:
            continue  # disabled
        src = int(conn_arr[1, col])
        dst = int(conn_arr[2, col])
        w = float(conn_arr[3, col])
        if src not in id_map or dst not in id_map:
            raise ValueError(
                f"NEAT edge {src}->{dst} references unknown node"
            )
        edges.append({
            "from": id_map[src],
            "to": id_map[dst],
            "weight": round(w, 6),
        })

    md = dict(metadata or {})
    md.setdefault("source", "neat_champion")
    md.setdefault("stage", "stage_2_payload")  # Stage 2 で読み込むときの目印
    return Network(nodes=nodes, edges=edges, metadata=md)


def save_expert_yaml(network: Network, yaml_path: str | Path) -> None:
    """Validation 後に YAML として書き出す薄い wrapper."""
    ok, err = validate_network(network)
    if not ok:
        raise ValueError(f"expert network failed validation: {err}")
    serialize_network(network, yaml_path)


def export_expert_yaml(
    npz_path: str | Path,
    yaml_path: str | Path,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Network:
    """End-to-end: ``best.npz`` → ``expert_genome.yaml``.

    Parameters
    ----------
    npz_path : str or Path
        ``best.npz`` 形式のファイル.
    yaml_path : str or Path
        書き出し先の YAML パス.
    metadata : dict or None
        ``Network.metadata`` に追加する補助情報.

    Returns
    -------
    Network
        export 後の :class:`Network`. policy 化等で再利用できるよう返す.
    """
    neat = load_neat_champion(npz_path)
    net = convert_neat_to_yaml(neat, metadata=metadata)
    save_expert_yaml(net, yaml_path)
    return net


# ---------------------------------------------------------------------------
# Trajectory collection (Task F)
# ---------------------------------------------------------------------------


def collect_expert_trajectories(
    expert_network: Network,
    *,
    num_episodes: int = 5,
    save_path: Optional[str | Path] = None,
    base_seed: int = 0,
    subsample_interval: int = 10,
    max_steps: int = 3000,
) -> Dict[str, Any]:
    """Expert を SlimeVolley で複数 episode play させ trajectory を保存する.

    Parameters
    ----------
    expert_network : Network
        変換済みの NEAT champion network.
    num_episodes : int, default 5
        収集する episode 数.
    save_path : str or Path or None
        ``data/expert_trajectory.yaml`` 等. None なら保存しない.
    base_seed : int, default 0
        episode i は ``base_seed + i`` で seed.
    subsample_interval : int, default 10
        trajectory 内 step 間隔.
    max_steps : int, default 3000
        episode 最大 step.

    Returns
    -------
    dict
        ``{"episodes": [...], "summary": {...}}``. summary には
        episode 毎の reward / 長さ / mean reward が入る.
    """
    # 遅延 import: env_wrapper 側で gym/slimevolleygym を import するため
    from .env_wrapper import play_episode

    policy = network_to_policy(expert_network)
    episodes: List[Dict[str, Any]] = []
    rewards: List[float] = []
    lengths: List[int] = []
    for i in range(num_episodes):
        result = play_episode(
            policy,
            seed=base_seed + i,
            subsample_interval=subsample_interval,
            max_steps=max_steps,
        )
        episodes.append(
            {
                "episode": i,
                "seed": base_seed + i,
                "total_reward": result["total_reward"],
                "episode_length": result["episode_length"],
                "trajectory": result["trajectory"],
            }
        )
        rewards.append(result["total_reward"])
        lengths.append(result["episode_length"])

    summary = {
        "num_episodes": num_episodes,
        "subsample_interval": subsample_interval,
        "rewards": rewards,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }
    bundle = {"summary": summary, "episodes": episodes}

    if save_path is not None:
        from ruamel.yaml import YAML
        yaml = YAML(typ="rt")
        yaml.default_flow_style = False
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.width = 4096
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(bundle, f)

    return bundle
