"""Novice 暗黙層を表現する graph YAML の load / forward / mutate / serialize.

YAML schema
-----------
::

    nodes:
      - {id: <str>, type: <"input"|"hidden"|"output"|"bias">,
         activation: <"linear"|"tanh"|"sigmoid"|"relu">,
         label: <str, optional>}
      - ...
    edges:
      - {from: <node id>, to: <node id>, weight: <float>}
      - ...
    metadata:        # optional
      source: <str>
      cycle: <int>
      format_id: <str>

設計上の合意
~~~~~~~~~~~~

* SlimeVolley の 12 次元 obs に対応する input 群は ``in_0 .. in_11``
* 出力 3 次元 (left/right/jump) に対応する output 群は ``out_0 .. out_2``
* Bias node は ``id == "bias"``、``type == "bias"``、常に活性 1.0 を出力する
  (NEAT champion 由来の expert genome を素直に表現するため、仕様書 §4.3.1 の
  「シンプルな graph YAML」原則を最小限に拡張している。Novice 初期 network は
  bias を持たない。)
* Edge は DAG (cycle なし)
* weight range: ``[-3.0, 3.0]``。Validator が enforce する
* Mutation operator は v4 §4.3.1 の Pattern C 2 種:
    - ``change_weight``: edge があれば修正、なければ新設
    - ``add_node``: hidden node を追加し、初期 edges を同時に指定

Public API
~~~~~~~~~~

* :func:`load_network` — YAML path → :class:`Network`
* :func:`network_to_policy` — :class:`Network` → ``Callable[[np.ndarray], np.ndarray]``
* :func:`validate_network` — DAG / weight / id 一意性チェック
* :func:`apply_mutations` — mutation list を当てる
* :func:`serialize_network` — :class:`Network` → YAML
* :func:`init_minimal_network` — input → output 直結の minimal 全結合 network を生成
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from ruamel.yaml import YAML

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

NUM_INPUTS = 12
NUM_OUTPUTS = 3
INPUT_IDS = tuple(f"in_{i}" for i in range(NUM_INPUTS))
OUTPUT_IDS = tuple(f"out_{i}" for i in range(NUM_OUTPUTS))
BIAS_ID = "bias"
ALLOWED_TYPES = ("input", "hidden", "output", "bias")
ALLOWED_ACTIVATIONS = ("linear", "tanh", "sigmoid", "relu")
WEIGHT_MIN = -3.0
WEIGHT_MAX = 3.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Network:
    """Novice/Expert 共通の network 表現.

    Attributes
    ----------
    nodes : list[dict]
        各要素は ``{id, type, activation, label?}``.
    edges : list[dict]
        各要素は ``{from, to, weight}``.
    metadata : dict
        ``stage`` / ``cycle`` / ``format_id`` 等の補助情報.
    """

    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"nodes": self.nodes, "edges": self.edges}
        if self.metadata:
            out["metadata"] = self.metadata
        return out


# ---------------------------------------------------------------------------
# YAML round-trip helpers
# ---------------------------------------------------------------------------


def _make_yaml() -> YAML:
    """ruamel.yaml の YAML instance を round-trip 設定で返す.

    Returns
    -------
    YAML
        compact dump 用に flow style を許す YAML object.
    """
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.default_flow_style = False
    yaml.width = 4096
    return yaml


def load_network(yaml_path: str | Path) -> Network:
    """YAML file を Network に読み込む.

    Parameters
    ----------
    yaml_path : str or Path
        Network YAML のパス.

    Returns
    -------
    Network
        load 結果. validation はここでは行わない (caller 側で
        :func:`validate_network` を呼ぶこと).
    """
    yaml = _make_yaml()
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.load(f) or {}
    nodes = [dict(n) for n in (data.get("nodes") or [])]
    edges = [dict(e) for e in (data.get("edges") or [])]
    metadata = dict(data.get("metadata") or {})
    return Network(nodes=nodes, edges=edges, metadata=metadata)


def serialize_network(network: Network, yaml_path: str | Path) -> None:
    """Network を YAML として書き出す.

    Parameters
    ----------
    network : Network
        書き出す対象.
    yaml_path : str or Path
        出力先パス.
    """
    yaml = _make_yaml()
    out = network.to_dict()
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(out, f)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _node_index(network: Network) -> Dict[str, Dict[str, Any]]:
    """node id → node dict の lookup を作る."""
    return {n["id"]: n for n in network.nodes}


def _has_cycle(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    """DAG check (Kahn's algorithm). 循環があれば True."""
    in_deg: Dict[str, int] = {n: 0 for n in nodes}
    out_adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for src, dst in edges:
        if src not in in_deg or dst not in in_deg:
            continue
        in_deg[dst] += 1
        out_adj[src].append(dst)
    q = [n for n, d in in_deg.items() if d == 0]
    visited = 0
    while q:
        cur = q.pop()
        visited += 1
        for nxt in out_adj[cur]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                q.append(nxt)
    return visited != len(nodes)


def validate_network(network: Network) -> Tuple[bool, Optional[str]]:
    """Network が schema/制約を満たすか検査する.

    Parameters
    ----------
    network : Network
        検査対象.

    Returns
    -------
    tuple[bool, str or None]
        ``(ok, error_message)``. ok=True のとき error_message は None.
    """
    if not network.nodes:
        return False, "no nodes"

    seen: Dict[str, Dict[str, Any]] = {}
    for n in network.nodes:
        if "id" not in n or "type" not in n or "activation" not in n:
            return False, f"node missing required field: {n}"
        if n["id"] in seen:
            return False, f"duplicate node id: {n['id']}"
        if n["type"] not in ALLOWED_TYPES:
            return False, f"invalid type {n['type']!r} for node {n['id']}"
        if n["activation"] not in ALLOWED_ACTIVATIONS:
            return False, f"invalid activation {n['activation']!r} for node {n['id']}"
        if n["type"] == "bias" and n["id"] != BIAS_ID:
            return False, f"bias node must have id 'bias' (got {n['id']!r})"
        seen[n["id"]] = n

    # required input/output id 群が揃っているか
    for required in INPUT_IDS:
        if required not in seen or seen[required]["type"] != "input":
            return False, f"missing or wrong-typed input node: {required}"
    for required in OUTPUT_IDS:
        if required not in seen or seen[required]["type"] != "output":
            return False, f"missing or wrong-typed output node: {required}"

    edge_keys = set()
    for e in network.edges:
        if "from" not in e or "to" not in e or "weight" not in e:
            return False, f"edge missing required field: {e}"
        if e["from"] not in seen or e["to"] not in seen:
            return False, f"edge references unknown node: {e}"
        if seen[e["to"]]["type"] in ("input", "bias"):
            return False, f"edge cannot target {seen[e['to']]['type']} node: {e}"
        if seen[e["from"]]["type"] == "output":
            return False, f"edge cannot originate from output node: {e}"
        try:
            w = float(e["weight"])
        except (TypeError, ValueError):
            return False, f"non-numeric weight in edge: {e}"
        if not (WEIGHT_MIN <= w <= WEIGHT_MAX):
            return False, f"weight {w} out of range [{WEIGHT_MIN},{WEIGHT_MAX}]: {e}"
        key = (e["from"], e["to"])
        if key in edge_keys:
            return False, f"duplicate edge: {e}"
        edge_keys.add(key)

    if _has_cycle(
        [n["id"] for n in network.nodes],
        [(e["from"], e["to"]) for e in network.edges],
    ):
        return False, "graph contains a cycle"

    return True, None


# ---------------------------------------------------------------------------
# Activation dispatch + forward
# ---------------------------------------------------------------------------


def _activation(name: str, x: np.ndarray) -> np.ndarray:
    if name == "linear":
        return x
    if name == "tanh":
        return np.tanh(x)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if name == "relu":
        return np.maximum(0.0, x)
    raise ValueError(f"unknown activation: {name}")


def _topological_order(network: Network) -> List[str]:
    """Network の topological 順序を返す. cycle があれば例外."""
    nodes = [n["id"] for n in network.nodes]
    in_deg: Dict[str, int] = {n: 0 for n in nodes}
    out_adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for e in network.edges:
        in_deg[e["to"]] += 1
        out_adj[e["from"]].append(e["to"])
    order: List[str] = []
    q = [n for n, d in in_deg.items() if d == 0]
    while q:
        cur = q.pop(0)
        order.append(cur)
        for nxt in out_adj[cur]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                q.append(nxt)
    if len(order) != len(nodes):
        raise ValueError("network has a cycle, cannot compute forward")
    return order


def network_to_policy(
    network: Network,
    *,
    threshold: float = 0.0,
    deterministic: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Network を ``policy(obs) -> action`` callable に変換する.

    Parameters
    ----------
    network : Network
        forward の対象.
    threshold : float, default 0.0
        Output を binary action に変換する閾値.
        SlimeVolley の action 規約 (forward/backward/jump > 0) に揃えて 0.0.
        tanh 出力は ``[-1, 1]`` の範囲を取るので threshold 0.5 だと action が
        ほぼ発火しなくなる点に注意.
    deterministic : bool, default True
        ``True`` なら threshold で binary 化、``False`` なら 0/1 を
        そのまま返さず raw output を numpy.array で返す (debug 用).

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        12 次元 obs を受け取り 3 次元 binary action (np.int_) を返す.
    """
    order = _topological_order(network)
    nodes_by_id = _node_index(network)
    in_edges: Dict[str, List[Tuple[str, float]]] = {n: [] for n in nodes_by_id}
    for e in network.edges:
        in_edges[e["to"]].append((e["from"], float(e["weight"])))

    def policy(obs: np.ndarray) -> np.ndarray:
        """SlimeVolley の 12 次元 obs から 3 次元 action を計算する.

        Parameters
        ----------
        obs : np.ndarray
            shape ``(12,)``. 値は SlimeVolley が返す raw float (already /10).

        Returns
        -------
        np.ndarray
            shape ``(3,)``, dtype ``np.int_``. left/right/jump の binary action.
        """
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        if obs.shape[0] != NUM_INPUTS:
            raise ValueError(f"expected obs shape ({NUM_INPUTS},), got {obs.shape}")
        act: Dict[str, float] = {}
        for node_id in order:
            node = nodes_by_id[node_id]
            ntype = node["type"]
            if ntype == "input":
                idx = INPUT_IDS.index(node_id)
                v = obs[idx]
                act[node_id] = float(_activation(node["activation"], np.array(v)))
            elif ntype == "bias":
                act[node_id] = 1.0
            else:  # hidden / output
                z = 0.0
                for src, w in in_edges[node_id]:
                    z += act[src] * w
                act[node_id] = float(_activation(node["activation"], np.array(z)))
        out = np.array([act[oid] for oid in OUTPUT_IDS], dtype=np.float64)
        if not deterministic:
            return out
        return (out > threshold).astype(np.int_)

    return policy


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


def _clip_weight(w: float) -> float:
    return float(max(WEIGHT_MIN, min(WEIGHT_MAX, w)))


def apply_mutations(
    network: Network, mutations: List[Dict[str, Any]]
) -> Network:
    """Mutation list を順に当てた新しい Network を返す.

    対応する operation:

    * ``change_weight``: edge の重み変更 (無ければ新設, v4 §4.3.1)
    * ``add_node``:      hidden node を追加し initial edges を併せて指定 (v4 §4.3.1)
    * ``delete_node``:   hidden node を削除 (v4 spec の Pattern C 2 種を 3 種に
                         拡張、ユーザー要請 2026-05-10、`meta/lessons.md` 参照).
                         input/output/bias は削除不可. 関連 edges は連鎖削除される.

    不正な mutation は例外を投げる.

    Parameters
    ----------
    network : Network
        変異前の network. 引数は破壊しない.
    mutations : list[dict]
        各要素は ``{op, ...}``. 詳細は :mod:`lib.mutator` の prompt 参照.

    Returns
    -------
    Network
        mutation 適用後の network.
    """
    new_nodes = [dict(n) for n in network.nodes]
    new_edges = [dict(e) for e in network.edges]
    nodes_by_id = {n["id"]: n for n in new_nodes}

    def _find_edge(src: str, dst: str) -> Optional[Dict[str, Any]]:
        for e in new_edges:
            if e["from"] == src and e["to"] == dst:
                return e
        return None

    for mu in mutations:
        op = mu.get("op")
        if op == "change_weight":
            src = mu.get("from")
            dst = mu.get("to")
            w = mu.get("new_weight")
            if src is None or dst is None or w is None:
                raise ValueError(f"change_weight requires from/to/new_weight: {mu}")
            if src not in nodes_by_id:
                raise ValueError(f"change_weight: unknown 'from' node {src}")
            if dst not in nodes_by_id:
                raise ValueError(f"change_weight: unknown 'to' node {dst}")
            if nodes_by_id[dst]["type"] in ("input", "bias"):
                raise ValueError(f"change_weight: cannot target input/bias node: {dst}")
            if nodes_by_id[src]["type"] == "output":
                raise ValueError(
                    f"change_weight: cannot originate from output node: {src}"
                )
            w_clipped = _clip_weight(float(w))
            existing = _find_edge(src, dst)
            if existing is not None:
                existing["weight"] = w_clipped
            else:
                new_edges.append({"from": src, "to": dst, "weight": w_clipped})
        elif op == "add_node":
            new_id = mu.get("new_id")
            activation = mu.get("activation", "tanh")
            initial_edges = mu.get("initial_edges") or []
            if not new_id:
                raise ValueError(f"add_node requires new_id: {mu}")
            if new_id in nodes_by_id:
                raise ValueError(f"add_node: node id already exists: {new_id}")
            if activation not in ALLOWED_ACTIVATIONS:
                raise ValueError(f"add_node: invalid activation {activation!r}")
            new_node = {"id": new_id, "type": "hidden", "activation": activation}
            new_nodes.append(new_node)
            nodes_by_id[new_id] = new_node
            for spec in initial_edges:
                src = spec.get("from")
                dst = spec.get("to")
                w = spec.get("weight")
                # 片側だけ指定されている場合は new_node を補完する
                if src is None and dst is not None:
                    src = new_id
                if dst is None and src is not None and src != new_id:
                    dst = new_id
                if src is None or dst is None or w is None:
                    raise ValueError(
                        f"add_node initial_edge needs from/to/weight (or one of "
                        f"from/to plus weight): {spec}"
                    )
                if src not in nodes_by_id or dst not in nodes_by_id:
                    raise ValueError(f"add_node initial_edge unknown node: {spec}")
                if nodes_by_id[dst]["type"] in ("input", "bias"):
                    raise ValueError(
                        f"add_node initial_edge cannot target input/bias: {spec}"
                    )
                if nodes_by_id[src]["type"] == "output":
                    raise ValueError(
                        f"add_node initial_edge cannot originate from output: {spec}"
                    )
                w_clipped = _clip_weight(float(w))
                existing = _find_edge(src, dst)
                if existing is None:
                    new_edges.append({"from": src, "to": dst, "weight": w_clipped})
                else:
                    existing["weight"] = w_clipped
        elif op == "delete_node":
            target = mu.get("id") or mu.get("node_id")
            if not target:
                raise ValueError(f"delete_node requires id (or node_id): {mu}")
            if target not in nodes_by_id:
                raise ValueError(f"delete_node: unknown id: {target!r}")
            tgt_type = nodes_by_id[target]["type"]
            if tgt_type != "hidden":
                raise ValueError(
                    f"delete_node: only hidden nodes can be deleted "
                    f"(got type={tgt_type!r} for id {target!r})"
                )
            # 関連 edges を連鎖削除 (target が from または to のもの)
            new_edges = [
                e for e in new_edges
                if e["from"] != target and e["to"] != target
            ]
            # node 本体を削除
            new_nodes = [n for n in new_nodes if n["id"] != target]
            nodes_by_id.pop(target, None)
        else:
            raise ValueError(f"unknown mutation op: {op!r}")

    return Network(nodes=new_nodes, edges=new_edges, metadata=dict(network.metadata))


# ---------------------------------------------------------------------------
# Initial novice network
# ---------------------------------------------------------------------------


def init_minimal_network(
    *,
    seed: Optional[int] = None,
    weight_scale: float = 0.1,
    activation: str = "tanh",
) -> Network:
    """Input → output 直結の minimal 全結合 network を作る (v4 §4.3.1).

    12 inputs × 3 outputs = 36 個の edge が初期化される. hidden node や
    bias は持たない.

    Parameters
    ----------
    seed : int or None
        random weight 用 seed.
    weight_scale : float, default 0.1
        初期重みの std.
    activation : str, default "tanh"
        output node の活性化関数.

    Returns
    -------
    Network
        minimal 全結合 network.
    """
    rng = np.random.default_rng(seed)
    nodes: List[Dict[str, Any]] = []
    for iid in INPUT_IDS:
        nodes.append({"id": iid, "type": "input", "activation": "linear"})
    for oid in OUTPUT_IDS:
        nodes.append({"id": oid, "type": "output", "activation": activation})
    edges: List[Dict[str, Any]] = []
    for iid in INPUT_IDS:
        for oid in OUTPUT_IDS:
            w = float(rng.normal(0.0, weight_scale))
            edges.append({"from": iid, "to": oid, "weight": _clip_weight(w)})
    return Network(nodes=nodes, edges=edges, metadata={})
