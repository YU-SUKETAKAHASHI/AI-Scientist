"""lib/network.py の最小 round-trip テスト."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.network import (  # noqa: E402
    Network,
    apply_mutations,
    init_minimal_network,
    load_network,
    network_to_policy,
    serialize_network,
    validate_network,
)


def test_minimal_network_round_trip(tmp_path: Path) -> None:
    net = init_minimal_network(seed=0)
    ok, err = validate_network(net)
    assert ok, err
    assert len(net.nodes) == 15  # 12 inputs + 3 outputs
    assert len(net.edges) == 36

    f = tmp_path / "n.yaml"
    serialize_network(net, f)
    again = load_network(f)
    ok2, err2 = validate_network(again)
    assert ok2, err2
    # round-trip preserves edge count
    assert len(again.edges) == len(net.edges)


def test_policy_returns_3dim_binary() -> None:
    net = init_minimal_network(seed=1)
    policy = network_to_policy(net)
    out = policy(np.zeros(12))
    assert out.shape == (3,)
    assert set(np.unique(out)).issubset({0, 1})


def test_invalid_cycle_detected() -> None:
    net = init_minimal_network(seed=0)
    # Add cyclic hidden pair via direct mutation
    net.nodes.append({"id": "h_a", "type": "hidden", "activation": "tanh"})
    net.nodes.append({"id": "h_b", "type": "hidden", "activation": "tanh"})
    net.edges.append({"from": "h_a", "to": "h_b", "weight": 0.1})
    net.edges.append({"from": "h_b", "to": "h_a", "weight": 0.1})
    ok, err = validate_network(net)
    assert not ok
    assert "cycle" in err.lower()


def test_apply_mutations_change_weight_and_add_node() -> None:
    net = init_minimal_network(seed=2)
    mutations = [
        {"op": "change_weight", "from": "in_5", "to": "out_2", "new_weight": 0.9},
        {
            "op": "add_node",
            "new_id": "h_z",
            "activation": "tanh",
            "initial_edges": [
                {"from": "in_8", "to": "h_z", "weight": 0.5},
                {"from": "h_z", "to": "out_0", "weight": 0.3},
            ],
        },
    ]
    out = apply_mutations(net, mutations)
    ok, err = validate_network(out)
    assert ok, err
    assert any(n["id"] == "h_z" for n in out.nodes)


def test_weight_clipping_and_invalid_op() -> None:
    net = init_minimal_network(seed=3)
    # weight outside [-3, 3] should be clipped
    out = apply_mutations(
        net, [{"op": "change_weight", "from": "in_0", "to": "out_0", "new_weight": 999.0}]
    )
    we = next(e for e in out.edges if e["from"] == "in_0" and e["to"] == "out_0")
    assert we["weight"] == 3.0
    # unknown op should raise
    with pytest.raises(ValueError):
        apply_mutations(net, [{"op": "non_existent_op"}])


def test_delete_node_round_trip(tmp_path: Path) -> None:
    """add_node → delete_node で初期状態に戻ることを確認."""
    net = init_minimal_network(seed=4)
    n_init = len(net.nodes)
    e_init = len(net.edges)
    # add a hidden node with 3 incident edges
    net2 = apply_mutations(
        net,
        [
            {
                "op": "add_node",
                "new_id": "h_temp",
                "activation": "tanh",
                "initial_edges": [
                    {"from": "in_5", "to": "h_temp", "weight": 1.0},
                    {"from": "in_8", "to": "h_temp", "weight": -0.5},
                    {"from": "h_temp", "to": "out_2", "weight": 0.7},
                ],
            }
        ],
    )
    assert len(net2.nodes) == n_init + 1
    assert len(net2.edges) == e_init + 3
    # delete h_temp; both node and 3 edges should disappear
    net3 = apply_mutations(net2, [{"op": "delete_node", "id": "h_temp"}])
    assert len(net3.nodes) == n_init
    assert len(net3.edges) == e_init
    ok, err = validate_network(net3)
    assert ok, err


def test_delete_node_rejects_non_hidden() -> None:
    net = init_minimal_network(seed=5)
    for bad_id in ("in_0", "out_0"):
        with pytest.raises(ValueError, match="only hidden nodes"):
            apply_mutations(net, [{"op": "delete_node", "id": bad_id}])


def test_delete_node_rejects_unknown_id() -> None:
    net = init_minimal_network(seed=6)
    with pytest.raises(ValueError, match="unknown id"):
        apply_mutations(net, [{"op": "delete_node", "id": "h_does_not_exist"}])


def test_delete_node_requires_id() -> None:
    net = init_minimal_network(seed=7)
    with pytest.raises(ValueError, match="requires id"):
        apply_mutations(net, [{"op": "delete_node"}])


def test_delete_node_chain_with_other_ops() -> None:
    """delete_node を含む混合 mutation list が正しく動作する."""
    net = init_minimal_network(seed=8)
    out = apply_mutations(
        net,
        [
            # 1. add a hidden node
            {"op": "add_node", "new_id": "h_a", "activation": "tanh",
             "initial_edges": [
                 {"from": "in_4", "to": "h_a", "weight": 0.5},
                 {"from": "h_a", "to": "out_0", "weight": 0.3},
             ]},
            # 2. modify a weight on the new node
            {"op": "change_weight", "from": "in_4", "to": "h_a", "new_weight": 1.0},
            # 3. delete it again
            {"op": "delete_node", "id": "h_a"},
            # 4. add a different one
            {"op": "add_node", "new_id": "h_b", "activation": "relu",
             "initial_edges": [{"from": "in_5", "to": "h_b", "weight": -0.4},
                                {"from": "h_b", "to": "out_2", "weight": 0.8}]},
        ],
    )
    ok, err = validate_network(out)
    assert ok, err
    ids = {n["id"] for n in out.nodes}
    assert "h_a" not in ids
    assert "h_b" in ids
