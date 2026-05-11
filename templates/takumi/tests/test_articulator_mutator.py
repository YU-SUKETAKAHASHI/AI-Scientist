"""lib/articulator.py + lib/mutator.py の interface 検査 (mock / parser only)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.articulator import (  # noqa: E402
    SECTION_FIXED_TEXTS,
    _format_hypothesis_components,
    _validate_skills,
    render_minimal_skills,
)
from lib.mutator import (  # noqa: E402
    _normalise_mutations,
    _parse_mutator_yaml,
    stage_label_from_history,
)


def test_format_hypothesis_components_dedup_and_filter() -> None:
    fh = {"components": ["S6", "s3", "S99", "S6"], "show_genome": False}
    out = _format_hypothesis_components(fh)
    assert out == ["S1", "S2", "S6", "S3"]


def test_render_minimal_skills_contains_required_sections() -> None:
    md = render_minimal_skills({"components": ["S3", "S7"], "show_genome": False})
    missing = _validate_skills(md, ["S1", "S2", "S3", "S7"])
    assert missing == []
    assert SECTION_FIXED_TEXTS["S1"][:30] in md


def test_mutator_parse_and_normalise() -> None:
    sample = """```yaml
reasoning: |
  testing
mutations:
  - op: change_weight
    from: in_5
    to: out_2
    new_weight: 0.8
  - op: add_node
    new_id: h_1
    activation: tanh
    initial_edges:
      - {from: in_8, to: h_1, weight: 0.5}
      - {from: h_1, to: out_0, weight: 0.4}
  - op: delete_node
    id: h_old
```"""
    data = _parse_mutator_yaml(sample)
    muts = _normalise_mutations(data["mutations"])
    assert len(muts) == 3
    assert muts[0]["op"] == "change_weight"
    assert muts[1]["op"] == "add_node"
    assert muts[2]["op"] == "delete_node"
    assert muts[2]["id"] == "h_old"


def test_mutator_rejects_unknown_op() -> None:
    import pytest
    bad = {"reasoning": "x", "mutations": [{"op": "rename_node", "id": "h_1"}]}
    with pytest.raises(ValueError, match="unknown op"):
        _normalise_mutations(bad["mutations"])


def test_stage_label_thresholds() -> None:
    assert stage_label_from_history([-5, -5, -5]) == "novice"
    assert stage_label_from_history([0, 0, 0]) == "intermediate"
    assert stage_label_from_history([3, 3, 3]) == "expert"
    assert stage_label_from_history([]) == "novice"
