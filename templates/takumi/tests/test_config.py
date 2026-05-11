"""lib/config.py — load + validate のテスト."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import load_config  # noqa: E402


def test_default_config_loads() -> None:
    cfg = load_config(ROOT / "config.yaml")
    assert isinstance(cfg.experiment.run_label, str)
    assert cfg.inner_loop.K > 0
    assert cfg.inner_loop.eval_episodes > 0
    assert cfg.budget.warning_threshold_usd <= cfg.budget.hard_limit_usd
    assert cfg.expert_trajectory_path().exists()
    assert cfg.expert_genome_path().exists()


def test_missing_section_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment:\n  run_label: minimal\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing section"):
        load_config(bad)


def test_with_overrides_returns_new_instance() -> None:
    cfg = load_config(ROOT / "config.yaml")
    cfg2 = cfg.with_overrides(experiment={"run_label": "alt"})
    assert cfg2.experiment.run_label == "alt"
    assert cfg.experiment.run_label != "alt"
    # source_path preserved on copy
    assert cfg2.source_path == cfg.source_path


def test_budget_threshold_inequality(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    src = (ROOT / "config.yaml").read_text(encoding="utf-8")
    bad.write_text(
        src.replace(
            "warning_threshold_usd: 70.0", "warning_threshold_usd: 200.0"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="warning_threshold_usd"):
        load_config(bad)
