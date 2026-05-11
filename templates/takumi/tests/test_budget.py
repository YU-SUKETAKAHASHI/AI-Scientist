"""lib/budget.py — prompt caching 対応の cost 計算テスト."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.budget import (  # noqa: E402
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    BudgetTracker,
)


def test_estimate_without_cache() -> None:
    b = BudgetTracker()
    # Opus: $15/M input, $75/M output
    cost = b.estimate("claude-opus-4-7", input_tokens=20_000, output_tokens=300)
    expected = 20_000 * (15 / 1_000_000) + 300 * (75 / 1_000_000)
    assert abs(cost - expected) < 1e-9


def test_estimate_with_cache_write() -> None:
    b = BudgetTracker()
    cost = b.estimate(
        "claude-opus-4-7",
        input_tokens=100,
        output_tokens=300,
        cache_creation_input_tokens=20_000,
    )
    expected = (
        100 * (15 / 1_000_000)
        + 20_000 * (15 / 1_000_000) * CACHE_WRITE_MULTIPLIER
        + 300 * (75 / 1_000_000)
    )
    assert abs(cost - expected) < 1e-9


def test_estimate_with_cache_read_is_cheaper() -> None:
    b = BudgetTracker()
    fresh = b.estimate("claude-opus-4-7", input_tokens=20_000, output_tokens=300)
    cached = b.estimate(
        "claude-opus-4-7",
        input_tokens=100,
        output_tokens=300,
        cache_read_input_tokens=20_000,
    )
    # cache read = 0.1× → should be much cheaper
    assert cached < fresh * 0.3


def test_add_tracks_cache_totals(tmp_path: Path) -> None:
    b = BudgetTracker(out_dir=tmp_path)
    b.add(
        role="articulator", model="claude-opus-4-7",
        input_tokens=100, output_tokens=200,
        cache_creation_input_tokens=10_000,
    )
    b.add(
        role="articulator", model="claude-opus-4-7",
        input_tokens=100, output_tokens=200,
        cache_read_input_tokens=10_000,
    )
    snap = b.snapshot()
    assert snap["total_cache_creation_tokens"] == 10_000
    assert snap["total_cache_read_tokens"] == 10_000
    assert snap["n_calls"] == 2
    payload = json.loads((tmp_path / "cost_log.json").read_text())
    assert payload["calls"][0]["cache_creation_input_tokens"] == 10_000
    assert payload["calls"][1]["cache_read_input_tokens"] == 10_000


def test_cache_break_even_at_one_reuse() -> None:
    """1 write + 1 read が 2 fresh より安いはず."""
    b = BudgetTracker()
    fresh_total = 2 * b.estimate("claude-opus-4-7", input_tokens=20_000, output_tokens=300)
    cached_total = (
        b.estimate(
            "claude-opus-4-7", input_tokens=100, output_tokens=300,
            cache_creation_input_tokens=20_000,
        )
        + b.estimate(
            "claude-opus-4-7", input_tokens=100, output_tokens=300,
            cache_read_input_tokens=20_000,
        )
    )
    assert cached_total < fresh_total


def test_hard_limit_still_works() -> None:
    b = BudgetTracker(hard_limit=0.01, warning_threshold=0.001)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        # Opus 1k tokens output = 75/M × 1000 = $0.075 → exceeds limit
        b.add(
            role="articulator", model="claude-opus-4-7",
            input_tokens=0, output_tokens=1000,
        )
