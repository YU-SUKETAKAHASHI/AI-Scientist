"""LLM 呼び出しのトークン消費とコストを追跡する薄いユーティリティ.

* warning_threshold ($70) で警告
* hard_limit ($95) を超えたら :class:`RuntimeError` を投げる
* 呼び出しごとの input/output token 数 + 合計コストを ``cost_log.json`` に書き出す
* Anthropic prompt caching を考慮: ``cache_creation_input_tokens`` (1.25× input rate)
  と ``cache_read_input_tokens`` (0.1× input rate) を別集計する.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


# 単価 (USD per token). 仕様書 §12 を参照.
# Anthropic 公式の prompt caching 料金倍率:
#   cache write = 1.25 × input
#   cache read  = 0.10 × input
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10

DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    # Opus 4.6 / 4.7 系
    "opus": {"input": 15.0 / 1_000_000, "output": 75.0 / 1_000_000},
    # Sonnet 4.6 系
    "sonnet": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    # Haiku 4.5 系
    "haiku": {"input": 1.0 / 1_000_000, "output": 5.0 / 1_000_000},
}


def _model_family(model_id: str) -> str:
    m = model_id.lower()
    if "opus" in m:
        return "opus"
    if "sonnet" in m:
        return "sonnet"
    if "haiku" in m:
        return "haiku"
    return "sonnet"  # default


@dataclass
class CostEntry:
    """1 つの LLM 呼び出しの記録 (prompt caching 対応)."""

    role: str
    model: str
    input_tokens: int           # 非 cache 部分の input tokens
    output_tokens: int
    cost_usd: float
    note: str = ""
    cache_creation_input_tokens: int = 0   # 初回 cache 書込時の input tokens
    cache_read_input_tokens: int = 0       # 再利用 cache 読込時の input tokens


@dataclass
class BudgetTracker:
    """累積 USD コストを追跡し、安全閾値を超えたら警告/中断する.

    Parameters
    ----------
    out_dir : Path or None
        ``cost_log.json`` を書き出すディレクトリ. None なら書き出さない.
    warning_threshold : float, default 70.0
        この USD を超えたら 1 度だけ ``print`` で警告.
    hard_limit : float, default 95.0
        この USD を超えたら ``RuntimeError("budget exceeded")`` を投げる.
    """

    out_dir: Optional[Path] = None
    warning_threshold: float = 70.0
    hard_limit: float = 95.0
    pricing: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: dict(DEFAULT_PRICING)
    )
    entries: List[CostEntry] = field(default_factory=list)
    total_usd: float = 0.0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    _warned: bool = False
    _lock: Lock = field(default_factory=Lock, repr=False)

    def estimate(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> float:
        """USD コストを計算する.

        non-cache input + cache write (1.25×) + cache read (0.10×) + output.
        """
        family = _model_family(model)
        rates = self.pricing.get(family, self.pricing["sonnet"])
        in_rate = rates["input"]
        out_rate = rates["output"]
        return (
            input_tokens * in_rate
            + cache_creation_input_tokens * in_rate * CACHE_WRITE_MULTIPLIER
            + cache_read_input_tokens * in_rate * CACHE_READ_MULTIPLIER
            + output_tokens * out_rate
        )

    def add(
        self,
        *,
        role: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        note: str = "",
    ) -> CostEntry:
        cost = self.estimate(
            model,
            input_tokens,
            output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )
        with self._lock:
            self.total_usd += cost
            self.total_cache_creation_tokens += int(cache_creation_input_tokens)
            self.total_cache_read_tokens += int(cache_read_input_tokens)
            entry = CostEntry(
                role=role,
                model=model,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                cost_usd=cost,
                note=note,
                cache_creation_input_tokens=int(cache_creation_input_tokens),
                cache_read_input_tokens=int(cache_read_input_tokens),
            )
            self.entries.append(entry)
            if self.total_usd >= self.warning_threshold and not self._warned:
                print(
                    f"[budget][WARN] cumulative cost {self.total_usd:.2f} USD "
                    f"exceeded warning threshold {self.warning_threshold:.2f} USD"
                )
                self._warned = True
            self._flush()
            if self.total_usd >= self.hard_limit:
                raise RuntimeError(
                    f"budget exceeded: cumulative {self.total_usd:.2f} USD >= "
                    f"hard limit {self.hard_limit:.2f} USD"
                )
        return entry

    def _flush(self) -> None:
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "cost_log.json"
        payload: Dict[str, Any] = {
            "total_usd": self.total_usd,
            "warning_threshold": self.warning_threshold,
            "hard_limit": self.hard_limit,
            "n_calls": len(self.entries),
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "calls": [vars(e) for e in self.entries],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_usd": self.total_usd,
            "n_calls": len(self.entries),
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
        }
