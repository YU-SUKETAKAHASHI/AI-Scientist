"""Novice 意識層 (Mutator) — Sonnet が network 改造案を出す.

* Inputs (毎 cycle):
    1. 現在の Novice network YAML (dict)
    2. 直近 cycle の novice trajectory (1 episode 分、N=10 subsample)
    3. Skills file (Articulator 出力 markdown)
    4. dynamic stage label ("novice"/"intermediate"/"expert")
* Output: ``{"reasoning": str, "mutations": [{"op": ..., ...}, ...]}``

Validator は :func:`lib.network.apply_mutations` で実際に適用してみることで
graph 整合性を強制する. 失敗したら 3 回まで retry.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ruamel.yaml import YAML

from .budget import BudgetTracker
from .network import (
    BIAS_ID,
    INPUT_IDS,
    Network,
    OUTPUT_IDS,
    WEIGHT_MAX,
    WEIGHT_MIN,
    apply_mutations,
    validate_network,
)


MUTATOR_SYSTEM_PROMPT_TEMPLATE = (
    "You are the conscious layer (Mutator) of a novice SlimeVolley player. Your tacit "
    "layer is a small modifiable network (12 inputs → 3 outputs, optional hidden "
    "nodes). Each cycle you observe your own network, your most recent trajectory, "
    "the Skills document handed to you by the expert's Articulator, and a stage label. "
    "Your job is to propose a *small* set of mutations that should improve next cycle's "
    "score.\n\n"
    "Allowed operations:\n"
    "- change_weight: modify (or create) a single edge between two existing nodes.\n"
    "- add_node: add a hidden node with a chosen activation and a small initial set of "
    "edges.\n"
    "- delete_node: remove an existing hidden node. All edges touching that node are "
    "automatically removed. Only hidden nodes can be deleted (input / output / bias "
    "nodes are fixed and cannot be removed).\n\n"
    "Constraints:\n"
    "- Edges must form a DAG (no cycles).\n"
    "- Edge weights are clipped to [-3.0, 3.0].\n"
    "- Edge sources cannot be output nodes; edge targets cannot be input or bias nodes.\n"
    "- Propose at most {max_mutations} mutations per cycle.\n"
    "- Output MUST be valid YAML matching the schema described in the user message. "
    "No prose outside the YAML.\n"
    "- Keep your `reasoning:` field UNDER 400 WORDS (about 2500 characters). "
    "Reference only the 2-3 most decisive Skills rules; do NOT enumerate every "
    "section that could be relevant. Finish reasoning quickly so the YAML "
    "mutations block fits within the output token budget.\n"
)


YAML_SCHEMA_EXAMPLE = """\
reasoning: |
  <multi-line analysis: which inputs the trajectory shows the novice ignored,
   which Skills section informed your edits, and the expected effect>
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
    id: h_old        # only hidden nodes; all edges touching h_old are removed automatically
"""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _yaml_dump(data: Any) -> str:
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096
    buf = io.StringIO()
    yaml.dump(data, buf)
    return buf.getvalue()


def _network_block(network: Network) -> str:
    return _yaml_dump(network.to_dict())


def _trajectory_block(
    trajectory: List[Dict[str, Any]], *, precision: int = 1
) -> str:
    """Recent trajectory を flow-style + 低精度でレンダリング.

    詳細は :mod:`lib.trajectory_format` 参照.
    """
    from .trajectory_format import render_trajectory_entries  # noqa: WPS433
    body = render_trajectory_entries(trajectory, precision=precision, indent="  ")
    return "trajectory:\n" + body + "\n" if body else "trajectory: []\n"


def _render_static_section(skills_markdown: str) -> str:
    """K=10 cycle 共通 (= prompt caching 対象) の static 部分.

    含むもの: output schema example + 一般制約 + Articulator が生成した
    Skills 文書. ここまでが prompt の prefix で、cache_control: ephemeral を
    付ける. 1 idea 内の K=10 cycle で全部同じ内容なので、最初の cycle 以降は
    cache_read (0.1× input rate) になる.
    """
    parts: List[str] = []
    parts.append("=== REFERENCE: Output schema (emit YAML in this exact shape) ===")
    parts.append("```yaml")
    parts.append(YAML_SCHEMA_EXAMPLE.rstrip())
    parts.append("```")
    parts.append("")
    parts.append("=== REFERENCE: Constraints ===")
    parts.append(
        f"- Edge weight range: [{WEIGHT_MIN}, {WEIGHT_MAX}] (clipped automatically)"
    )
    parts.append("- Allowed activations for add_node: linear, tanh, sigmoid, relu")
    parts.append("- Edges must form a DAG (no cycles)")
    parts.append("- Edge sources cannot be output nodes")
    parts.append("- Edge targets cannot be input or bias nodes")
    parts.append("- delete_node only works for hidden nodes")
    parts.append("")
    parts.append("=== SKILLS DOCUMENT (Articulator output) ===")
    parts.append(skills_markdown.strip())
    return "\n".join(parts)


def _render_dynamic_section(
    *,
    network: Network,
    recent_trajectory: List[Dict[str, Any]],
    cycle_score: float,
    cycle_index: int,
    stage_label: str,
    extra_feedback: Optional[str],
) -> str:
    """cycle ごとに変わる dynamic 部分."""
    parts: List[str] = []
    parts.append("=== CURRENT STATE (this cycle) ===")
    parts.append(f"Cycle index: {cycle_index}")
    parts.append(f"Most recent cycle score: {cycle_score:+.3f}")
    parts.append(f"Dynamic stage label: {stage_label}")
    parts.append(
        "Use the stage label to pick the right level of advice from S12 if it is"
        " present in the Skills document. Reference specific Skills sections in"
        " your reasoning."
    )
    parts.append("")
    parts.append("# Current novice network YAML")
    parts.append("```yaml")
    parts.append(_network_block(network).rstrip())
    parts.append("```")
    parts.append("")
    parts.append("# Recent novice trajectory (subsampled, last episode)")
    parts.append("```yaml")
    parts.append(_trajectory_block(recent_trajectory).rstrip())
    parts.append("```")
    parts.append("")
    parts.append(
        "Allowed node ids in `from`/`to` for this cycle: "
        + ", ".join(sorted({n["id"] for n in network.nodes}))
    )
    if extra_feedback:
        parts.append("")
        parts.append("# Validation feedback from previous attempt")
        parts.append(extra_feedback)
    return "\n".join(parts)


def _build_user_blocks(
    *,
    network: Network,
    skills_markdown: str,
    recent_trajectory: List[Dict[str, Any]],
    cycle_score: float,
    cycle_index: int,
    stage_label: str,
    extra_feedback: Optional[str],
) -> List[Dict[str, Any]]:
    """Anthropic ``messages.create`` 用の content block 配列を組み立てる.

    1. **Static** (cached, K=10 cycle 共通): schema + 制約 + skills.md
    2. **Dynamic** (per cycle): cycle state + network + recent trajectory

    Static block (skills.md 込み) が ~1024 tokens を超えれば prompt caching
    の対象になる. A0 baseline は skills.md が短いため cache が効かない場合
    があるが、A_human / A1 等の richer ablation では効く.
    """
    static_text = _render_static_section(skills_markdown)
    dynamic_text = _render_dynamic_section(
        network=network,
        recent_trajectory=recent_trajectory,
        cycle_score=cycle_score,
        cycle_index=cycle_index,
        stage_label=stage_label,
        extra_feedback=extra_feedback,
    )
    return [
        {"type": "text", "text": static_text, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": dynamic_text},
    ]


# ---------------------------------------------------------------------------
# Output parsing + validation
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:yaml|yml)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def _extract_yaml(text: str) -> str:
    """``` ... ``` ブロックがあれば中身、なければ全文を返す."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _parse_mutator_yaml(text: str) -> Dict[str, Any]:
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    body = _extract_yaml(text)
    data = yaml.load(body)
    if not isinstance(data, dict):
        raise ValueError(f"top-level YAML must be a dict, got: {type(data).__name__}")
    if "mutations" not in data:
        raise ValueError("YAML missing 'mutations' field")
    if not isinstance(data["mutations"], list):
        raise ValueError("'mutations' must be a list")
    return data


def _normalise_mutations(mutations: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in mutations:
        if not isinstance(m, dict):
            raise ValueError(f"each mutation must be a dict, got: {m!r}")
        op = m.get("op")
        if op not in ("change_weight", "add_node", "delete_node"):
            raise ValueError(f"unknown op: {op!r}")
        out.append(dict(m))
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class MutatorResult:
    """Mutator の出力."""

    network: Network
    mutations: List[Dict[str, Any]]
    reasoning: str
    raw_text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    retries: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def _client():
    import anthropic  # noqa: WPS433

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; cannot call Claude API."
        )
    return anthropic.Anthropic(api_key=api_key)


# Articulator と同じ理由で、temperature を受け付けない model を記録する.
_NO_TEMPERATURE_MODELS: set[str] = set()


def _create_message(client, *, model: str, **kwargs):
    """``client.messages.create`` の薄い wrapper. articulator.py と同じ機能."""
    import anthropic  # noqa: WPS433
    if model in _NO_TEMPERATURE_MODELS:
        kwargs.pop("temperature", None)
    try:
        return client.messages.create(model=model, **kwargs)
    except anthropic.BadRequestError as exc:
        msg = str(exc).lower()
        if (
            "temperature" in msg
            and "deprecated" in msg
            and "temperature" in kwargs
        ):
            _NO_TEMPERATURE_MODELS.add(model)
            kwargs.pop("temperature", None)
            return client.messages.create(model=model, **kwargs)
        raise


def propose_mutations(
    current_network: Network,
    recent_trajectory: List[Dict[str, Any]],
    skills_file: str,
    *,
    cycle_score: float = 0.0,
    cycle_index: int = 0,
    current_stage_label: str = "novice",
    model: Optional[str] = None,
    budget: Optional[BudgetTracker] = None,
    max_tokens: int = 1500,
    temperature: float = 0.5,
    max_retries: int = 3,
    max_mutations_per_cycle: int = 6,
) -> MutatorResult:
    """Mutator を呼び mutations を取得し、当てた network を返す.

    Parameters
    ----------
    current_network : Network
        現在の novice 暗黙層.
    recent_trajectory : list[dict]
        直近 cycle の trajectory entries (``{t, o, a}``).
    skills_file : str
        Articulator の出力 markdown.
    cycle_score : float, default 0.0
        前 cycle の reward.
    cycle_index : int, default 0
        cycle 番号 (0-indexed).
    current_stage_label : str, default "novice"
        ``"novice" | "intermediate" | "expert"``.
    model : str or None
        override する model id. None なら env ``TAKUMI_MUTATOR_MODEL``、
        さらに無ければ ``claude-sonnet-4-6``.
    budget : BudgetTracker or None
        コスト追跡.
    max_tokens : int, default 1500
        Mutator の出力 max tokens.
    max_retries : int, default 3
        YAML / mutation validation 失敗時の retry 上限.

    Returns
    -------
    MutatorResult
        変異後 network, mutations, reasoning, token consumption.
    """
    model_id = (
        model
        or os.environ.get("TAKUMI_MUTATOR_MODEL")
        or "claude-sonnet-4-6"
    )
    system_prompt = MUTATOR_SYSTEM_PROMPT_TEMPLATE.format(
        max_mutations=max_mutations_per_cycle
    )

    client = _client()
    feedback: Optional[str] = None
    in_tokens = 0
    out_tokens = 0
    cache_creation = 0
    cache_read = 0
    raw_text = ""
    last_error: Optional[Exception] = None
    last_data: Optional[Dict[str, Any]] = None
    last_mutations: List[Dict[str, Any]] = []
    last_reasoning: str = ""

    for attempt in range(max_retries):
        # 毎 attempt で blocks を再構築 (feedback を dynamic 部分に注入).
        # static block (schema + skills) は同じ内容なので prompt caching 対象.
        user_blocks = _build_user_blocks(
            network=current_network,
            skills_markdown=skills_file,
            recent_trajectory=recent_trajectory,
            cycle_score=cycle_score,
            cycle_index=cycle_index,
            stage_label=current_stage_label,
            extra_feedback=feedback,
        )
        try:
            resp = _create_message(
                client,
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_blocks}],
            )
        except Exception as e:  # noqa: BLE001
            last_error = e
            time.sleep(min(2 ** attempt, 8))
            continue
        text_parts = [
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        ]
        raw_text = "".join(text_parts).strip()
        usage = resp.usage
        in_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        out_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)

        try:
            data = _parse_mutator_yaml(raw_text)
            mutations = _normalise_mutations(data["mutations"])
            last_data = data
            last_mutations = mutations
            last_reasoning = str(data.get("reasoning") or "").strip()
        except Exception as e:  # noqa: BLE001
            feedback = f"YAML parse failure: {e}"
            continue

        try:
            tentative = apply_mutations(current_network, mutations)
        except Exception as e:  # noqa: BLE001
            feedback = (
                f"apply_mutations rejected your mutations: {e}. "
                "Inspect the constraints and emit valid mutations."
            )
            continue

        ok, err = validate_network(tentative)
        if not ok:
            feedback = (
                f"Resulting network failed validation: {err}. "
                "Adjust your mutations and re-emit."
            )
            continue

        # success
        if budget is not None:
            entry = budget.add(
                role="mutator",
                model=model_id,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                note=(
                    f"cycle={cycle_index} stage={current_stage_label} "
                    f"n_mut={len(mutations)} retries={attempt} "
                    f"cache_write={cache_creation} cache_read={cache_read}"
                ),
            )
            cost = entry.cost_usd
        else:
            cost = 0.0
        return MutatorResult(
            network=tentative,
            mutations=mutations,
            reasoning=last_reasoning,
            raw_text=raw_text,
            model=model_id,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=cost,
            retries=attempt,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        )

    # All retries exhausted.
    if budget is not None and (in_tokens or out_tokens):
        entry = budget.add(
            role="mutator",
            model=model_id,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            note=(
                f"cycle={cycle_index} FAILED retries={max_retries} "
                f"final_feedback={feedback} "
                f"cache_write={cache_creation} cache_read={cache_read}"
            ),
        )
        cost = entry.cost_usd
    else:
        cost = 0.0

    if last_error is not None and not last_data:
        raise last_error

    # Valid YAML を取得できたが apply 失敗だった場合、空 mutations を当てて続行
    return MutatorResult(
        network=current_network,
        mutations=[],
        reasoning=last_reasoning,
        raw_text=raw_text,
        model=model_id,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        cost_usd=cost,
        retries=max_retries,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )


def stage_label_from_history(
    cycle_scores: List[float],
    *,
    novice_below: float = -2.0,
    expert_above: float = 2.0,
    window: int = 3,
) -> str:
    """直近 ``window`` cycle の平均 score から novice/intermediate/expert を判定する.

    仕様書 §5 Step 3 (S12 の dynamic 判定) に対応する小さな heuristic. 閾値は
    config.yaml の ``stage_label`` ブロックから注入することを想定.

    Parameters
    ----------
    cycle_scores : list of float
        各 cycle の reward.
    novice_below : float
        この値より直近平均が低ければ "novice".
    expert_above : float
        この値以上で "expert".
    window : int
        直近 N cycle.
    """
    if not cycle_scores:
        return "novice"
    recent = cycle_scores[-window:]
    avg = sum(recent) / len(recent)
    if avg < novice_below:
        return "novice"
    if avg < expert_above:
        return "intermediate"
    return "expert"
