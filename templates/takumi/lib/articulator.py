"""Expert 意識層 (Articulator) — `show_genome: bool` で genome 可視性を切替えて Skills file (Markdown) を生成する.

* Model: Opus 4 系 (env ``TAKUMI_ARTICULATOR_MODEL`` で上書き可能).
* `show_genome=False` (旧 Stage 1): Inputs = trajectory のみ.
* `show_genome=True`  (旧 Stage 2): Inputs = trajectory + Expert 暗黙層 network YAML.
* 出力: Markdown. S1+S2 (fixed) と format_hypothesis.components で
  指定された S3-S12 sections を必ず含む.

Skills file の構造
~~~~~~~~~~~~~~~~~~

::

    # Skills: SlimeVolley Expert
    ## S1. Goal
    ...
    ## S2. Rules
    ...
    ## S3. Phase markers (rally phase)   <-- format_hypothesis に含まれている場合のみ
    ...

S1/S2 は固定文 (`SECTION_FIXED_TEXTS`). Articulator は他 sections のみを生成する.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from ruamel.yaml import YAML

from .budget import BudgetTracker

# ---------------------------------------------------------------------------
# Section 定義
# ---------------------------------------------------------------------------

SECTION_TITLES: Dict[str, str] = {
    "S1": "Goal",
    "S2": "Rules",
    "S3": "Phase markers (rally phase)",
    "S4": "Key states",
    "S5": "Trajectory exemplars",
    "S6": "Counter-examples (Gotchas)",
    "S7": "External focus cues",
    "S8": "Internal focus cues",
    "S9": "Metaphor / analogy",
    "S10": "Onomatopoeia",
    "S11": "Statistics",
    "S12": "Stage-conditional cues",
}

SECTION_DESCRIPTIONS: Dict[str, str] = {
    "S3": (
        "Decompose play into rally phases (e.g., serve, approach, contact, recovery, idle). "
        "For each phase, give (a) a one-line definition, (b) the observation pattern that "
        "marks its onset (e.g., 'ball.vx changes sign'), and (c) the dominant action class "
        "the expert takes. Infer phases from the trajectory — phase labels are NOT provided. "
        "Target: 4-6 phases."
    ),
    "S4": (
        "Distill the expert's decision logic into 6-10 production rules of the form "
        "`IF <condition on observation> THEN <action>`, where conditions use the 12-dim "
        "obs vocabulary (agent.x, ball.vy, etc.) and actions are subsets of {left, right, "
        "jump}. Rules should be discrete, near-mutually-exclusive, and together cover the "
        "phases in S3. Ground each rule in at least one trajectory moment."
    ),
    "S5": (
        "Cite 3-5 worked examples (≤5 timesteps each) drawn from DIFFERENT phases of S3. "
        "Format: `(t=120, ball=(0.4,0.7,vy=-0.2), opp.x=-0.5) → JUMP` + one-sentence "
        "rationale (why this action, not the alternatives). Prefer non-obvious moments "
        "where the expert's choice deviates from a naive heuristic."
    ),
    "S6": (
        "5-8 anti-patterns the novice must avoid, each as `NEVER <action> WHEN <condition>` "
        "+ a one-line reason. Prioritise mistakes a randomly-initialised novice is most "
        "likely to make: reflexive jumping when the ball is far, idling when the ball "
        "approaches, double-commitment (left+right). Where useful, contrast with an S5 "
        "excerpt where the expert pointedly did NOT do the obvious thing."
    ),
    "S7": (
        "External-focus cues (Wulf, 2013): 4-6 directives that point the novice's attention "
        "AWAY from the body and TOWARD the ball, opponent, or environmental effect to "
        "produce. Examples: 'send the ball to the back corner', 'meet the ball at its "
        "peak', 'shadow the opponent's x-position'. Phrase as imperatives."
    ),
    "S8": (
        "Internal-focus cues (Wulf control condition): 4-6 directives that point the "
        "novice's attention TOWARD its own body and movement mechanics — joint extension, "
        "timing relative to its own posture, effort level. Examples: 'extend the leg "
        "fully at jump apex', 'plant before pushing off'. WRITE WITH THE SAME CARE AS S7 — "
        "the experiment tests whether external > internal, not whether S7 was written "
        "better than S8."
    ),
    "S9": (
        "One or two unifying metaphors capturing the expert's overall stance — defensive "
        "wall vs opportunistic counter-puncher, anticipatory vs reactive, etc. Vivid "
        "enough to bias the novice's global posture. Avoid trivial restatements "
        "('be like a slime')."
    ),
    "S10": (
        "Re-narrate the expert's play using Japanese onomatopoeia (擬音語/擬態語) as the "
        "primary descriptive vocabulary, mirroring the phase decomposition of S3. For each "
        "phase in S3, describe what the expert does and how it feels timing-wise, with "
        "onomatopoeia embedded directly in the action descriptions. DO NOT gloss or "
        "explain the onomatopoeia themselves — let them function as description. "
        "Example: 'approach phase: the slime スッと shifts under the ball's landing "
        "spot, タメて waits half a beat, タンッと plants and pushes off'. Cover every "
        "phase from S3. The hypothesis is that onomatopoeia-laden narration transmits "
        "the timing and feel that plain prose cannot."
    ),
    "S11": (
        "Compact quantitative profile of the expert, computed from the trajectory. "
        "Present as a markdown table. Include at minimum: (a) global action frequency "
        "for {left, right, jump, no-op}, (b) action distribution per S3 phase, (c) mean "
        "and std of ball.y at the moment of jump, (d) mean horizontal distance kept "
        "from the opponent. Numbers, not adjectives."
    ),
    "S12": (
        "Three stage-conditional advice sets indexed by the novice's current proficiency "
        "(Mutator selects stage from its rolling eval score). Stage 守 (basic): "
        "positioning, ball tracking, never-no-op-near-ball. Stage 破 (intermediate): "
        "apply S4 production rules, respect S6 anti-patterns. Stage 離 (advanced): "
        "exploit S7 external-focus targeting, refine timing per S10. (Fitts & Posner "
        "1967; 守破離.) Each stage: 3-5 directives, written in stage-appropriate "
        "vocabulary."
    ),
}

# S1, S2 は実質固定文だが、format_hypothesis 側からの上書きは許す.
SECTION_FIXED_TEXTS: Dict[str, str] = {
    "S1": (
        "Maximize total reward in 1 episode (3000 steps) of SlimeVolley against the "
        "built-in opponent. Reward is +1 when the agent scores a point and -1 when the "
        "opponent scores. The novice begins each cycle with a fresh policy; cycle "
        "improvement is achieved by editing its 12-input → 3-output network "
        "(left, right, jump)."
    ),
    "S2": (
        "Action semantics: out_0 (left), out_1 (right), out_2 (jump). Each action fires "
        "when the corresponding output > 0. Observation indices: 0..3 = agent (x,y,vx,"
        "vy), 4..7 = ball (x,y,vx,vy), 8..11 = opponent (x,y,vx,vy), all scaled by 1/10."
    ),
}


# ---------------------------------------------------------------------------
# Skills file rendering helpers
# ---------------------------------------------------------------------------


def _format_hypothesis_components(format_hypothesis: Dict[str, Any]) -> List[str]:
    """format_hypothesis.components を S1/S2 を加えた重複なしのリストに揃える."""
    raw = list(format_hypothesis.get("components") or [])
    out: List[str] = []
    seen: set[str] = set()
    for s in ["S1", "S2"] + raw:
        s = str(s).strip().upper()
        if s in seen:
            continue
        if s not in SECTION_TITLES:
            continue  # ignore unknown labels silently
        out.append(s)
        seen.add(s)
    return out


def _short_genome_summary(network_yaml: Dict[str, Any]) -> str:
    """`show_genome=True` で genome を渡すときの small summary を作る (token 節約)."""
    nodes = network_yaml.get("nodes") or []
    edges = network_yaml.get("edges") or []
    n_input = sum(1 for n in nodes if n.get("type") == "input")
    n_output = sum(1 for n in nodes if n.get("type") == "output")
    n_hidden = sum(1 for n in nodes if n.get("type") == "hidden")
    n_bias = sum(1 for n in nodes if n.get("type") == "bias")
    return (
        f"NEAT champion genome: {len(nodes)} nodes "
        f"(in={n_input}, out={n_output}, hidden={n_hidden}, bias={n_bias}); "
        f"{len(edges)} active edges."
    )


def _render_genome_block(network: Dict[str, Any]) -> str:
    """`show_genome=True` 用に genome を YAML として埋め込む文字列を返す."""
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096
    buf = io.StringIO()
    payload = {"nodes": network.get("nodes", []), "edges": network.get("edges", [])}
    yaml.dump(payload, buf)
    return buf.getvalue()


def _render_trajectory_block(
    expert_trajectory: Dict[str, Any],
    *,
    max_episodes: int = 1,
    precision: int = 1,
) -> str:
    """Trajectory bundle を flow-style + 低精度 (precision=1) で整形する.

    各 entry を 1 行 (``- {t: 0, o: [...], a: [...]}``) にすることで
    block-style YAML より ~50–60% コンパクトになる. 詳細は
    :mod:`lib.trajectory_format` 参照.

    Parameters
    ----------
    expert_trajectory : dict
        ``collect_expert_trajectories`` が保存した bundle.
    max_episodes : int, default 1
        prompt に載せる episode 数 (lost-in-the-middle 対策で default 1).
    precision : int, default 1
        小数桁数. SlimeVolley の obs は /10 スケール済みなので 1 で十分.
    """
    from .trajectory_format import render_trajectory_bundle  # noqa: WPS433
    return render_trajectory_bundle(
        expert_trajectory, max_episodes=max_episodes, precision=precision
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

ARTICULATOR_SYSTEM_PROMPT = (
    """
    You are the conscious layer (Articulator) of an expert SlimeVolley player.
    Your tacit layer is a small neural network that has been trained against the 
    built-in opponent. Your task is to verbalize the expert's tactical knowledge into 
    a Skills document for a novice — a separate agent whose own conscious layer (the 
    Mutator) will consume your document and edit its own network accordingly.\n\n
    Write Markdown that is concrete, concise, and grounded in the data you receive (trajectories, optionally 
    the genome). Do not invent details that are not supported. Each section must be a 
    GitHub Flavored Markdown subsection (## S<n>. ...). Output Markdown only; no 
    preamble, no code fences around the whole document.
    """
)


_ALL_SECTION_IDS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
                    "S10", "S11", "S12")


def _render_all_sections_reference() -> str:
    """全 S1..S12 の reference (title + fixed_text + description) を静的にレンダリング.

    全 idea で内容が同じなので prompt caching の対象になる. Articulator は
    この reference 全文を見た上で、後続の "task" block で指定された subset だけ
    出力に含める.
    """
    parts: List[str] = []
    parts.append("=== REFERENCE: Available Skills Sections (S1..S12) ===")
    parts.append("")
    parts.append(
        "Below is the complete reference for all sections. The task block that "
        "follows (after the reference + expert data) will specify which subset "
        "to include in your output."
    )
    parts.append("")
    for sid in _ALL_SECTION_IDS:
        title = SECTION_TITLES[sid]
        parts.append(f"## {sid}. {title}")
        if sid in SECTION_FIXED_TEXTS:
            parts.append(
                "FIXED TEXT (emit verbatim when this section is requested):"
            )
            parts.append(SECTION_FIXED_TEXTS[sid])
        if sid in SECTION_DESCRIPTIONS:
            parts.append(f"Description: {SECTION_DESCRIPTIONS[sid]}")
        parts.append("")
    return "\n".join(parts)


def _render_task_section(
    *,
    components: List[str],
    format_hypothesis: Dict[str, Any],
    show_genome: bool,
    extra_instructions: Optional[str],
) -> str:
    """idea ごとに変わる dynamic 部分."""
    parts: List[str] = []
    parts.append("=== YOUR TASK ===")
    parts.append(
        f"Skills format hypothesis: {json.dumps(format_hypothesis, ensure_ascii=False)}"
    )
    parts.append(f"show_genome: {show_genome}")
    if extra_instructions:
        parts.append(f"Extra instructions: {extra_instructions}")
    parts.append("")
    parts.append(
        f"Required sections to include in this run (subset of S1..S12): "
        f"{', '.join(components)}"
    )
    parts.append(
        "For S1 (Goal) and S2 (Rules), emit the EXACT fixed text from the reference"
        " above. For other requested sections (S3..S12), generate content grounded in"
        " the trajectory data (and genome if show_genome=True), following the"
        " description given for that section in the reference."
    )
    parts.append("")
    parts.append(
        "Output the complete Skills document as Markdown. Begin with the heading"
        " `# Skills: SlimeVolley Expert`. Then emit the listed sections in the"
        " order given above. Keep total length under 1500 tokens. Output Markdown"
        " only — no preamble, no code fences around the whole document."
    )
    return "\n".join(parts)


def _build_user_blocks(
    *,
    components: List[str],
    format_hypothesis: Dict[str, Any],
    show_genome: bool,
    trajectory_block: str,
    genome_block: Optional[str],
    extra_instructions: Optional[str],
) -> List[Dict[str, Any]]:
    """Anthropic ``messages.create`` 用の content block 配列を組み立てる.

    構造 (上から順):

    1. **Reference** (static, 全 idea 共通): S1..S12 の完全リファレンス
    2. **Trajectory** (static, 共通): expert trajectory YAML
    3. **Genome** (static, `show_genome=True` のみ): expert genome YAML
    4. **[cache_control: ephemeral]** が 3 (または無ければ 2) に付く ← cache breakpoint
    5. **Task** (dynamic, idea ごとに変わる): format_hypothesis + 必要 section リスト

    prefix が byte-identical なら Anthropic side で **cache hit** され、cache_read
    料金 (0.1× input rate) で済む. 初回は cache write (1.25× input rate).

    詳細は :mod:`lib.budget` の caching コメント参照.
    """
    blocks: List[Dict[str, Any]] = []

    # 1. Reference (全 idea で同一)
    blocks.append({"type": "text", "text": _render_all_sections_reference()})

    # 2. Trajectory (show_genome on/off 共通; ~22k tokens で支配的)
    traj_section = (
        "=== EXPERT TRAJECTORY DATA (subsampled) ===\n"
        "```yaml\n"
        f"{trajectory_block.rstrip()}\n"
        "```"
    )
    blocks.append({"type": "text", "text": traj_section})

    # 3. Genome (`show_genome=True` のみ)
    if genome_block is not None:
        genome_section = (
            "=== EXPERT TACIT-LAYER NETWORK (show_genome=True — genome inspection) ===\n"
            "Reference the structure below when writing mechanistic statements such"
            " as 'the expert relies strongly on input ball.vy via h_32'.\n"
            "```yaml\n"
            f"{genome_block.rstrip()}\n"
            "```"
        )
        blocks.append({"type": "text", "text": genome_section})

    # 4. cache breakpoint を最後の静的 block に付与
    blocks[-1]["cache_control"] = {"type": "ephemeral"}

    # 5. Task (dynamic)
    blocks.append(
        {
            "type": "text",
            "text": _render_task_section(
                components=components,
                format_hypothesis=format_hypothesis,
                show_genome=show_genome,
                extra_instructions=extra_instructions,
            ),
        }
    )
    return blocks


# ---------------------------------------------------------------------------
# LLM client wrapper
# ---------------------------------------------------------------------------


def _client():
    """anthropic SDK の Client を返す (lazy import)."""
    import anthropic  # noqa: WPS433

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; cannot call Claude API. "
            "Export the env var or run articulate in mock mode."
        )
    return anthropic.Anthropic(api_key=api_key)


# 一部の新世代 model (例: claude-opus-4-7) は temperature を受け付けない.
# 一度 deprecated エラーを観測したら以降は temperature を送らない.
_NO_TEMPERATURE_MODELS: set[str] = set()


def _create_message(client, *, model: str, **kwargs):
    """``client.messages.create`` の薄い wrapper.

    "`temperature` is deprecated for this model." エラーを 1 度 観測したら
    その model は :data:`_NO_TEMPERATURE_MODELS` に記録し、以降の呼び出しでは
    automatically temperature を省く. 他のエラーはそのまま raise する.
    """
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


@dataclass
class ArticulationResult:
    """Articulator の出力."""

    skills_markdown: str
    components_used: List[str]
    show_genome: bool
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def _validate_skills(markdown: str, required: Iterable[str]) -> List[str]:
    """Markdown 内に必要な ``## S<n>. ...`` 見出しが揃っているかチェック.

    返り値は 欠けている section ID リスト (空ならOK).
    """
    missing: List[str] = []
    for sid in required:
        pattern = rf"(?im)^\s*##\s*{re.escape(sid)}\b"
        if not re.search(pattern, markdown):
            missing.append(sid)
    return missing


def articulate_skills(
    format_hypothesis: Dict[str, Any],
    expert_trajectory: Dict[str, Any],
    *,
    show_genome: bool = False,
    expert_genome: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    budget: Optional[BudgetTracker] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    max_retries: int = 3,
    trajectory_episodes: int = 1,
    extra_instructions: Optional[str] = None,
) -> ArticulationResult:
    """Skills file を生成する.

    Parameters
    ----------
    format_hypothesis : dict
        ``{"components": [...], "show_genome": <bool>, "instructions": <opt>}``.
        ``show_genome`` は AI Scientist が制御する 1 bit 軸. 呼び出し側で
        format_hypothesis から取り出して下記 ``show_genome`` 引数に渡す.
    expert_trajectory : dict
        :func:`lib.expert_loader.collect_expert_trajectories` の出力 bundle.
    show_genome : bool, default False
        ``False`` (旧 Stage 1): trajectory のみで articulate.
        ``True``  (旧 Stage 2): trajectory + Expert genome (network YAML) で articulate.
    expert_genome : dict or None
        ``show_genome=True`` のときの :class:`lib.network.Network` の dict 表現
        (``nodes``/``edges`` を持つ). ``show_genome=False`` のときは無視.
    model : str or None
        override する model id. None なら env ``TAKUMI_ARTICULATOR_MODEL``、
        さらに無ければ ``claude-opus-4-7``.
    budget : BudgetTracker or None
        コスト追跡. None なら追跡しない.
    max_tokens : int, default 4096
        Articulator の出力 max tokens.
    temperature : float, default 0.7
        sampling temperature.
    max_retries : int, default 3
        欠落 section や 5xx エラーでの retry 上限.
    trajectory_episodes : int, default 1
        prompt に乗せる expert episode 数 (lost-in-the-middle 対策).
    extra_instructions : str or None
        prompt 末尾に追加するヒント.

    Returns
    -------
    ArticulationResult
        生成 Markdown と metadata.
    """
    components = _format_hypothesis_components(format_hypothesis)
    if show_genome and expert_genome is None:
        raise ValueError("show_genome=True requires expert_genome to be provided")

    model_id = (
        model
        or os.environ.get("TAKUMI_ARTICULATOR_MODEL")
        or "claude-opus-4-7"
    )

    trajectory_block = _render_trajectory_block(
        expert_trajectory, max_episodes=trajectory_episodes, precision=1
    )
    genome_block: Optional[str] = None
    if show_genome and expert_genome is not None:
        # full genome を埋め込む (network は <30 nodes 想定)
        genome_block = (
            _short_genome_summary(expert_genome)
            + "\n\n"
            + _render_genome_block(expert_genome)
        )

    user_blocks = _build_user_blocks(
        components=components,
        format_hypothesis=format_hypothesis,
        show_genome=show_genome,
        trajectory_block=trajectory_block,
        genome_block=genome_block,
        extra_instructions=extra_instructions,
    )

    client = _client()
    last_error: Optional[Exception] = None
    last_text = ""
    last_in_tokens = 0
    last_out_tokens = 0
    last_cache_creation = 0
    last_cache_read = 0
    for attempt in range(max_retries):
        try:
            resp = _create_message(
                client,
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=ARTICULATOR_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_blocks}],
            )
        except Exception as e:  # noqa: BLE001
            last_error = e
            time.sleep(min(2 ** attempt, 8))
            continue
        text_parts = [
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        ]
        last_text = "".join(text_parts).strip()
        usage = resp.usage
        last_in_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        last_out_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        last_cache_creation = int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )
        last_cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        missing = _validate_skills(last_text, components)
        if not missing:
            break
        # retry: 最後の (dynamic) block に validation feedback を追記
        user_blocks[-1]["text"] = (
            user_blocks[-1]["text"]
            + f"\n\n# Validation feedback (attempt {attempt + 1})\n"
            + f"Missing required sections: {missing}. Re-emit the full document with "
            + "all required sections present."
        )
    if last_error is not None and not last_text:
        raise last_error

    cost = 0.0
    if budget is not None:
        entry = budget.add(
            role="articulator",
            model=model_id,
            input_tokens=last_in_tokens,
            output_tokens=last_out_tokens,
            cache_creation_input_tokens=last_cache_creation,
            cache_read_input_tokens=last_cache_read,
            note=(
                f"show_genome={show_genome} components={components} "
                f"cache_write={last_cache_creation} cache_read={last_cache_read}"
            ),
        )
        cost = entry.cost_usd

    return ArticulationResult(
        skills_markdown=last_text,
        components_used=components,
        show_genome=show_genome,
        model=model_id,
        input_tokens=last_in_tokens,
        output_tokens=last_out_tokens,
        cost_usd=cost,
        cache_creation_input_tokens=last_cache_creation,
        cache_read_input_tokens=last_cache_read,
    )


def render_minimal_skills(
    format_hypothesis: Dict[str, Any],
    *,
    extra_note: Optional[str] = None,
) -> str:
    """LLM を呼ばず S1+S2 + 指定 sections の dummy 文書を返す (mock/test 用).

    Articulator が API 不在 / mock テスト時に使う最小実装.
    """
    components = _format_hypothesis_components(format_hypothesis)
    out: List[str] = ["# Skills: SlimeVolley Expert (mock)"]
    for sid in components:
        title = SECTION_TITLES[sid]
        out.append(f"## {sid}. {title}")
        if sid in SECTION_FIXED_TEXTS:
            out.append(SECTION_FIXED_TEXTS[sid])
        elif sid in SECTION_DESCRIPTIONS:
            out.append(f"(mock) {SECTION_DESCRIPTIONS[sid]}")
        else:
            out.append(f"(mock content for {sid})")
    if extra_note:
        out.append("")
        out.append(f"<!-- {extra_note} -->")
    return "\n\n".join(out)
