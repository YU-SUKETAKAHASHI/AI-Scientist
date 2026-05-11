"""LLM prompt 用の trajectory 圧縮レンダリング (flow-style + 低精度).

S1 sanity 実測で trajectory block が prompt input token の 47–65% を占めて
いた. このヘルパは保存ファイル (``data/expert_trajectory.yaml``) には触れず、
**LLM に送る瞬間だけ** 以下 2 つの最適化をかける:

1. **Flow-style YAML**: 各 entry を ``{t: 0, o: [...], a: [...]}`` の 1 行
   レンダリングにする (block style だと 1 entry が 14 行に膨らむ).
2. **低精度小数**: 保存時の 3 桁精度を 1 桁に丸める. SlimeVolley の obs は
   既に /10 スケール済みなので 0.1 の精度で力学情報は十分保たれる.

両方合わせて trajectory block を **約 50–60% 圧縮** できる (S1 実測で確認).
"""

from __future__ import annotations

import io
from typing import Any, Dict, Iterable, List


def _round_strip(x: Any, precision: int) -> str:
    """Float を ``precision`` 桁で round し、末尾の ``0``/``.`` を削った文字列を返す.

    例: ``_round_strip(1.234, 1)`` → ``"1.2"``,
        ``_round_strip(1.0, 1)`` → ``"1"``,
        ``_round_strip(0, 1)`` → ``"0"``.
    """
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    try:
        rounded = round(float(x), precision)
    except (TypeError, ValueError):
        return str(x)
    if rounded == int(rounded):
        return str(int(rounded))
    s = f"{rounded:.{precision}f}"
    return s.rstrip("0").rstrip(".") or "0"


def render_trajectory_entries(
    trajectory: Iterable[Dict[str, Any]],
    *,
    precision: int = 1,
    indent: str = "",
) -> str:
    """Trajectory entries を 1-line/entry の compact YAML にレンダリングする.

    出力例 (precision=1)::

        - {t: 0, o: [1.2, 0.2, 0, 0, 0, 1.2, 1.9, 1.6, 1.2, 0.2, 0, 0], a: [0, 0, 0]}
        - {t: 10, o: [...], a: [...]}

    Parameters
    ----------
    trajectory : iterable of dict
        各 entry は ``{t: int, o: list[float, 12], a: list[int, 3]}``.
    precision : int, default 1
        ``o`` の小数桁数.
    indent : str, default ""
        各行の先頭に付ける文字列 (外側 YAML に埋め込むときに使う).

    Returns
    -------
    str
        改行区切りの YAML 文字列. 末尾改行は付かない.
    """
    out: List[str] = []
    for ent in trajectory:
        t = int(ent.get("t", 0))
        o_fmt = [_round_strip(x, precision) for x in (ent.get("o") or [])]
        a_fmt = [str(int(x)) for x in (ent.get("a") or [])]
        out.append(
            f"{indent}- {{t: {t}, o: [{', '.join(o_fmt)}], a: [{', '.join(a_fmt)}]}}"
        )
    return "\n".join(out)


def render_trajectory_bundle(
    bundle: Dict[str, Any],
    *,
    max_episodes: int = 1,
    precision: int = 1,
) -> str:
    """``collect_expert_trajectories`` の bundle を Articulator 用に整形する.

    summary は block style、各 episode の trajectory entries は flow-style.

    Parameters
    ----------
    bundle : dict
        ``{"summary": {...}, "episodes": [...]}``.
    max_episodes : int, default 1
        prompt に乗せる episode 数 (lost-in-the-middle 対策).
    precision : int, default 1
        小数桁数.

    Returns
    -------
    str
        外側は普通の block YAML、trajectory のみ flow style.
    """
    eps = list(bundle.get("episodes") or [])[:max_episodes]
    summary = bundle.get("summary") or {}

    buf = io.StringIO()
    buf.write("summary:\n")
    for k, v in summary.items():
        buf.write(f"  {k}: {_yaml_scalar(v)}\n")
    buf.write("episodes:\n")
    for e in eps:
        buf.write(f"  - episode: {_yaml_scalar(e.get('episode'))}\n")
        buf.write(f"    total_reward: {_yaml_scalar(e.get('total_reward'))}\n")
        buf.write(f"    episode_length: {_yaml_scalar(e.get('episode_length'))}\n")
        buf.write("    trajectory:\n")
        body = render_trajectory_entries(
            e.get("trajectory") or [],
            precision=precision,
            indent="      ",
        )
        if body:
            buf.write(body)
            buf.write("\n")
    return buf.getvalue()


def _yaml_scalar(v: Any) -> str:
    """簡易 scalar レンダラ. list / dict は手抜きで repr."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _round_strip(v, 3)
    if isinstance(v, list):
        # YAML flow list, e.g. [1, 2, 3]
        return "[" + ", ".join(_yaml_scalar(x) for x in v) + "]"
    s = str(v)
    # 単純な英字 + 記号なら quote 不要
    if any(ch in s for ch in ":#&*!|>'\"%@`,[]{}\n") or s != s.strip():
        return repr(s)
    return s
