"""参数精修（贪心坐标搜索）

为什么不让模型调参，而是搜索
----------------------------
模型对参数没有像素级校准能力，"clip=2.3" 和 "clip=2.0" 对它没区别。
但**在给定链里选哪个值更好，是可以测出来的**。所以这一层用测的，不用猜的。

搜索方式
--------
逐 stage、逐参数做一维扫描，每改一个其它保持不变（坐标下降）。
不做组合网格，16 个算子的组合会爆炸，而且边际收益极低。

三个预算一起卡：
  budget       最多评估几次（默认 6）
  max_seconds  最多花多少秒（默认 4）
  小图上很快、大图上 retinex 很慢，所以时间预算比次数预算更重要
"""

from __future__ import annotations

import copy
import time

import numpy as np

from . import metrics as MT
from .ops import apply_recipe

# 只搜"有物理意义、影响明显"的参数。
# 每个 stage 只扫第一个命中的参数，避免一条链上搜出十几次评估。
TUNE_GRID = {
    "retinex": {"sigmas": [[10, 60, 180], [15, 80, 250], [25, 120, 350], [15, 80, 250, 400]]},
    "normalize": {"low": [0.3, 1.0, 2.0], "high": [99.0, 99.7]},
    "clahe": {"clip": [1.2, 2.0, 3.0]},
    "tophat": {"k": [7, 9, 13]},
    "gamma": {"gamma": [0.35, 0.45, 0.6]},
    "illum_norm": {"sigma": [40.0, 61.0, 90.0]},
    "homomorphic": {"gain": [1.2, 1.6, 2.0]},
    "usm": {"amount": [0.3, 0.6, 0.9]},
    "median": {"k": [3, 5]},
    "gauss": {"k": [3, 5]},
    "bilateral": {"sigma_color": [30.0, 50.0, 80.0]},
    "dog": {"k2": [2.0, 3.0, 4.5]},
    "bandpass": {"high": [0.15, 0.25, 0.4]},
}

# 审计等级，用来判断"提分但引入伪影"该不该采纳
RANK = {"pass": 0, "warn": 1, "reject": 2}

# 从"不合格"往"合格"救的时候，允许分数掉这么多。
# 否则会卡死：一条被截断判死的链分数很高，任何能把截断压下去的参数
# 都会让分数下降，于是永远不采纳，这条链就白白废了。
RESCUE_SLACK = 0.12


def _is_better(new_s: float, new_a: dict, cur_s: float, cur_a: dict, min_gain: float) -> bool:
    """采纳判据。

    分两种情况，方向完全不同：
      1. 审计等级能改善  -> 优先救活，允许分数小幅下降（RESCUE_SLACK）
      2. 审计等级不变    -> 只认实打实的提分（超过 min_gain）
    审计变差一律不采纳，哪怕分数涨得很多——那正是"把噪声标成缺陷"的来源。
    """
    nr, cr = RANK.get(new_a["verdict"], 9), RANK.get(cur_a["verdict"], 9)
    if nr < cr:
        return new_s > cur_s - RESCUE_SLACK
    if nr == cr:
        return new_s > cur_s + min_gain
    return False


def _eval(img: np.ndarray, stages: list, base_m: dict) -> tuple[float, dict, str | None]:
    try:
        out, _ = apply_recipe(img, {"stages": stages})
    except Exception as e:
        return -9.9, {"verdict": "reject", "risk": 1.0, "reasons": []}, f"{type(e).__name__}: {e}"
    m = MT.measure(out)
    s, _ = MT.score(m, base_m)
    a = MT.audit_artifact(m, base_m)
    return float(s), a, None


def tune(img: np.ndarray, stages: list, base_m: dict,
         budget: int = 6, max_seconds: float = 4.0, min_gain: float = 0.005) -> dict:
    """在给定链上做参数精修，返回新链和尝试记录。

    采纳条件（两个都要满足）：
      1. 分数提升超过 min_gain
      2. 审计等级不劣化 —— 提分但把图搞出伪影，不如不改
    """
    t0 = time.time()
    cur = copy.deepcopy(stages or [])
    if not cur:
        return {"improved": False, "stages": cur, "score": 0.0, "base_score": 0.0,
                "trials": [], "elapsed_ms": 0, "reason": "空链"}

    best_score, best_audit, err = _eval(img, cur, base_m)
    base_score = best_score
    if err:
        return {"improved": False, "stages": cur, "score": best_score, "base_score": base_score,
                "trials": [], "elapsed_ms": 0, "reason": f"当前链已无法执行：{err}"}

    trials = []
    used = 0
    stop_reason = "搜完"

    for i, st in enumerate(cur):
        grid = TUNE_GRID.get(st.get("op"))
        if not grid:
            continue
        if used >= budget or (time.time() - t0) > max_seconds:
            stop_reason = "预算用尽"
            break

        p = st.get("p") or {}
        pkey = next((k for k in grid if k in p), None)
        if pkey is None:
            continue

        cur_val = p[pkey]
        for v in grid[pkey]:
            if v == cur_val:
                continue
            if used >= budget or (time.time() - t0) > max_seconds:
                stop_reason = "预算用尽"
                break

            trial = copy.deepcopy(cur)
            trial[i]["p"][pkey] = v
            s, a, e = _eval(img, trial, base_m)
            used += 1

            better = (e is None
                      and _is_better(s, a, best_score, best_audit, min_gain))
            trials.append({
                "rescued": bool(e is None and RANK.get(a["verdict"], 9) < RANK.get(best_audit["verdict"], 9)),
                "stage": i + 1,
                "op": st["op"],
                "param": pkey,
                "from": cur_val,
                "to": v,
                "score": round(s, 4),
                "verdict": a["verdict"],
                "accepted": bool(better),
                "error": e,
            })
            if better:
                cur = trial
                cur_val = v
                best_score, best_audit = s, a

    improved = best_score > base_score + 1e-6
    return {
        "improved": improved,
        "stages": cur,
        "score": round(best_score, 4),
        "base_score": round(base_score, 4),
        "gain": round(best_score - base_score, 4),
        "trials": trials,
        "evaluations": used,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "stop_reason": stop_reason,
        "reason": None,
    }
