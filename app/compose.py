"""让模型自己拼算子链

和"查表套用"的本质区别
----------------------
查表：诊断 → DIAGNOSIS_MAP → 预设配方。只在人手写过的组合里选。
拼链：诊断 + 指标 + 已试失败原因 → 模型现场组一条新的。能走出配方库。

为什么敢让它自己拼
------------------
因为出口有闸门：拼出来的链照样要跑、要打分、要过伪影审计、要人核验。
分数不如查表结果就直接丢掉。所以这里是"提案"不是"结论"。

四道约束（在 ops.normalize_stages 里执行）
  - 算子必须在白名单，参数必须在 schema 内、超界钳制
  - 链必须以通道算子开头，长度 1-6
  - 相邻重复算子合并
  - 执行异常整链作废，回退到查表结果
"""

from __future__ import annotations

import json

from . import vlm
from .metrics import GATE
from .ops import SCHEMA, normalize_stages
from .recipes import DIAGNOSES

SYSTEM_PROMPT = """你是工业视觉预处理工程师。给定一张图的成像诊断，你要设计一条图像处理算子链，
目的是让标注员能更容易看清缺陷区域、更快下笔标注。

硬性规则：
1. 只能使用给定算子表中的算子，op 名必须逐字一致，不要自造算子。
2. 第一步必须是 gray 或 channel，把彩色图收成单通道。
3. 总步数 3-5 步最合适，最多 6 步。步数越多越容易堆积伪影。
4. 所有参数必须在给定范围内。
5. 必须避开「已试过的方案」，不要给出相同或等价的组合。
6. 直接输出 JSON 对象本身，不要输出思考过程、不要解释、不要用代码块包裹。
   你的第一个字符必须是 { ，最后一个字符必须是 } 。

硬性闸门（方案跑完要被自动检查，不通过就直接作废，等于白做）：
- 截断率必须 ≤ 10%。把两端压死会丢掉过曝/过暗区的缺陷。
- 输出噪声 σ 必须 ≤ 7.6/255，并且放大倍数不超过 2.5×。
  这是最常见的失败原因，下面单独讲。

关于噪声（务必认真对待）：
这些算子会把噪声一起放大，用得越猛越危险：
  illum_norm（除以低频背景，暗区被猛烈放大）
  homomorphic / retinex（对数域拉伸，暗部噪声跟着抬升）
  clahe（局部均衡，平坦区噪点被拉成麻点）
  usm（锐化，高频噪声直接翻倍）
安全的做法：
- 必须用上面这些算子时，前面先加 median(k=3) 或 gauss(k=3) 压一道噪。
- 能用 gamma 提亮暗部就别用 retinex，gamma 不放大噪声。
- 用 clahe 时 clip 取 1.2-1.5，不要超过 2.0。
- usm 的 amount 不超过 0.4。
- 链路越长，噪声累积越多。4 步往往比 6 步好。

工程经验（这些是踩过的坑，请遵守）：
- 先修光照，再修对比度。顺序反了会把光照梯度一起放大。
- retinex / homomorphic / illum_norm 都是修光照的，同类只用一个，别叠加。
- tophat 会丢掉全局对比，只用它抽局部细节时必须跟一个 normalize。
- usm（锐化）放在最后一步，否则后续拉伸会把 halo 一起放大。
- 最后一句：直接输出 JSON 对象，第一个字符必须是 { ，不要输出思考过程。"""


def _op_catalog() -> str:
    """把算子表压成紧凑文本塞进 prompt。

    每个算子的 hint 是当初写算子时就写好的，直接复用——
    这样加新算子不用回来改 prompt。
    """
    lines = []
    for op, spec in SCHEMA.items():
        ps = spec.get("params") or {}
        if not ps:
            arg = ""
        else:
            parts = []
            for k, sp in ps.items():
                if sp["type"] == "enum":
                    parts.append(f"{k}={'|'.join(sp['options'])}")
                elif sp["type"] == "floats":
                    parts.append(f"{k}=数值数组({sp['min']:g}~{sp['max']:g})")
                elif sp["type"] == "int":
                    parts.append(f"{k}=整数{sp['min']:g}~{sp['max']:g}")
                else:
                    parts.append(f"{k}={sp['min']:g}~{sp['max']:g}")
            arg = "  参数: " + ", ".join(parts)
        lines.append(f"- {op}（{spec['label']}）{spec['hint']}{arg}")
    return "\n".join(lines)


def _fmt_tried(tried: list[dict]) -> str:
    """把已试过的方案列给模型，让它别重复。"""
    if not tried:
        return "（暂无）"
    rows = []
    for r in tried:
        if not r.get("ok"):
            continue
        chain = "→".join(s.get("op", "?") for s in (r.get("stages") or []))
        verdict = (r.get("audit") or {}).get("verdict", "?")
        cn = {"pass": "通过", "warn": "告警", "reject": "被拒"}.get(verdict, verdict)
        why = ""
        reasons = (r.get("audit") or {}).get("reasons") or []
        if reasons:
            why = "；原因：" + reasons[0].get("text", "")
        rows.append(f"- {r.get('recipe_id')} ({chain}) 得分 {r.get('score', 0):+.3f}，{cn}{why}")
    return "\n".join(rows) or "（暂无）"


def compose_recipe(diag: dict, base_m: dict, tried: list[dict], cfg: dict,
                   feedback: str | None = None) -> dict:
    """让模型拼一条新链。

    返回结构固定，失败也有明确原因：
    {"ok","stages","rationale","confidence","notes","error","elapsed_ms"}
    """
    provider = (cfg or {}).get("provider")
    if provider not in ("cloud", "local"):
        return {"ok": False, "stages": [], "error": "当前未启用视觉模型，跳过自主拼链",
                "rationale": "", "confidence": 0, "notes": [], "elapsed_ms": 0}

    ds = diag.get("diagnoses") or []
    ds_cn = "、".join(DIAGNOSES[d]["label"] for d in ds if d in DIAGNOSES) or "未明确"
    metrics_line = (
        f"平均亮度 {base_m.get('mean', 0):.0f}/255，"
        f"1%分位 {base_m.get('p1', 0):.0f}，99%分位 {base_m.get('p99', 0):.0f}，"
        f"动态范围 {base_m.get('range_used', 0):.2f}，"
        f"局部对比度 {base_m.get('local_contrast', 0):.4f}，"
        f"梯度能量 {base_m.get('grad_energy', 0):.4f}，"
        f"噪声 {base_m.get('noise', 0) * 255:.1f}/255，"
        f"直方图熵 {base_m.get('entropy', 0):.2f}，"
        f"截断率 {(base_m.get('clip_low', 0) + base_m.get('clip_high', 0)) * 100:.1f}%"
    )

    n_base = base_m.get("noise", 0.0) * 255
    n_limit = GATE["noise_abs_reject"] * 255
    ratio_lim = GATE["noise_ratio_reject"]
    n_safe = min(n_limit, n_base * ratio_lim)
    if n_safe < 4.0:
        noise_hint = (
            f"原图本身很干净（噪声 σ={n_base:.1f}/255），这意味着你的容忍度很低："
            f"输出 σ 不能超过 {n_safe:.1f}。此时请避开 retinex / illum_norm / homomorphic，"
            f"改用 gamma + normalize + 温和 clahe 这类不放噪声的组合。"
        )
    else:
        noise_hint = (
            f"原图噪声 σ={n_base:.1f}/255，输出上限 σ≤{n_limit:.1f}（放大不超过 {ratio_lim}×），"
            f"你的安全余量约 σ{n_safe:.1f}。"
        )

    user = f"""## 成像诊断
{ds_cn}（标签：{', '.join(ds)}）
观察：{diag.get('observation') or '无'}

## 关键指标
{metrics_line}

## 噪声预算（决定你能用多猛的算子）
{noise_hint}

## 已试过的方案（不要重复）
{_fmt_tried(tried)}

{f"## 上一次尝试被闸门拒绝了{chr(10)}{feedback}{chr(10)}这次请优先保证噪声不超标，宁可少提一点对比度。" if feedback else ""}

## 可用算子表
{_op_catalog()}

请设计一条不同于上面的算子链，输出 JSON：
{{"stages":[{{"op":"gray","p":{{}}}},{{"op":"retina","p":{{"sigmas":[15,80]}}}}],
"rationale":"一句话说明组合思路","confidence":0.0-1.0}}"""

    import time
    t0 = time.time()
    try:
        raw = vlm._call_llm(
            cfg,
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            max_tokens=4000,   # 思考链会吃掉 800-1500，900 必然截断成半截 JSON
            tag="compose",
        )
    except Exception as e:
        return {"ok": False, "stages": [], "rationale": "", "confidence": 0,
                "notes": [], "error": str(e)[:240],
                "elapsed_ms": int((time.time() - t0) * 1000)}

    stages, notes = normalize_stages(raw.get("stages") or [])
    conf = raw.get("confidence", 0)
    try:
        conf = round(max(0.0, min(1.0, float(conf))), 2)
    except (TypeError, ValueError):
        conf = 0.0

    if not stages:
        return {"ok": False, "stages": [], "rationale": str(raw.get("rationale", ""))[:200],
                "confidence": conf, "notes": notes,
                "error": "模型给的算子全部不合法",
                "elapsed_ms": int((time.time() - t0) * 1000)}

    # 和已试方案完全一样就没意义了，标出来但保留（分数可能因参数不同而不同）
    sig = json.dumps([{"op": s["op"], "p": s.get("p", {})} for s in stages],
                     sort_keys=True, ensure_ascii=False)
    for r in tried:
        if not r.get("ok"):
            continue
        old = json.dumps([{"op": s["op"], "p": s.get("p", {})} for s in (r.get("stages") or [])],
                         sort_keys=True, ensure_ascii=False)
        if old == sig:
            notes.append(f"与已试方案 {r.get('recipe_id')} 完全相同")
            break

    return {
        "ok": True,
        "stages": stages,
        "rationale": str(raw.get("rationale", ""))[:240],
        "confidence": conf,
        "notes": notes,
        "error": None,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }
