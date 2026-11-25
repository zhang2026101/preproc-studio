"""代理指标 + 伪影审计

为什么需要代理指标
------------------
agent 每轮决策都要即时反馈，但下游 ΔRecall 太贵（要训练/推理一整批）。
所以用一组便宜的成像指标做"代理 reward"，**只用于排序和早期淘汰，不能当结论**。

结论永远是：固定下游模型、固定标注预算下的 ΔRecall / 标注耗时。
这一点答辩时必须自己先说清，否则会被问"你这指标跟检测精度什么关系"。

为什么需要伪影审计
------------------
增强算法会把噪声、量化台阶一起放大。标注员看着增强图，可能把一团噪声标成缺陷——
这条**假标签**会直接进训练集，比漏标更毒。所以每条配方过闸门前必须先过审计。

审计只看三件事：
  截断率      两端压死 = 过曝/过暗区的缺陷直接消失
  噪声放大比  拉普拉斯鲁棒方差的比值，反映"无中生有"的风险
  台阶指数    极暗图线性拉伸后，原有量化台阶被放成假边缘
"""

from __future__ import annotations

import cv2
import numpy as np


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def local_contrast(g, win: int = 15) -> float:
    """滑动窗口标准差的均值。比全局 std 更贴近"缺陷所在尺度上还能不能分出来"。"""
    f = g.astype(np.float32)
    mean = cv2.blur(f, (win, win))
    sq = cv2.blur(f * f, (win, win))
    var = np.maximum(sq - mean * mean, 0.0)
    return float(np.sqrt(var).mean() / 255.0)


def grad_energy(g) -> float:
    f = g.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean() / 255.0)


def noise_sigma(g) -> float:
    """拉普拉斯响应的鲁棒标准差(MAD)，对缺陷边缘不敏感，只反映噪声底。"""
    lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)
    med = np.median(lap)
    return float(1.4826 * np.median(np.abs(lap - med)) / 255.0)


def entropy(g) -> float:
    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def banding(g) -> float:
    """量化台阶指数：有效码值区间内空 bin 的占比。

    极暗图做线性拉伸后，原来的量化台阶会被一起放大成假边缘——
    这是纯拉伸方案最大的隐性代价，也是"为什么不能只调对比度"的硬证据。
    """
    p1, p99 = np.percentile(g, [1, 99])
    lo, hi = int(p1), int(p99)
    span = hi - lo + 1
    if span <= 1:
        return 1.0
    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()[lo:hi + 1]
    return float(1.0 - (hist > 0).sum() / span)


def block_spread(g) -> float:
    """3×3 分块均值的离散度，用来粗判"光照不均"。

    全局没有别的量能反映这个——一张整体偏暗但均匀的图，和一张一半黑一半亮的图，
    mean/std 可能几乎一样，但标注难度天差地别。
    """
    h, w = g.shape[:2]
    if h < 12 or w < 12:
        return 0.0
    means = []
    for i in range(3):
        for j in range(3):
            blk = g[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3]
            if blk.size:
                means.append(float(blk.mean()))
    return float(np.std(means) / 255.0)


def measure(img) -> dict:
    g = _gray(img)
    p1, p99 = np.percentile(g, [1, 99])
    return {
        "mean": float(g.mean()),
        "std": float(g.std()),
        "p1": float(p1),
        "p99": float(p99),
        "range_used": float((p99 - p1) / 255.0),
        "clip_low": float((g <= 1).mean()),
        "clip_high": float((g >= 254).mean()),
        "rms_contrast": float(g.std() / 255.0),
        "local_contrast": local_contrast(g),
        "grad_energy": grad_energy(g),
        "noise": noise_sigma(g),
        "entropy": entropy(g),
        "banding": banding(g),
        "block_spread": block_spread(g),
    }


WEIGHTS = {
    "local_contrast": 0.28,
    "grad_energy": 0.18,
    "range_used": 0.14,
    "entropy": 0.40,
    "clip_penalty": -1.50,
}

# 截断率超过这个值才开始扣分：少量截断是正常代价，大面积截断等于把信息压死。
CLIP_TOLERANCE = 0.10


def _gain(ratio: float) -> float:
    """把"相对基线的倍数"映射到以基线为原点、约 -0.5..+1 的量。

    比值 1.0 -> 0（基线得 0 分，符合直觉）
    比值 2.0 -> +0.5    比值 3.0 -> +1.0    比值 0.5 -> -0.25
    """
    return float(np.clip(ratio - 1.0, -1.0, 2.0) / 2.0)


def score(m: dict, base: dict):
    """相对基线的加权得分，基线恒为 0。

    注意 clip 是惩罚项：增强最容易靠"把两端压死"换中间段对比度，
    工业场景上这等于直接丢掉过曝/过暗区域的缺陷，必须扣回去。

    刻意不用"相对基线噪声"做惩罚——极暗量化图的基线噪声底是 0，
    相对比值会退化。噪声和台阶指数只作为诊断列输出。
    """
    def ratio(x, b):
        return x / max(b, 1e-6)

    clip_abs = m["clip_low"] + m["clip_high"]
    parts = {
        "local_contrast": _gain(ratio(m["local_contrast"], base["local_contrast"])),
        "grad_energy": _gain(ratio(m["grad_energy"], base["grad_energy"])),
        "range_used": _gain(ratio(m["range_used"], base["range_used"])),
        "entropy": _gain(ratio(m["entropy"], base["entropy"])),
        "clip_penalty": max(0.0, clip_abs - CLIP_TOLERANCE),
    }
    s = sum(WEIGHTS[k] * v for k, v in parts.items())
    return float(s), parts


# ---------------------------------------------------------------- 伪影审计

GATE = {
    "clip_max": 0.10,        # 截断率上限：超过即判不可用
    "noise_abs_reject": 0.030,  # 绝对噪声上限（σ≈7.6/255，肉眼可见颗粒）
    "noise_abs_warn": 0.020,
    "noise_ratio_reject": 2.5,  # 相对放大倍数（配合绝对值一起判）
    "noise_ratio_warn": 2.0,
    "banding_max": 0.50,     # 台阶指数上限（8bit 图直方图有空隙是常态，0.5 才是真台阶化）
}

# 噪声比的分母下限：极干净的图噪声底接近 0，比值会退化成 0/0。
NOISE_FLOOR = 0.002


def audit_artifact(m: dict, base: dict) -> dict:
    """用增强前后的指标差判断伪影风险。

    噪声判据必须"绝对 + 相对"双条件，只看相对比会误杀：
    Retinex 在噪声底极低的图上会把 0.001 放到 0.017，比值 8.7× 看起来很吓人，
    但 0.017 的抖动（σ≈4/255）对标注其实无害——这是实测踩过的误判。
    只看绝对值又会漏掉"原本就有噪声、被进一步放大"的情况。

    返回 verdict: "pass" | "warn" | "reject"
    """
    clip_abs = m["clip_low"] + m["clip_high"]
    noise_abs = m["noise"]
    noise_ratio = noise_abs / max(base["noise"], NOISE_FLOOR)
    banding_delta = m["banding"] - base["banding"]

    reasons = []
    if clip_abs > GATE["clip_max"]:
        reasons.append(
            {"level": "reject", "code": "clip",
             "text": f"截断率 {clip_abs * 100:.1f}%，超过 {GATE['clip_max'] * 100:.0f}% 上限，两端信息被压死",
             "value": round(clip_abs, 4)}
        )
    if noise_abs > GATE["noise_abs_reject"] and noise_ratio > GATE["noise_ratio_reject"]:
        reasons.append(
            {"level": "reject", "code": "noise",
             "text": f"噪声 σ≈{noise_abs * 255:.1f}/255（放大 {noise_ratio:.1f}×），可能出现虚假瑕疵",
             "value": round(noise_abs, 4)}
        )
    elif noise_abs > GATE["noise_abs_warn"] and noise_ratio > GATE["noise_ratio_warn"]:
        reasons.append(
            {"level": "warn", "code": "noise",
             "text": f"噪声 σ≈{noise_abs * 255:.1f}/255（放大 {noise_ratio:.1f}×），注意别把颗粒标成缺陷",
             "value": round(noise_abs, 4)}
        )
    # 台阶判据用**绝对值**而不是增量。
    # 用增量会让"每一条 normalize 链"都被警告——拉伸本来就会改变空 bin 占比，
    # 警告一泛滥就失去了区分度。真正该拦的是"增强后仍有大量空 bin（≥35%）"。
    if m["banding"] > GATE["banding_max"]:
        reasons.append(
            {"level": "warn", "code": "banding",
             "text": f"台阶指数 {m['banding']:.2f}（原图 {base['banding']:.2f}），量化台阶可能被放成假边缘",
             "value": round(m["banding"], 4)}
        )

    if any(r["level"] == "reject" for r in reasons):
        verdict = "reject"
    elif reasons:
        verdict = "warn"
    else:
        verdict = "pass"

    # 风险分：0 干净，1 不可用。给界面一个直观的条。
    risk = min(1.0, 0.55 * min(clip_abs / GATE["clip_max"], 1.5) / 1.5
               + 0.30 * min(noise_abs / GATE["noise_abs_reject"], 1.5) / 1.5
               + 0.15 * min(max(banding_delta, 0) / GATE["banding_max"], 1.5) / 1.5)

    return {
        "verdict": verdict,
        "risk": round(float(risk), 3),
        "clip_abs": round(clip_abs, 4),
        "noise_abs": round(float(noise_abs), 4),
        "noise_ratio": round(float(noise_ratio), 3),
        "banding_delta": round(float(banding_delta), 4),
        "reasons": reasons,
    }


def delta_summary(m: dict, base: dict) -> dict:
    """给界面用的"相对原图"变化摘要，带方向与百分比。"""
    def pct(x, b):
        if abs(b) < 1e-6:
            return None
        return round((x / b - 1.0) * 100.0, 1)

    return {
        "local_contrast": {"value": round(m["local_contrast"], 4), "delta_pct": pct(m["local_contrast"], base["local_contrast"])},
        "entropy": {"value": round(m["entropy"], 4), "delta_pct": pct(m["entropy"], base["entropy"])},
        "grad_energy": {"value": round(m["grad_energy"], 4), "delta_pct": pct(m["grad_energy"], base["grad_energy"])},
        "mean": {"value": round(m["mean"], 1), "delta_pct": pct(m["mean"], base["mean"])},
        "range_used": {"value": round(m["range_used"], 4), "delta_pct": pct(m["range_used"], base["range_used"])},
        "clip": {"value": round(m["clip_low"] + m["clip_high"], 4), "delta_pct": None},
    }
