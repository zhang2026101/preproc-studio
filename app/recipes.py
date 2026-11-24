"""配方库 + 诊断映射表

两条硬约束
----------
1. 配方只允许**几何不变算子**。增强视图上的框要能直接落到原图上，不能有 resize/crop/warp。
   这条写在这里，是给后来改代码的人看的，不是装饰。

2. 诊断映射表（DIAGNOSIS_MAP）是"AI 诊断"与"算子链"之间的唯一桥梁。
   VLM 只输出诊断标签，不输出参数——参数永远来自配方库。
   这样"为什么选这条链"是可追溯的，不是模型临场编的。
"""

from __future__ import annotations

# ---------------------------------------------------------------- 诊断标签

DIAGNOSES = {
    "underexposed": {
        "label": "低照度 / 整体偏暗",
        "desc": "有效码值集中在暗端，细节压在低码值里",
        "icon_hint": "dark",
    },
    "low_contrast": {
        "label": "低对比 / 雾感",
        "desc": "动态范围没铺开，缺陷与背景灰度接近",
        "icon_hint": "flat",
    },
    "uneven_light": {
        "label": "光照不均",
        "desc": "一侧亮一侧暗，或存在大尺度渐变",
        "icon_hint": "gradient",
    },
    "glare": {
        "label": "高光 / 反光过曝",
        "desc": "局部饱和到 255，过曝区内部信息已丢失",
        "icon_hint": "bright",
    },
    "noisy": {
        "label": "噪声偏大",
        "desc": "颗粒感明显，增强时容易被一起放大",
        "icon_hint": "grain",
    },
    "blurry": {
        "label": "偏模糊 / 对焦不足",
        "desc": "边缘不锐，缺陷边界发虚",
        "icon_hint": "soft",
    },
    "normal": {
        "label": "无明显问题",
        "desc": "成像条件尚可，仅需基础拉伸",
        "icon_hint": "ok",
    },
}


# ---------------------------------------------------------------- 配方库
# 全部为几何不变算子链，顺序即执行顺序。

RECIPES = [
    {
        "id": "r00_baseline",
        "desc": "只转灰度，作为对照基线",
        "family": "baseline",
        "stages": [{"op": "gray"}],
    },
    {
        "id": "r01_pct_stretch",
        "desc": "百分位拉伸：码值区间铺满 0-255",
        "family": "tone",
        "stages": [
            {"op": "gray"},
            {"op": "normalize", "p": {"low": 1.0, "high": 99.0}},
        ],
    },
    {
        "id": "r02_stretch_clahe",
        "desc": "拉伸 + 局部直方图均衡",
        "family": "local",
        "stages": [
            {"op": "gray"},
            {"op": "normalize", "p": {"low": 1.0, "high": 99.0}},
            {"op": "clahe", "p": {"clip": 2.0, "grid": 8}},
        ],
    },
    {
        "id": "r03_gamma_stretch",
        "desc": "gamma 提亮暗部后再拉伸",
        "family": "tone",
        "stages": [
            {"op": "gray"},
            {"op": "gamma", "p": {"gamma": 0.45}},
            {"op": "normalize", "p": {"low": 0.5, "high": 99.7}},
        ],
    },
    {
        "id": "r04_illum_norm",
        "desc": "平场校正：除掉低频光照不均",
        "family": "illum",
        "stages": [
            {"op": "gray"},
            {"op": "illum_norm", "p": {"sigma": 61.0}},
            {"op": "normalize", "p": {"low": 0.5, "high": 99.5}},
        ],
    },
    {
        "id": "r05_msr",
        "desc": "多尺度 Retinex：低照度场景主力",
        "family": "illum",
        "stages": [
            {"op": "gray"},
            {"op": "retinex", "p": {"sigmas": [15, 80, 250]}},
            {"op": "normalize", "p": {"low": 1.0, "high": 99.0}},
        ],
    },
    {
        "id": "r06_tophat_usm",
        "desc": "黑帽抽暗细节 + 轻度锐化",
        "family": "local",
        "stages": [
            {"op": "gray"},
            {"op": "normalize", "p": {"low": 1.0, "high": 99.0}},
            {"op": "tophat", "p": {"k": 9, "kind": "black"}},
            {"op": "normalize", "p": {"low": 0.5, "high": 99.5}},
            {"op": "usm", "p": {"amount": 0.4, "radius": 2.0}},
        ],
    },
    {
        "id": "r07_maxch_msr",
        "desc": "取最亮通道再过 Retinex：保住高光里的信息",
        "family": "illum",
        "stages": [
            {"op": "channel", "p": {"mode": "max"}},
            {"op": "retinex", "p": {"sigmas": [15, 80]}},
            {"op": "normalize", "p": {"low": 1.0, "high": 99.0}},
        ],
    },
    {
        "id": "r08_homomorphic",
        "desc": "同态滤波：同时压低频光照、放高频细节",
        "family": "illum",
        "stages": [
            {"op": "gray"},
            {"op": "homomorphic", "p": {"sigma": 31.0, "gain": 1.6}},
            {"op": "normalize", "p": {"low": 5.0, "high": 95.0}},
        ],
    },
    {
        "id": "r09_denoise_stretch",
        "desc": "先中值去噪再拉伸：噪声大的图先治噪",
        "family": "noise",
        "stages": [
            {"op": "gray"},
            {"op": "median", "p": {"k": 3}},
            {"op": "normalize", "p": {"low": 1.0, "high": 99.0}},
            {"op": "clahe", "p": {"clip": 1.5, "grid": 8}},
        ],
    },
    {
        "id": "r10_sharpen_stretch",
        "desc": "拉伸 + 小幅锐化：边缘发虚时用",
        "family": "sharpen",
        "stages": [
            {"op": "gray"},
            {"op": "normalize", "p": {"low": 1.0, "high": 99.0}},
            {"op": "usm", "p": {"amount": 0.5, "radius": 2.0}},
        ],
    },
    {
        "id": "r11_glare_recover",
        "desc": "最亮通道 + 平场 + 拉伸：保高光区信息",
        "family": "illum",
        "stages": [
            {"op": "channel", "p": {"mode": "max"}},
            {"op": "illum_norm", "p": {"sigma": 45.0}},
            {"op": "normalize", "p": {"low": 0.5, "high": 99.5}},
        ],
    },
]

RECIPE_BY_ID = {r["id"]: r for r in RECIPES}


# ---------------------------------------------------------------- 诊断 -> 候选
# 每类诊断最多 3 条候选，顺序即优先级。agent 三次迭代按此顺序消耗。

DIAGNOSIS_MAP = {
    "underexposed": ["r05_msr", "r03_gamma_stretch", "r07_maxch_msr"],
    "low_contrast": ["r01_pct_stretch", "r02_stretch_clahe", "r05_msr"],
    "uneven_light": ["r04_illum_norm", "r05_msr", "r08_homomorphic"],
    "glare": ["r11_glare_recover", "r07_maxch_msr", "r01_pct_stretch"],
    "noisy": ["r09_denoise_stretch", "r01_pct_stretch", "r02_stretch_clahe"],
    "blurry": ["r10_sharpen_stretch", "r01_pct_stretch", "r02_stretch_clahe"],
    "normal": ["r01_pct_stretch", "r02_stretch_clahe", "r06_tophat_usm"],
}

# 多标签同时命中时的合并顺序：先解决"硬伤"（信息丢失类），再谈锦上添花。
DIAG_PRIORITY = ["glare", "underexposed", "noisy", "uneven_light", "blurry", "low_contrast", "normal"]


def candidates_for(diagnoses: list[str], limit: int = 3) -> list[dict]:
    """按诊断标签合并候选配方，去重并按优先级排序。

    多条标签时交错取（每条标签的第一候选优先），避免被单一标签垄断。
    """
    ordered = [d for d in DIAG_PRIORITY if d in diagnoses]
    if not ordered:
        ordered = ["normal"]

    buckets = [list(DIAGNOSIS_MAP.get(d, DIAGNOSIS_MAP["normal"])) for d in ordered]

    picked = []
    seen = set()
    round_idx = 0
    while len(picked) < limit and round_idx < 3:
        for b in buckets:
            if round_idx < len(b):
                rid = b[round_idx]
                if rid not in seen:
                    seen.add(rid)
                    picked.append(rid)
                if len(picked) >= limit:
                    break
        round_idx += 1

    if not picked:
        picked = ["r01_pct_stretch"]

    out = []
    for i, rid in enumerate(picked):
        r = RECIPE_BY_ID.get(rid)
        if not r:
            continue
        out.append({
            "recipe_id": rid,
            "desc": r["desc"],
            "family": r["family"],
            "stages": r["stages"],
            "priority": i + 1,
            "source": "preset",
        })
    return out


def diagnose_by_rules(m: dict) -> list[str]:
    """无 VLM 时的规则兜底诊断。

    只用全局统计量做粗判——它判不准"缺陷是否可见"，但判"成像条件"够用。
    这也是没有 API key 时整条链路仍能跑通的原因。
    注意顺序：先报会造成信息不可逆丢失的（过曝），再报可逆的（偏暗、对比低）。
    """
    hits = []
    mean, rng = m["mean"], m["range_used"]
    if m["clip_high"] > 0.02:
        hits.append("glare")
    if mean < 70 and rng < 0.55:
        hits.append("underexposed")
    if m.get("block_spread", 0.0) > 0.10:
        hits.append("uneven_light")
    if m["noise"] > 0.02:
        hits.append("noisy")
    # 低照度图的梯度能量天然低，不能据此判"模糊"——否则会把每条暗图都推去锐化，
    # 而锐化在低照度图上只会把噪声放大成假边缘。只有动态范围正常时才谈对焦。
    if m["grad_energy"] < 0.02 and rng > 0.5:
        hits.append("blurry")
    if rng < 0.45:
        hits.append("low_contrast")
    if not hits:
        hits.append("normal")
    return hits
