"""算子 DSL —— 预处理流程的最小执行单元

设计约定
--------
1. 配方(recipe)是一张 JSON 配方单：
     {"id": "...", "desc": "...", "stages": [{"op": "...", "p": {...}}, ...]}
   agent 的产物因此可 diff、可缓存、可回放、可入库，而不是"调了个参"。

2. 所有算子签名统一为 fn(img, params) -> img。
   img 为 uint8，单通道(gray)或三通道(BGR)，具体由算子声明。

3. **所有算子均为几何不变算子**（逐像素或逐邻域，不改变图像尺寸与空间对应关系）。
   这是本项目最关键的一条约束：增强视图上的标注框可以直接落到原图上，
   不需要逆变换。任何新增算子若涉及 resize / crop / warp，必须在此处显式声明
   并配套补上逆变换，否则整个"双视图标注"的前提失效。

4. 参数缺省值写在函数签名与 SCHEMA 两处，SCHEMA 供前端自动生成控件。
   答辩被问"这个 31 哪来的"，至少能说清是默认值而不是玄学。
"""

from __future__ import annotations

import cv2
import numpy as np


def _u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def _is_color(img) -> bool:
    return img.ndim == 3 and img.shape[2] == 3


# ---------------------------------------------------------------- 通道类

def op_gray(img, p=None):
    """BGR -> 单通道灰度。"""
    return img if not _is_color(img) else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def op_channel(img, p=None):
    """按通道取值。铜线这类高光在 R/G 更亮的场景，max 往往比灰度更保信息。"""
    p = p or {}
    mode = p.get("mode", "max")
    if not _is_color(img):
        return img
    if mode in ("b", "g", "r"):
        return img[:, :, "bgr".index(mode)]
    if mode == "min":
        return img.min(axis=2)
    return img.max(axis=2)


def op_invert(img, p=None):
    """反相。缺陷为亮、背景为暗的场景用。"""
    return 255 - img


# ---------------------------------------------------------------- 色调类

def op_normalize(img, p=None):
    """百分位拉伸：把实际用到的码值区间铺满 0-255。最基础也最不该省的一步。"""
    p = p or {}
    low, high = float(p.get("low", 1.0)), float(p.get("high", 99.0))
    lo, hi = np.percentile(img, [low, high])
    if hi - lo < 1e-6:
        return img
    return _u8((img.astype(np.float32) - lo) * (255.0 / (hi - lo)))


def op_gamma(img, p=None):
    """gamma<1 提亮暗部，gamma>1 压暗。查表加速。"""
    p = p or {}
    g = float(p.get("gamma", 1.0))
    if abs(g - 1.0) < 1e-9:
        return img
    lut = _u8(((np.arange(256) / 255.0) ** g) * 255.0)
    return cv2.LUT(img, lut)


# ---------------------------------------------------------------- 光照类

def op_illum_norm(img, p=None):
    """平场校正：除以大 sigma 的高斯背景，抹掉低频光照不均。"""
    p = p or {}
    sigma = float(p.get("sigma", 61.0))
    f = img.astype(np.float32)
    bg = cv2.GaussianBlur(f, (0, 0), sigma)
    out = f / np.maximum(bg, 1e-3) * float(np.median(bg))
    return _u8(out)


def op_retinex(img, p=None):
    """多尺度 Retinex (MSR)。低照度、高动态范围图像的主力族。"""
    p = p or {}
    sigmas = p.get("sigmas", [15, 80, 250])
    eps = 1.0

    def single(ch):
        log_ch = np.log10(ch.astype(np.float32) + eps)
        acc = np.zeros_like(log_ch)
        for s in sigmas:
            acc += log_ch - np.log10(cv2.GaussianBlur(ch.astype(np.float32), (0, 0), float(s)) + eps)
        return acc / len(sigmas)

    if _is_color(img):
        chans = [single(img[:, :, i]) for i in range(3)]
        out = np.stack(chans, axis=2)
    else:
        out = single(img)
    return _u8(cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX))


def op_homomorphic(img, p=None):
    """同态滤波：对数域放大高频、压缩低频，光照不均 + 暗部细节同时处理。"""
    p = p or {}
    sigma = float(p.get("sigma", 31.0))
    gain = float(p.get("gain", 1.5))
    f = np.log1p(img.astype(np.float32))
    low = cv2.GaussianBlur(f, (0, 0), sigma)
    out = np.expm1(low + gain * (f - low))
    return _u8(cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX))


# ---------------------------------------------------------------- 滤波类

def op_gauss(img, p=None):
    """高斯平滑。降噪用，但会一起钝掉缺陷边缘。"""
    p = p or {}
    k = int(p.get("k", 3)) | 1
    return cv2.GaussianBlur(img, (k, k), 0)


def op_median(img, p=None):
    """中值滤波。椒盐噪声首选。"""
    p = p or {}
    k = int(p.get("k", 3)) | 1
    return cv2.medianBlur(img, k)


def op_bilateral(img, p=None):
    """双边滤波。保边降噪，代价是慢。"""
    p = p or {}
    return cv2.bilateralFilter(
        img, int(p.get("d", 5)), float(p.get("sigma_color", 50.0)), float(p.get("sigma_space", 50.0))
    )


def op_usm(img, p=None):
    """非锐化掩模。工业图上只用小幅值，过大必然出 halo（伪影来源之一）。"""
    p = p or {}
    amount = float(p.get("amount", 0.6))
    radius = float(p.get("radius", 3.0))
    blur = cv2.GaussianBlur(img.astype(np.float32), (0, 0), radius)
    return _u8(img.astype(np.float32) + amount * (img.astype(np.float32) - blur))


# ---------------------------------------------------------------- 局部对比

def op_clahe(img, p=None):
    """限制对比度的自适应直方图均衡。放大局部对比，也放大噪声。"""
    p = p or {}
    clip = float(p.get("clip", 2.0))
    grid = int(p.get("grid", 8))
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    if _is_color(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cl.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cl.apply(img)


def op_tophat(img, p=None):
    """顶帽/黑帽：抽出比结构元更小的亮/暗细节，对细线状、暗点缺陷有效。"""
    p = p or {}
    k = int(p.get("k", 9))
    kind = p.get("kind", "black")
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if kind == "white":
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, se)


def op_dog(img, p=None):
    """高斯差分：带通，压制低频光照、保留缺陷尺度的边缘。"""
    p = p or {}
    k1, k2 = float(p.get("k1", 1.0)), float(p.get("k2", 3.0))
    g1 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), k1)
    g2 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), k2)
    return _u8((g2 - g1) + 128)


def op_bandpass(img, p=None):
    """频域带通。low/high 为归一化频率(周期/像素)，用于隔离缺陷纹理尺度。"""
    p = p or {}
    low, high = float(p.get("low", 0.02)), float(p.get("high", 0.25))
    f = img.astype(np.float32)
    h, w = f.shape[:2]
    F = np.fft.fftshift(np.fft.fft2(f))
    y, x = np.ogrid[:h, :w]
    d = np.sqrt(((y - h / 2) / h) ** 2 + ((x - w / 2) / w) ** 2)
    F *= ((d >= low) & (d <= high)).astype(np.float32)
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    return _u8(cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX))


OPS = {
    "gray": op_gray,
    "channel": op_channel,
    "invert": op_invert,
    "normalize": op_normalize,
    "gamma": op_gamma,
    "illum_norm": op_illum_norm,
    "retinex": op_retinex,
    "homomorphic": op_homomorphic,
    "gauss": op_gauss,
    "median": op_median,
    "bilateral": op_bilateral,
    "usm": op_usm,
    "clahe": op_clahe,
    "tophat": op_tophat,
    "dog": op_dog,
    "bandpass": op_bandpass,
}


# ---------------------------------------------------------------- 参数 schema
# 前端据此自动生成控件：没有任何参数类型是写死在前端代码里的。
# type: float -> 滑杆 | int -> 数字框(步长1) | enum -> 下拉 | floats -> 多值滑杆

SCHEMA = {
    "gray": {
        "label": "转灰度", "group": "通道", "hint": "BGR 转单通道，无损",
        "params": {},
    },
    "channel": {
        "label": "通道选择", "group": "通道", "hint": "取最亮/最暗通道，保住高光信息",
        "params": {
            "mode": {"type": "enum", "options": ["max", "min", "r", "g", "b"], "default": "max", "label": "取法"},
        },
    },
    "invert": {
        "label": "反相", "group": "通道", "hint": "缺陷为亮背景为暗时使用",
        "params": {},
    },
    "normalize": {
        "label": "百分位拉伸", "group": "色调", "hint": "把实际用到的码值区间铺满 0-255",
        "params": {
            "low": {"type": "float", "min": 0.0, "max": 10.0, "step": 0.1, "default": 1.0, "label": "下分位 %"},
            "high": {"type": "float", "min": 90.0, "max": 100.0, "step": 0.1, "default": 99.0, "label": "上分位 %"},
        },
    },
    "gamma": {
        "label": "gamma", "group": "色调", "hint": "小于 1 提亮暗部",
        "params": {
            "gamma": {"type": "float", "min": 0.1, "max": 3.0, "step": 0.05, "default": 0.45, "label": "gamma"},
        },
    },
    "illum_norm": {
        "label": "平场校正", "group": "光照", "hint": "除掉低频光照不均",
        "params": {
            "sigma": {"type": "float", "min": 10.0, "max": 200.0, "step": 1.0, "default": 61.0, "label": "背景尺度"},
        },
    },
    "retinex": {
        "label": "多尺度 Retinex", "group": "光照", "hint": "低照度场景主力",
        "params": {
            "sigmas": {"type": "floats", "min": 5.0, "max": 400.0, "step": 5.0, "default": [15, 80, 250], "label": "尺度组"},
        },
    },
    "homomorphic": {
        "label": "同态滤波", "group": "光照", "hint": "压低频光照、放高频细节",
        "params": {
            "sigma": {"type": "float", "min": 5.0, "max": 120.0, "step": 1.0, "default": 31.0, "label": "低通尺度"},
            "gain": {"type": "float", "min": 0.5, "max": 4.0, "step": 0.1, "default": 1.5, "label": "高频增益"},
        },
    },
    "gauss": {
        "label": "高斯平滑", "group": "滤波", "hint": "降噪，但会钝化边缘",
        "params": {
            "k": {"type": "int", "min": 1, "max": 21, "step": 2, "default": 3, "label": "核大小"},
        },
    },
    "median": {
        "label": "中值滤波", "group": "滤波", "hint": "椒盐噪声首选",
        "params": {
            "k": {"type": "int", "min": 1, "max": 21, "step": 2, "default": 3, "label": "核大小"},
        },
    },
    "bilateral": {
        "label": "双边滤波", "group": "滤波", "hint": "保边降噪，速度慢",
        "params": {
            "d": {"type": "int", "min": 3, "max": 15, "step": 2, "default": 5, "label": "邻域直径"},
            "sigma_color": {"type": "float", "min": 10.0, "max": 150.0, "step": 5.0, "default": 50.0, "label": "颜色σ"},
        },
    },
    "usm": {
        "label": "非锐化掩模", "group": "锐化", "hint": "小幅值，过大必出 halo",
        "params": {
            "amount": {"type": "float", "min": 0.0, "max": 2.0, "step": 0.05, "default": 0.6, "label": "强度"},
            "radius": {"type": "float", "min": 0.5, "max": 10.0, "step": 0.5, "default": 3.0, "label": "半径"},
        },
    },
    "clahe": {
        "label": "自适应直方图均衡", "group": "局部对比", "hint": "放大局部对比，也放大噪声",
        "params": {
            "clip": {"type": "float", "min": 0.5, "max": 8.0, "step": 0.1, "default": 2.0, "label": "裁剪限"},
            "grid": {"type": "int", "min": 2, "max": 24, "step": 2, "default": 8, "label": "网格数"},
        },
    },
    "tophat": {
        "label": "顶帽 / 黑帽", "group": "局部对比", "hint": "抽细线状、暗点细节",
        "params": {
            "k": {"type": "int", "min": 3, "max": 41, "step": 2, "default": 9, "label": "结构元"},
            "kind": {"type": "enum", "options": ["black", "white"], "default": "black", "label": "类型"},
        },
    },
    "dog": {
        "label": "高斯差分", "group": "局部对比", "hint": "带通，压光照保边缘",
        "params": {
            "k1": {"type": "float", "min": 0.3, "max": 5.0, "step": 0.1, "default": 1.0, "label": "内尺度"},
            "k2": {"type": "float", "min": 1.0, "max": 12.0, "step": 0.5, "default": 3.0, "label": "外尺度"},
        },
    },
    "bandpass": {
        "label": "频域带通", "group": "局部对比", "hint": "隔离缺陷纹理尺度",
        "params": {
            "low": {"type": "float", "min": 0.0, "max": 0.3, "step": 0.005, "default": 0.02, "label": "低截止"},
            "high": {"type": "float", "min": 0.05, "max": 0.8, "step": 0.01, "default": 0.25, "label": "高截止"},
        },
    },
}


def default_params(op: str) -> dict:
    """取算子默认参数，用于"加一个算子"时的初始值。"""
    spec = SCHEMA.get(op, {}).get("params", {})
    return {k: v["default"] for k, v in spec.items()}


def make_stage(op: str, p: dict | None = None) -> dict:
    if op not in OPS:
        raise KeyError(f"未知算子: {op}")
    return {"op": op, "p": default_params(op) if p is None else p}


def apply_recipe(img, recipe):
    """按配方依次执行，返回 (结果图, 执行轨迹)。轨迹用于失败时定位是哪一步炸的。"""
    cur = img
    trace = []
    for stage in recipe.get("stages", []):
        name = stage.get("op")
        if name not in OPS:
            raise KeyError(f"未知算子: {name}")
        before = cur.shape
        cur = OPS[name](cur, stage.get("p") or {})
        trace.append(f"{name}{stage.get('p') or {}} {before}->{cur.shape}")
    return cur, trace


def apply_with_stages(img, recipe):
    """逐步执行并保留每一步的中间结果 —— 界面上的"逐节预览"就靠它。

    返回 (最终图, [(阶段序号, op, 中间图), ...])，第 0 项为原图。
    """
    cur = img
    steps = [(0, "原图", img)]
    for i, stage in enumerate(recipe.get("stages", []), start=1):
        name = stage.get("op")
        if name not in OPS:
            raise KeyError(f"未知算子: {name}")
        cur = OPS[name](cur, stage.get("p") or {})
        steps.append((i, name, cur))
    return cur, steps


# ---------------------------------------------------------------- 参数校验

CHANNEL_OPS = {"gray", "channel"}   # 能把彩色图收成单通道的算子


def clamp_params(op: str, params: dict) -> tuple[dict, list[str]]:
    """按 SCHEMA 校验模型给的参数：非法键丢弃、超范围钳制、类型对齐。

    模型拼链时一定会出现这三种错：
      - 编一个不存在的参数名（如 retinex 的 strength）
      - 数值超界（gamma=0.05）
      - 该给数组时给了标量（sigmas=80）
    不拦的话轻则结果难看，重则算子内部除零崩掉。
    返回 (合法参数, 调整说明)，说明会原样展示给人看。
    """
    spec = SCHEMA.get(op, {}).get("params", {})
    out, notes = {}, []
    for k, v in (params or {}).items():
        if k not in spec:
            notes.append(f"丢弃未知参数 {op}.{k}")
            continue
        sp = spec[k]
        t = sp["type"]

        if t == "enum":
            if v in sp["options"]:
                out[k] = v
            else:
                out[k] = sp["default"]
                notes.append(f"{op}.{k} 取值 {v} 不在 {sp['options']}，改用默认 {sp['default']}")

        elif t == "floats":
            try:
                arr = v if isinstance(v, (list, tuple)) else [v]
                vals = [float(x) for x in arr]
            except (TypeError, ValueError):
                out[k] = sp["default"]
                notes.append(f"{op}.{k} 不是数值数组，改用默认")
                continue
            if not vals:
                out[k] = sp["default"]
                notes.append(f"{op}.{k} 为空，改用默认")
                continue
            out[k] = [round(min(max(x, sp["min"]), sp["max"]), 2) for x in vals]
            if out[k] != [round(float(x), 2) for x in vals]:
                notes.append(f"{op}.{k} 超出 {sp['min']}-{sp['max']}，已钳制")
            out[k] = sorted(set(out[k]))

        else:  # float / int
            try:
                x = float(v)
            except (TypeError, ValueError):
                out[k] = sp["default"]
                notes.append(f"{op}.{k} 不是数值，改用默认")
                continue
            lo, hi = float(sp["min"]), float(sp["max"])
            if x < lo or x > hi:
                notes.append(f"{op}.{k}={x} 超出 {lo}-{hi}，已钳制")
                x = min(max(x, lo), hi)
            out[k] = int(round(x)) if t == "int" else round(x, 3)

    return out, notes


def normalize_stages(stages: list) -> tuple[list, list[str]]:
    """把模型给的一条链整理成能跑的链。

    三条硬规则：
      1. 长度 1-6，空的不行
      2. 必须以通道类算子（gray/channel）开头，否则彩色图会走三通道 Retinex，慢且结果怪
      3. 相邻重复算子合并（模型常连写两个 normalize）
    """
    notes = []
    clean = []
    for st in stages or []:
        if not isinstance(st, dict):
            continue
        op = st.get("op")
        if op not in OPS:
            notes.append(f"丢弃未知算子 {op}")
            continue
        p, n = clamp_params(op, st.get("p") or {})
        notes.extend(n)
        clean.append({"op": op, "p": p})

    merged = []
    for st in clean:
        if merged and merged[-1]["op"] == st["op"] and st["op"] not in ("normalize",):
            notes.append(f"合并相邻重复的 {st['op']}")
            continue
        merged.append(st)

    if len(merged) > 6:
        notes.append(f"链太长（{len(merged)} 步），截断到 6 步")
        merged = merged[:6]

    if merged and merged[0]["op"] not in CHANNEL_OPS:
        notes.append("链首不是通道算子，自动补 gray")
        merged.insert(0, {"op": "gray", "p": {}})

    return merged, notes
