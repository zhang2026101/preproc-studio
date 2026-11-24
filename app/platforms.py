"""算子到目标流程平台的映射

当前目标：海康 VisionMaster（VM）

为什么这张表要以"能不能直接配"为核心
------------------------------------
流程搭建时最怕的是拿着一份参数单，却发现 VM 里根本没有对应的模块。
所以每个算子都要回答三件事：
  1. VM 里叫什么（模块名）
  2. 是原生模块还是要写脚本
  3. 参数怎么对应

标注说明
--------
how = "native"  VM 有原生模块，按参数填即可
how = "combo"   需要两三个模块串起来，但不是写代码
how = "script"  VM 没有等价模块，必须用脚本模块实现（附参考代码）
how = "none"    实现不了，或在 VM 里没有实用价值

模块名以 VM 4.x 常见叫法为准，不同版本可能略有差异，
所以每条都额外写了「算法语义」，方便在模块树上按语义找。
"""

from __future__ import annotations

# 每个算子：VM 模块、实现方式、参数映射、注意事项、脚本参考
VM_MAP: dict[str, dict] = {
    "gray": {
        "module": "图像转换 · 彩色转灰度",
        "how": "native",
        "params": {},
        "note": "加权系数默认 BT.601（0.299/0.587/0.114），与 OpenCV 一致即可保证结果相同",
        "semantics": "对三通道做加权求和，输出单通道",
    },
    "channel": {
        "module": "图像转换 · 通道分离 / 通道运算",
        "how": "combo",
        "params": {"mode": "取通道：r/g/b；取最大/最小需用通道运算或脚本"},
        "note": "mode=max 在 VM 里没有一键模块，可用两次「通道运算·取最大值」串接，或走脚本",
        "semantics": "按通道取像素值，输出单通道",
    },
    "invert": {
        "module": "图像转换 · 反色",
        "how": "native",
        "params": {},
        "note": "",
        "semantics": "255 - pixel",
    },
    "normalize": {
        "module": "亮度对比度调整 / 灰度拉伸",
        "how": "native",
        "params": {
            "low": "拉伸下限（按百分位算出的灰度值，需先算原图分位）",
            "high": "拉伸上限",
        },
        "note": "VM 的灰度拉伸按绝对灰度值给区间，不是百分位。先把 low/high 的百分位换算成该图的实际灰度，再填。",
        "semantics": "(x - lo) * 255 / (hi - lo)，其中 lo/hi 为指定百分位对应的灰度",
    },
    "gamma": {
        "module": "亮度对比度调整 · Gamma",
        "how": "native",
        "params": {"gamma": "Gamma 系数（VM 可能是 1/gamma 口径，填之前用一张图核对）"},
        "note": "OpenCV 的 gamma<1 提亮；有些软件用反向口径，务必先验一张",
        "semantics": "out = 255 * (x/255)^gamma",
    },
    "illum_norm": {
        "module": "阴影校正",
        "how": "native",
        "params": {"sigma": "背景估计尺度（模块里可能叫「滤波尺寸」或「背景尺寸」）"},
        "note": "VM 的阴影校正通常是「估计背景后相减」，我们是「相除」。"
                "相除对暗部放大更狠、对光照梯度压得更彻底。语义差别在这里，配置后要比一次图。",
        "semantics": "out = x / blur(x, sigma) * median(blur(x, sigma))",
        "script": "bg = cv2.GaussianBlur(img, (0,0), sigma)\nout = np.clip(img / np.maximum(bg, 1e-3) * np.median(bg), 0, 255).astype(np.uint8)",
    },
    "retinex": {
        "module": None,
        "how": "script",
        "params": {"sigmas": "多尺度高斯尺度组，如 [15, 80, 250]"},
        "note": "VM 无等价模块。这是低照度场景的主力算子，建议在脚本模块里实现，注意 log 域运算要用 float32，最后归一化回 8bit。",
        "semantics": "对数域减去多尺度高斯的对数，再归一化",
        "script": "f = img.astype(np.float32) + 1.0\nacc = np.zeros_like(f)\nfor s in sigmas:\n    acc += np.log10(f) - np.log10(cv2.GaussianBlur(f, (0,0), s))\nout = cv2.normalize(acc/len(sigmas), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)",
    },
    "homomorphic": {
        "module": None,
        "how": "script",
        "params": {"sigma": "低通尺度", "gain": "高频增益"},
        "note": "VM 无等价模块。耗时不低（log/exp 逐像素），上产线前务必实测节拍。",
        "semantics": "log 域内压缩低频、放大高频，再 exp 回来",
        "script": "f = np.log1p(img.astype(np.float32))\nlow = cv2.GaussianBlur(f, (0,0), sigma)\nout = np.expm1(low + gain*(f-low))\nout = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)",
    },
    "gauss": {
        "module": "图像滤波 · 高斯",
        "how": "native",
        "params": {"k": "核大小（奇数）或等效半径"},
        "note": "",
        "semantics": "高斯卷积核平滑",
    },
    "median": {
        "module": "图像滤波 · 中值",
        "how": "native",
        "params": {"k": "窗口大小（奇数）"},
        "note": "噪点场景先用它，比高斯保边",
        "semantics": "窗口内取中位数",
    },
    "bilateral": {
        "module": "图像滤波 · 双边（部分版本提供）",
        "how": "native",
        "params": {"d": "邻域直径", "sigma_color": "颜色σ"},
        "note": "若你的版本没有，用脚本。双边在 4K 上很慢，慎用于产线",
        "semantics": "按空间距离与灰度差双权重加权平均",
    },
    "usm": {
        "module": "图像锐化 / 图像增强",
        "how": "native",
        "params": {"amount": "增强强度", "radius": "半径"},
        "note": "VM 锐化可能只有强度没有半径；半径用高斯滤波+差值自己拼也行",
        "semantics": "x + amount * (x - blur(x, radius))",
    },
    "clahe": {
        "module": "直方图均衡 · 局部/自适应（按版本）",
        "how": "native",
        "params": {"clip": "对比度限制（部分版本无此参数，只有全局均衡）", "grid": "分块数"},
        "note": "没有 clip 参数的版本只能用全局均衡，效果会过冲。核对方法：看是否有「限制对比度」或「clipLimit」字样。",
        "semantics": "分块直方图均衡，并对每块直方图做 clip 限幅",
    },
    "tophat": {
        "module": "形态学处理 · 顶帽 / 黑帽",
        "how": "native",
        "params": {"k": "结构元尺寸", "kind": "black=黑帽（抽暗细节）/ white=顶帽（抽亮细节）"},
        "note": "会丢掉全局对比，后面必须跟一步拉伸",
        "semantics": "黑帽 = 闭运算 - 原图；顶帽 = 原图 - 开运算",
    },
    "dog": {
        "module": "图像滤波 · 高斯（组合两次）",
        "how": "combo",
        "params": {"k1": "内尺度", "k2": "外尺度"},
        "note": "VM 无 DoG 模块，用两次高斯相减实现，注意相减后要加 128 平移，否则负值被截断",
        "semantics": "(blur(x,k2) - blur(x,k1)) + 128",
    },
    "bandpass": {
        "module": None,
        "how": "script",
        "params": {"low": "低截止频率", "high": "高截止频率"},
        "note": "频域带通，VM 无对应。除非缺陷有明确周期特征，一般不建议上产线（FFT 耗时不可控）",
        "semantics": "FFT → 环形掩膜 → IFFT",
        "script": "F = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))\ny,x = np.ogrid[:h,:w]\nd = np.sqrt(((y-h/2)/h)**2 + ((x-w/2)/w)**2)\nF *= ((d>=low)&(d<=high))\nout = np.real(np.fft.ifft2(np.fft.ifftshift(F)))\nout = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)",
    },
}


# 分辨率档位：覆盖用户提到的 3K/4K 产线
RESOLUTIONS = [
    ("720p", 1280, 720),
    ("960p（素材基准）", 1280, 960),
    ("1080p", 1920, 1080),
    ("3MP（2048×1536）", 2048, 1536),
    ("5MP（2448×2048）", 2448, 2048),
    ("4K（3840×2160）", 3840, 2160),
    ("12MP（4096×3000）", 4096, 3000),
]


def describe(op: str) -> dict:
    """取一个算子的平台信息，没有就返回"未收录"。"""
    return VM_MAP.get(op) or {
        "module": None,
        "how": "script",
        "params": {},
        "note": "该算子尚未建立 VM 映射，需要手动确认",
        "semantics": "",
    }


def chain_plan(stages: list[dict]) -> list[dict]:
    """把一条链翻译成流程搭建用的节点清单。"""
    out = []
    for i, st in enumerate(stages, start=1):
        op = st.get("op")
        info = describe(op)
        out.append({
            "node": i,
            "op": op,
            "params": st.get("p") or {},
            "vm_module": info["module"],
            "how": info["how"],
            "param_map": info["params"],
            "note": info["note"],
            "semantics": info["semantics"],
            "script": info.get("script", ""),
        })
    return out


def script_only(stages: list[dict]) -> list[str]:
    """列出这条链里必须写脚本的算子，做节拍评估时要特别注意。"""
    return [st["op"] for st in stages if describe(st.get("op"))["how"] == "script"]


# ---------------------------------------------------------------- 全原生改写

NATIVE_ONLY = {"native", "combo"}


def _clamp(v: float, lo: float, hi: float) -> float:
    return round(min(max(float(v), lo), hi), 3)


def is_native(op: str) -> bool:
    return describe(op)["how"] in NATIVE_ONLY


def native_rewrite(stages: list[dict]) -> tuple[list[dict], list[dict]]:
    """把链里"海康没有原生模块"的算子，换成原生组合。

    只处理三个：retinex / homomorphic / bandpass。
    换完不是等价，是近似——所以每一步都记下 from/to，导出时写进配置单，
    让配流程的人知道"这条链跟分析时的不是同一条"。

    返回 (新链, 替换记录)
    """
    out: list[dict] = []
    changes: list[dict] = []

    for st in stages or []:
        op = st.get("op")
        p = st.get("p") or {}

        if op == "retinex":
            # 多尺度 Retinex ≈ 背景校正 + 局部对比增强
            raw = p.get("sigmas") or [15, 80, 250]
            arr = raw if isinstance(raw, (list, tuple)) else [raw]
            vals = sorted(float(x) for x in arr)
            sigma = _clamp(vals[len(vals) // 2], 10, 200)
            out.append({"op": "illum_norm", "p": {"sigma": sigma}})
            out.append({"op": "clahe", "p": {"clip": 1.5, "grid": 8}})
            changes.append({
                "from": "retinex", "to": "illum_norm → clahe",
                "detail": f"背景尺度取多尺度中值 σ={sigma:.0f}，clahe 用温和档 clip=1.5",
                "note": "Retinex 是对数域多尺度，这里换成线性域背景校正 + 局部均衡，暗部提升会弱一些",
            })

        elif op == "homomorphic":
            # 同态滤波 ≈ 背景校正 + 高频锐化
            sigma = _clamp(p.get("sigma") or 30, 10, 200)
            gain = float(p.get("gain") or 1.2)
            out.append({"op": "illum_norm", "p": {"sigma": sigma}})
            out.append({"op": "usm", "p": {"amount": _clamp(gain * 0.4, 0.0, 2.0), "radius": 3.0}})
            changes.append({
                "from": "homomorphic", "to": "illum_norm → usm",
                "detail": f"σ={sigma:.0f}，锐化强度按原高频增益折算 amount={_clamp(gain * 0.4, 0.0, 2.0)}",
                "note": "同态滤波在 log 域同时压低频、提高频；这里拆成两步线性操作，高光溢出风险更低",
            })

        elif op == "bandpass":
            # 频域带通 ≈ DoG（两个高斯相减）
            out.append({"op": "dog", "p": {"k1": 1.0, "k2": 3.0}})
            changes.append({
                "from": "bandpass", "to": "dog",
                "detail": "内尺度 k1=1.0、外尺度 k2=3.0",
                "note": "DoG 是空域近似，选通频段由尺度比决定，不能直接对应原 low/high 频率",
            })

        else:
            out.append(st)

    return out, changes


def script_ops_left(stages: list[dict]) -> list[str]:
    """改写之后还剩下哪些必须写脚本的算子（正常应为空）。"""
    return [st.get("op") for st in stages or [] if not is_native(st.get("op"))]
