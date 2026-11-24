"""AI 诊断层

职责边界（这条必须守住）
------------------------
VLM **只做诊断**：说出这张图的成像问题是什么（偏暗？反光？噪声大？）。
VLM **不出参数**：绝不让模型回答"用 clip=2.0、grid=8"。
原因很实际——模型对参数没有像素级校准能力，它给的数字是编的，
而且每次不一样，不可复现，现场演示必翻车。

参数永远来自 recipes.py 的配方库。这样"为什么选这条链"是可追溯的。

三种 provider
-------------
cloud   线上 OpenAI 兼容接口（演示用，效果最好）
local   公司内网 vLLM / Ollama，同样 OpenAI 兼容协议（正式部署用）
rules   无模型，用全局统计量兜底（没网没 key 也能跑通全流程）

三者的输出结构完全一致，上层 agent 不区分来源。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time

import cv2
import numpy as np

from . import metrics as MT
from .recipes import DIAGNOSES, diagnose_by_rules

SYSTEM_PROMPT = """你是工业视觉成像诊断助手。你的任务只有一个：判断这张工业检测图像的成像质量问题。

严格规则：
1. 你只输出诊断标签，绝对不要输出任何处理参数（如 clip、sigma、gamma 的数值）。
2. 只从给定标签表中选择，可以多选，按严重程度排序。
3. 看不出来就选 normal，不要硬凑。
4. 你无法判断"是否存在缺陷"，也不需要判断——只描述成像条件。

标签表：
- underexposed  低照度/整体偏暗，细节压在暗端
- low_contrast  低对比/雾感，缺陷与背景灰度接近
- uneven_light  光照不均，一侧亮一侧暗或存在大尺度渐变
- glare         高光/反光过曝，局部饱和到白
- noisy         噪声偏大，颗粒感明显
- blurry        偏模糊/对焦不足，边缘发虚
- normal        无明显问题

只返回 JSON，不要任何解释文字：
{"diagnoses": ["标签", ...], "observation": "一句话描述你看到的成像状况", "confidence": 0.0-1.0}
"""

_CACHE: dict[str, dict] = {}


def _img_hash(img: np.ndarray) -> str:
    """按缩小后的灰度内容做指纹，用于诊断缓存（同图不重复调 API）。"""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)
    return hashlib.md5(small.tobytes()).hexdigest()


def _to_data_url(img: np.ndarray, max_side: int = 1024) -> str:
    """转 base64 data URL。

    长边压到 1024 只为省 token。视觉模型在推理前都会把图缩到约 1300×1300 等效，
    传更大的没有收益，只会拖慢上传。
    """
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("图像编码失败")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------- 厂商预设
# 这些厂商都兼容 OpenAI Chat Completions，差别只在 base_url / model /
# 图片是 data URL 还是裸 base64 / 认不认 detail 字段。
# 预设的意义：让人不用记这些坑。

PRESETS = {
    "deepseek": {
        "label": "DeepSeek（国内直连）",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "image_style": "data_url",
        "detail": "low",
        "note": "V4.1 Flash 原生多模态，OpenAI 兼容。detail=low 会降到 512×512，诊断成像条件足够",
    },
    "qwen": {
        "label": "通义千问（阿里百炼）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "image_style": "data_url",
        "detail": "low",
        "note": "百炼兼容模式，使用 sk- 开头的 key",
    },
    "zhipu": {
        "label": "智谱 GLM-4V",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4v-flash",
        "image_style": "raw_base64",
        "detail": "",
        "note": "图片要用裸 base64，带 data: 前缀会被拒；不支持 detail",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "image_style": "data_url",
        "detail": "low",
        "note": "国内需代理",
    },
    "local": {
        "label": "本地 vLLM / Ollama",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "Qwen2.5-VL-7B-Instruct",
        "image_style": "data_url",
        "detail": "low",
        "note": "内网部署用，协议与线上一致，key 可留空",
    },
}


def _parse_json(text: str) -> dict:
    """模型偶尔会裹一层 ```json，容错剥掉。

    失败时抛的是带原文片段的 RuntimeError，不是裸的 JSONDecodeError——
    否则界面上只会看到 "line 1 column 1 (char 0)"，根本不知道模型回了什么。
    """
    if not text or not text.strip():
        raise RuntimeError("模型返回了空字符串，没有内容可解析")
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    t = t.strip().strip("`")
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        t = t[start:end + 1]
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"模型返回的不是 JSON：{text.strip()[:180]}") from e


def _log_raw(tag: str, obj) -> None:
    """把失败的原始响应留档，便于事后定位是模型抽风还是我们解析错了。"""
    try:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "vlm_debug.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, "raw": obj},
                          ensure_ascii=False)[:4000]
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _extract_text(data: dict) -> tuple[str, dict]:
    """从响应里取出正文，并返回诊断信息。

    空 content 是这个接口最常见的坑：
      - 模型开了思考模式，max_tokens 全被思考链吃掉，正文为空
      - finish_reason=length 就是明证
    这时若厂商把思考链放在 reasoning_content 里，退而求其次读它。
    """
    ch = (data.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):   # 少数实现返回 block 数组
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    info = {
        "finish_reason": ch.get("finish_reason"),
        "usage": data.get("usage"),
        "empty": not bool(content),
    }
    if not content:
        content = msg.get("reasoning_content") or msg.get("reasoning") or ""
        info["from_reasoning"] = bool(content)
    return (content or ""), info


def _sanitize(raw: dict) -> dict:
    """只保留合法标签。模型编出来的标签一律丢弃，宁可变 normal。"""
    ds = raw.get("diagnoses") or []
    if isinstance(ds, str):
        ds = [ds]
    valid = [d for d in ds if d in DIAGNOSES]
    seen = []
    for d in valid:
        if d not in seen:
            seen.append(d)
    if not seen:
        seen = ["normal"]
    conf = raw.get("confidence", 0.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "diagnoses": seen,
        "observation": str(raw.get("observation", ""))[:200],
        "confidence": round(max(0.0, min(1.0, conf)), 2),
    }


def _build_image_block(img: np.ndarray, cfg: dict) -> dict:
    """按厂商差异拼图片块。

    差异只有两处，但踩过一次就很难受：
      data_url   OpenAI / DeepSeek / 百炼 / vLLM 认 `data:image/jpeg;base64,...`
      raw_base64 智谱 GLM 只认裸 base64，带前缀直接 400
    """
    style = cfg.get("image_style") or "data_url"
    url = _to_data_url(img, max_side=int(cfg.get("image_max_side") or 1024))
    if style == "raw_base64":
        url = url.split(",", 1)[1]
    block = {"type": "image_url", "image_url": {"url": url}}
    detail = (cfg.get("detail") or "").strip()
    if detail and style == "data_url":
        block["image_url"]["detail"] = detail
    return block


def _friendly_http_error(code: int, body: str) -> str:
    hint = {
        400: "请求格式被拒。若模型不支持图片输入，或 image_style 与本厂商不匹配，都会返回 400",
        401: "API Key 无效或已过期",
        402: "账户余额不足",
        403: "无权访问该模型，检查是否已开通",
        404: "接口地址或模型名不对：base_url 通常要以 /v1 结尾（智谱为 /v4），model 必须与平台一致",
        413: "请求体过大，图片太多或太大",
        429: "触发限流，稍后重试或降低并发",
        500: "服务端错误，稍后重试",
        503: "服务暂不可用",
    }.get(code, "未知错误")
    return f"HTTP {code} · {hint}｜{body[:200]}"


def _call_llm(cfg: dict, messages: list, max_tokens: int = 1200, tag: str = "call") -> dict:
    """通用 Chat Completions 调用，返回解析后的 JSON object。

    诊断和拼链都走这里，省得两套几乎一样的降级逻辑各修一遍。
    三级降级：JSON 模式 → 纯文本 → 再试一次。
    每次失败都把原始响应写进 data/vlm_debug.log，
    否则只能对着 "line 1 column 1" 猜模型到底回了什么。
    """
    import urllib.error
    import urllib.request

    model = cfg.get("model") or "deepseek-flash"
    base = (cfg.get("base_url") or "https://api.deepseek.com").rstrip("/")
    key = cfg.get("api_key") or ""
    timeout = float(cfg.get("timeout", 60))

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": int(max_tokens),
        "messages": messages,
    }

    # 厂商特有参数（例如关掉思考链）放在配置的 extra_body 里。
    # 猜字段名的风险太大——不同厂商叫 thinking / reasoning_effort / enable_thinking，
    # 塞错了有的忽略、有的直接 400。所以做成可配置，且 400 时自动去掉重试。
    extra = {}
    raw_extra = (cfg.get("extra_body") or "").strip()
    if raw_extra:
        try:
            extra = json.loads(raw_extra)
            if isinstance(extra, dict):
                payload.update(extra)
            else:
                extra = {}
        except json.JSONDecodeError:
            extra = {}

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def call(body: dict) -> dict:
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "ignore")[:400]
            raise RuntimeError(_friendly_http_error(e.code, raw)) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"连不上 {base}：{e.reason}") from e

    plans = [
        ("json_mode", {**payload, "response_format": {"type": "json_object"}}),
        ("plain", payload),
        ("plain_retry", payload),
    ]
    if extra:
        # 最后兜底：去掉厂商特有参数，退回最标准的请求体
        bare = {k: v for k, v in payload.items() if k not in extra}
        plans.append(("no_extra", bare))
    errors = []
    for plan_name, body in plans:
        try:
            data = call(body)
        except RuntimeError as e:
            errors.append(f"{plan_name}: {e}")
            _log_raw(f"{tag}:{plan_name}:http", {"error": str(e)})
            continue

        text, info = _extract_text(data)
        if not text.strip():
            errors.append(f"{plan_name}: 空响应 finish_reason={info.get('finish_reason')}")
            _log_raw(f"{tag}:{plan_name}:empty", {"info": info, "data": data})
            continue
        try:
            return _parse_json(text)
        except RuntimeError as e:
            errors.append(f"{plan_name}: {e}")
            _log_raw(f"{tag}:{plan_name}:parse", {"info": info, "text": text[:600]})
            continue

    raise RuntimeError("；".join(errors[-2:]) or "接口未返回可用结果")


def diagnose_cloud(img: np.ndarray, cfg: dict) -> dict:
    """看图诊断。只输出成像问题标签，不出任何处理参数。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "诊断这张工业检测图像的成像质量。"},
                _build_image_block(img, cfg),
            ],
        },
    ]
    raw = _call_llm(cfg, messages, int(cfg.get("max_tokens") or 2000), tag="diagnose")
    return _sanitize(raw)


def diagnose(
    img: np.ndarray,
    cfg: dict | None = None,
    base_m: dict | None = None,
    force_provider: str | None = None,
) -> dict:
    """统一入口：按配置走 cloud / local，失败自动降级到 rules。

    返回结构固定：
    {"provider","diagnoses","observation","confidence","elapsed_ms","fallback_reason"}
    """
    cfg = cfg or {}
    provider = force_provider or cfg.get("provider") or "rules"
    t0 = time.time()

    if provider in ("cloud", "local"):
        # 缓存键必须带上"哪个模型"，不能只按图片指纹：
        # 换服务商/换模型后如果命中上一家的缓存，会拿线上结果冒充本地结果
        # （诊断卡上还写着 provider=cloud），连通性测试也就永远"通过"。
        h = _img_hash(img)
        ckey = f"{provider}|{cfg.get('base_url') or ''}|{cfg.get('model') or ''}|{h}"
        cached = _CACHE.get(ckey)
        if cached:
            out = dict(cached)
            out["cached"] = True
            out["elapsed_ms"] = 0
            return out

        try:
            raw = diagnose_cloud(img, cfg)
            out = {
                "provider": provider,
                "diagnoses": raw["diagnoses"],
                "observation": raw["observation"],
                "confidence": raw["confidence"],
                "cached": False,
                "fallback_reason": None,
            }
            _CACHE[ckey] = out
            out = dict(out)
            out["elapsed_ms"] = int((time.time() - t0) * 1000)
            return out
        except Exception as e:  # 网络/额度/格式，任何问题都不该阻塞流程
            reason = f"{type(e).__name__}: {str(e)[:160]}"
            m = base_m or MT.measure(img)
            out = {
                "provider": "rules",
                "diagnoses": diagnose_by_rules(m),
                "observation": "视觉接口不可用，已改用全局统计量兜底判断。",
                "confidence": 0.0,
                "cached": False,
                "fallback_reason": reason,
                "requested_provider": provider,
            }
            out["elapsed_ms"] = int((time.time() - t0) * 1000)
            return out

    m = base_m or MT.measure(img)
    out = {
        "provider": "rules",
        "diagnoses": diagnose_by_rules(m),
        "observation": "使用规则诊断（未配置视觉模型）。",
        "confidence": 0.0,
        "cached": False,
        "fallback_reason": None,
    }
    out["elapsed_ms"] = int((time.time() - t0) * 1000)
    return out


# ---------------------------------------------------------------- 配置

DEFAULT_CONFIG = {
    "provider": "rules",
    "preset": "deepseek",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-flash",
    "api_key": "",
    "timeout": 60,
    "detail": "low",          # low=缩到512×512，诊断成像条件足够且省 token；空串=不传
    "image_style": "data_url",  # data_url | raw_base64
    "image_max_side": 1024,
    "max_tokens": 2000,
    # 厂商特有参数，JSON 字符串。例如关思考：{"thinking": {"type": "disabled"}}
    # 不同厂商字段名不同（thinking / reasoning_effort / enable_thinking），
    # 所以做成可配置而不是硬编码；塞错了会自动去掉重试。
    "extra_body": "",
}


def load_config(path: str) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    # 环境变量优先，便于演示机临时切线上 key 而不改文件
    if os.environ.get("PREPROC_API_KEY"):
        cfg["api_key"] = os.environ["PREPROC_API_KEY"]
        cfg["provider"] = os.environ.get("PREPROC_PROVIDER", "cloud")
    if os.environ.get("PREPROC_BASE_URL"):
        cfg["base_url"] = os.environ["PREPROC_BASE_URL"]
    if os.environ.get("PREPROC_MODEL"):
        cfg["model"] = os.environ["PREPROC_MODEL"]
    return cfg


def save_config(path: str, cfg: dict) -> dict:
    cur = load_config(path)
    for k in DEFAULT_CONFIG:
        if k in cfg:
            cur[k] = cfg[k]
    # 选了预设但没单独填地址，就按预设补齐，省得人记 base_url
    preset = cfg.get("preset")
    if preset in PRESETS and not cfg.get("base_url"):
        p = PRESETS[preset]
        cur["base_url"] = p["base_url"]
        cur["model"] = p["model"]
        cur["image_style"] = p["image_style"]
        cur["detail"] = p["detail"]

    # 只在"调用方这次没明确指定来源"时才做纠正。
    # 之前是无条件纠正：只要填了 key 就强制切回线上模型，
    # 结果用户选了「本地规则」点保存，实际被改回线上——选择权被吞了。
    # 明确传了 provider 就完全尊重（断网演示、想省钱时都可能是刻意选规则）。
    switched = False
    if "provider" not in cfg and cur.get("provider") == "rules" \
            and (cur.get("api_key") or "").strip():
        cur["provider"] = "cloud"
        switched = True

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2)

    out = dict(cur)
    out["_auto_switched_provider"] = switched   # 只给调用方看，不落盘
    return out
