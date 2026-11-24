"""FastAPI 应用 —— 预处理工作台

接口分层与界面的五个节点对应：
  诊断 analyze -> 候选 rounds -> 试跑 preview -> 核验 accept/reject -> 导出 export

所有 CPU 密集的路由都用 def（不是 async def），让 FastAPI 丢到线程池，
否则一张 Retinex 就把事件循环堵死，界面会以为卡了。
"""

from __future__ import annotations

import json
import os
import time
import uuid

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import agent as AG
from . import __version__
from . import export as EX
from . import metrics as MT
from . import vlm as VLM
from .ops import SCHEMA, apply_with_stages, default_params
from .recipes import DIAGNOSES, RECIPES, RECIPE_BY_ID, candidates_for
from .store import Store, imread_unicode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
STATIC = os.path.join(ROOT, "static")

store = Store(DATA)
app = FastAPI(title="Preproc Studio", version=__version__)


@app.middleware("http")
async def no_cache_static(request, call_next):
    """静态资源和首页强制每次回源。

    这是个单机演示工具，没有 CDN 也没有 hash 文件名。
    浏览器把 app.js / style.css 缓存住之后，改了代码刷新页面还是旧的，
    表现就是"界面没变"——今天已经因此白排查过一次，所以直接禁缓存。
    """
    resp = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
    return resp


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def cfg() -> dict:
    return VLM.load_config(store.config_path)


# ---------------------------------------------------------------- 静态页

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/health")
def health():
    c = cfg()
    return {
        "ok": True,
        "version": app.version,
        "provider": c.get("provider"),
        "model": c.get("model"),
        "has_key": bool(c.get("api_key")),
        "ops": len(SCHEMA),
        "recipes": len(RECIPES),
        "batches": len(store.list_batches()),
    }


@app.get("/api/schema")
def schema():
    """前端据此生成参数控件 —— 没有任何控件类型写死在前端。"""
    ops = []
    for name, spec in SCHEMA.items():
        ops.append({
            "op": name,
            "label": spec["label"],
            "group": spec["group"],
            "hint": spec["hint"],
            "params": spec["params"],
            "defaults": default_params(name),
        })
    groups = []
    for o in ops:
        if o["group"] not in groups:
            groups.append(o["group"])
    return {
        "ops": ops,
        "groups": groups,
        "diagnoses": DIAGNOSES,
        "recipes": [
            {"id": r["id"], "desc": r["desc"], "family": r["family"], "stages": r["stages"]}
            for r in RECIPES
        ],
        "gate": MT.GATE,
    }


@app.get("/api/presets")
def presets():
    """厂商预设。这些接口都兼容 OpenAI Chat Completions，差别只在
    base_url / model / 图片是 data URL 还是裸 base64 / 认不认 detail。"""
    return {"presets": VLM.PRESETS}


@app.get("/api/config")
def get_config():
    c = cfg()
    # 别把 key 原样吐回去
    safe = dict(c)
    if safe.get("api_key"):
        k = safe["api_key"]
        safe["api_key_masked"] = k[:3] + "..." + k[-4:] if len(k) > 8 else "已设置"
        safe["api_key"] = ""
    safe["presets"] = VLM.PRESETS
    return safe


@app.post("/api/config")
def set_config(payload: dict = Body(...)):
    c = VLM.save_config(store.config_path, payload)
    switched = c.pop("_auto_switched_provider", False)
    out = dict(c)
    out["api_key"] = ""
    return {
        "ok": True,
        "config": out,
        "auto_switched_provider": switched,
        "message": "检测到已填 API Key，诊断来源已自动切到「线上模型」" if switched else "",
    }


@app.post("/api/test_vision")
def test_vision(payload: dict = Body(default={})):
    """连通性检查：发一张 32x32 灰图，只看接口是否通、模型认不认图。

    关键：**用请求里带的表单值，不用磁盘上已保存的配置**。
    否则用户改了地址没点保存就测，测的还是旧配置，会得到莫名其妙的结果。

    detail=low 时一张 32×32 图只有几十个 token，成本可以忽略。
    """
    c = cfg()   # 已存配置打底（含真 key，不会被遮罩）
    for k in ("provider", "base_url", "model", "detail", "image_style"):
        if payload.get(k) not in (None, ""):
            c[k] = payload[k]
    if payload.get("timeout"):
        c["timeout"] = payload["timeout"]
    if payload.get("api_key"):
        c["api_key"] = payload["api_key"]

    if c.get("provider") not in ("cloud", "local"):
        return {
            "ok": False,
            "provider": c.get("provider"),
            "reason": "「诊断来源」是本地规则，不调用任何模型。改成「线上模型」或「本地模型」再测。",
            "hint": "provider=rules",
        }
    if not c.get("base_url") or not c.get("model"):
        return {"ok": False, "provider": c.get("provider"), "reason": "Base URL 或模型名是空的"}
    # 线上服务没 key 必然 401，没必要真的等一次超时
    if c.get("provider") == "cloud" and not (c.get("api_key") or "").strip():
        return {"ok": False, "provider": "cloud", "hint": "no_key",
                "reason": "还没填 API Key。粘贴 sk- 开头的 key 再点测试。"}

    probe = np.full((32, 32, 3), 90, np.uint8)
    r = VLM.diagnose(probe, c, force_provider=c["provider"])

    # 来源选「本地模型」但地址还是公网：请求其实打到了线上，界面却写着本地。
    # 不拦（内网 vLLM 也可能走公网域名），但必须把这件事说出来，
    # 否则"测试连通成功"会给出一个假的"本地可用"结论。
    warning = ""
    if c.get("provider") == "local":
        base = c.get("base_url") or ""
        is_private = any(k in base for k in ("127.0.0.1", "localhost", "0.0.0.0",
                                             "10.", "192.168.", "172.16.", "172.17."))
        if base and not is_private:
            warning = f"诊断来源是本地模型，但 Base URL 指向公网地址（{base}），请求实际打的是这个地址"

    return {
        "ok": r.get("fallback_reason") is None,
        "provider": r["provider"],
        "model": c.get("model"),
        "base_url": c.get("base_url"),
        "elapsed_ms": r["elapsed_ms"],
        "reason": r.get("fallback_reason"),
        "warning": warning,
        "diagnoses": r["diagnoses"],
        "observation": r.get("observation"),
    }


# ---------------------------------------------------------------- 批次

@app.get("/api/batches")
def list_batches():
    return {"batches": store.list_batches()}


@app.post("/api/batches")
def create_batch(payload: dict = Body(...)):
    name = (payload.get("name") or "").strip()
    if not name:
        name = "批次 " + time.strftime("%m-%d %H:%M")
    b = store.create_batch(name, payload.get("note", ""))
    return {"batch": b}


@app.get("/api/batches/{bid}")
def get_batch(bid: str):
    b = store.get_batch(bid)
    if not b:
        raise HTTPException(404, "批次不存在")
    # 样本里的 rounds 含 metrics，直接给；不含 base64（前端按需 preview）
    return {"batch": b, "edit_stats": store.edit_stats(bid)}


@app.delete("/api/batches/{bid}")
def delete_batch(bid: str):
    store.delete_batch(bid)
    return {"ok": True}


@app.post("/api/batches/{bid}/upload")
def upload(bid: str, files: list[UploadFile] = File(...)):
    if not store.get_batch(bid):
        raise HTTPException(404, "批次不存在")
    added, skipped = [], []
    for f in files:
        raw = f.file.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        # 前端选目录时 multipart 里的 filename 可能带相对路径（"铜线/a.png"），
        # 甚至可能带 ../ —— 只保留最后一段，既避免导出拼出子目录，也挡住路径穿越
        name = os.path.basename((f.filename or "").replace("\\", "/")) or \
            f"image_{uuid.uuid4().hex[:6]}.png"
        if img is None:
            skipped.append(name)
            continue
        s = store.add_sample(bid, name, img)
        added.append({"id": s["id"], "filename": s["filename"], "w": s["w"], "h": s["h"]})
    return {"added": added, "skipped": skipped, "count": len(added)}


@app.post("/api/batches/{bid}/import_dir")
def import_dir(bid: str, payload: dict = Body(...)):
    """从本地目录批量导入 —— 比赛材料都是目录形式，逐张点上传太慢。"""
    path = payload.get("path") or ""
    limit = int(payload.get("limit") or 0)
    if not os.path.isdir(path):
        raise HTTPException(400, f"目录不存在: {path}")
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    names = [n for n in sorted(os.listdir(path)) if n.lower().endswith(exts)]
    if limit:
        names = names[:limit]
    added, skipped = [], []
    for n in names:
        img = imread_unicode(os.path.join(path, n))
        if img is None:
            skipped.append(n)
            continue
        s = store.add_sample(bid, n, img)
        added.append({"id": s["id"], "filename": s["filename"], "w": s["w"], "h": s["h"]})
    return {"added": added, "skipped": skipped, "count": len(added)}


@app.delete("/api/batches/{bid}/samples/{sid}")
def delete_sample(bid: str, sid: str):
    b = store.get_batch(bid)
    if not b:
        raise HTTPException(404, "批次不存在")
    s = store.get_sample(bid, sid)
    if s and os.path.exists(s["path"]):
        try:
            os.remove(s["path"])
        except OSError:
            pass
    b["samples"] = [x for x in b["samples"] if x["id"] != sid]
    store.save_batch(b)
    return {"ok": True}


# ---------------------------------------------------------------- 图像

@app.get("/api/batches/{bid}/samples/{sid}/image")
def sample_image(bid: str, sid: str, max_side: int = Query(900), raw: int = Query(0)):
    """raw=1 直接返回图片字节，供 <img src> 用；
    默认返回 JSON（base64），供前端内联渲染大图。

    两个模式都要有：缩略图走 raw 让浏览器自己做缓存，大图走 JSON 省一次请求。
    """
    s = store.get_sample(bid, sid)
    if not s:
        raise HTTPException(404, "样本不存在")
    img = store.load_image(s)
    if img is None:
        raise HTTPException(410, "原图无法读取")

    if raw:
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            sc = max_side / max(h, w)
            img = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise HTTPException(500, "图像编码失败")
        return Response(content=buf.tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=600"})

    return {"image": EX.render_preview(img, max_side=max_side), "w": s["w"], "h": s["h"]}


# ---------------------------------------------------------------- 核心：诊断 + 迭代

def _stages_out(steps, ref_img, base_m, max_side=420):
    """把逐节中间态转成前端可渲染的结构，每步都带"相对原图"的指标。"""
    out = []
    for i, op, im in steps:
        m = MT.measure(im)
        sc, _ = MT.score(m, base_m)
        audit = MT.audit_artifact(m, base_m)
        out.append({
            "step": i,
            "op": op,
            "label": "原图" if i == 0 else SCHEMA.get(op, {}).get("label", op),
            "img": EX.render_preview(im, max_side=max_side),
            "score": round(sc, 4) if i > 0 else 0.0,
            "delta": MT.delta_summary(m, base_m),
            "audit": audit,
        })
    return out


@app.post("/api/batches/{bid}/samples/{sid}/full")
def full_step(bid: str, sid: str, payload: dict = Body(default={})):
    """按需出一张"大图"，只给放大预览用。

    逐节预览的缩略图被压到 420px（12 张一起算，压不小会卡），
    放大看细节时靠这个接口单独算一次，长边默认 1600。
    """
    s = store.get_sample(bid, sid)
    if not s:
        raise HTTPException(404, "样本不存在")
    img = store.load_image(s)
    if img is None:
        raise HTTPException(410, "原图无法读取")

    stages = payload.get("stages")
    if not isinstance(stages, list):
        stages = s.get("stages") or []
    step = int(payload.get("step", len(stages)))
    max_side = int(payload.get("max_side") or 1600)

    work = AG._downscale(img, max_side)
    try:
        _final, steps = apply_with_stages(work, {"stages": stages})
    except Exception as e:
        raise HTTPException(400, f"算子执行失败: {type(e).__name__}: {e}")

    step = max(0, min(step, len(steps) - 1))
    _i, op, im = steps[step]
    return {
        "img": EX.render_preview(im, max_side=max_side),
        "step": step,
        "op": op,
        "w": int(im.shape[1]),
        "h": int(im.shape[0]),
    }


@app.post("/api/batches/{bid}/samples/{sid}/analyze")
def analyze(bid: str, sid: str, payload: dict = Body(default={})):
    """看图 -> 诊断 -> 候选 -> 三次迭代 -> 逐节预览，一次全返回。"""
    s = store.get_sample(bid, sid)
    if not s:
        raise HTTPException(404, "样本不存在")
    img = store.load_image(s)
    if img is None:
        raise HTTPException(410, "原图无法读取")

    force = payload.get("provider") or None
    c = cfg()
    work = AG._downscale(img, 1600)
    base_m = MT.measure(work)

    diag = VLM.diagnose(work, c, base_m=base_m, force_provider=force)
    res = AG.run_agent(
        work, diag,
        cfg=c,
        max_rounds=int(payload.get("max_rounds") or 3),
        accepted=None,
        max_side=1600,
        enable_compose=bool(payload.get("compose", True)),
        enable_tune=bool(payload.get("tune", True)),
    )

    # 逐节预览：基线 + 每条候选的最后一步（完整链太长，只给末态 + 推荐链的全过程）
    panels = [{
        "key": "baseline",
        "recipe_id": "r00_baseline",
        "desc": "原图",
        "img": EX.render_preview(work),
        "score": 0.0,
        "audit": {"verdict": "pass", "risk": 0.0, "reasons": []},
        "delta": MT.delta_summary(base_m, base_m),
    }]

    rounds_public = []
    for r in res["rounds"]:
        if not r.get("ok"):
            rounds_public.append({"round": r["round"], "ok": False, "error": r.get("error"),
                                  "recipe_id": r.get("recipe_id")})
            continue
        steps_pub = _stages_out(r["steps"], work, base_m)
        panels.append({
            "key": r["recipe_id"],
            "recipe_id": r["recipe_id"],
            "desc": r["desc"],
            "img": steps_pub[-1]["img"],
            "score": r["score"],
            "audit": r["audit"],
            "delta": r["delta"],
        })
        rounds_public.append({
            "round": r["round"],
            "ok": True,
            "recipe_id": r["recipe_id"],
            "desc": r["desc"],
            "family": r["family"],
            "score": r["score"],
            "score_parts": r["score_parts"],
            "audit": r["audit"],
            "delta": r["delta"],
            "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in r["metrics"].items()},
            "stages": r["stages"],
            "phase": r.get("phase", "preset"),
            "rationale": r.get("rationale", ""),
            "notes": r.get("notes", []),
            "confidence": r.get("confidence", 0),
            "tuned_from": r.get("tuned_from"),
            "tune_trials": r.get("tune_trials", []),
            "tune_gain": r.get("tune_gain"),
            "steps": steps_pub,
        })

    rec = res["recommendation"]
    selected = rec["stages"] if rec else RECIPE_BY_ID["r01_pct_stretch"]["stages"]
    rec_id = rec["recipe_id"] if rec else "r01_pct_stretch"

    # 落地到样本：AI 初版方案 + 全量轮次（不含 base64）
    slim_rounds = [{k: v for k, v in r.items() if k != "steps"} for r in rounds_public]
    store.update_sample(bid, sid, {
        "diagnosis": diag,
        "rounds": slim_rounds,
        "stages": selected,
        "selected_recipe": rec_id,
        "accepted": False,
        # 重出方案后链是新的，人工修改标记与计数一起归零。
        # 只清 edited 不清 edit_count 会出现"未改"但"改过 7 处"的矛盾状态，
        # 历史改动本身在 edits.jsonl 里，不会丢。
        "edited": False,
        "edit_count": 0,
        "review_required": bool(rec and rec.get("review_required")),
        "note": "闸门全灭，需人工复核" if (rec and rec.get("review_required")) else "",
    })

    chain_ms = EX.time_chain(work, selected) if selected else 0.0
    store.update_sample(bid, sid, {"chain_ms": chain_ms})

    return {
        "diagnosis": diag,
        "chain_ms": chain_ms,
        "baseline": panels[0],
        "panels": panels,
        "rounds": rounds_public,
        "compose": res.get("compose"),
        "tuning": res.get("tuning"),
        "recommendation": rec,
        "stop_reason": res["stop_reason"],
        "elapsed_ms": res["elapsed_ms"],
        "candidates_tried": res["candidates_tried"],
        "selected_recipe": rec_id,
        "stages": selected,
    }


@app.post("/api/batches/{bid}/samples/{sid}/preview")
def preview(bid: str, sid: str, payload: dict = Body(...)):
    """按前端当前编辑的 stages 重新算逐节预览。

    拖动滑杆时前端只调这个接口、并且只对当前这张图调用 —— 整批等"应用"再跑。
    """
    s = store.get_sample(bid, sid)
    if not s:
        raise HTTPException(404, "样本不存在")
    img = store.load_image(s)
    if img is None:
        raise HTTPException(410, "原图无法读取")

    stages = payload.get("stages")
    if not isinstance(stages, list):
        stages = s.get("stages") or []
    work = AG._downscale(img, 1600)
    base_m = MT.measure(work)

    try:
        final, steps = apply_with_stages(work, {"stages": stages})
    except Exception as e:
        raise HTTPException(400, f"算子执行失败: {type(e).__name__}: {e}")

    steps_pub = _stages_out(steps, work, base_m)
    m = MT.measure(final)
    sc, parts = MT.score(m, base_m)
    audit = MT.audit_artifact(m, base_m)

    return {
        "steps": steps_pub,
        "final": {
            "img": steps_pub[-1]["img"],
            "score": round(sc, 4),
            "score_parts": {k: round(v, 4) for k, v in parts.items()},
            "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
            "delta": MT.delta_summary(m, base_m),
            "audit": audit,
        },
    }


@app.post("/api/batches/{bid}/samples/{sid}/stages")
def save_stages(bid: str, sid: str, payload: dict = Body(...)):
    """保存人工修改，并写进改动流水。

    kind: add / remove / move / param / reset / ai
    这条流水是"AI 越用越准"唯一的语料来源，不能省。
    """
    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise HTTPException(400, "stages 必须是数组")
    s = store.get_sample(bid, sid)
    if not s:
        raise HTTPException(404, "样本不存在")

    old = s.get("stages") or []
    kind = payload.get("kind") or "param"
    store.update_sample(bid, sid, {
        "stages": stages,
        "edited": True,
        "edit_count": int(s.get("edit_count", 0)) + 1,
        "selected_recipe": payload.get("recipe_id") or "custom",
    })
    store.log_edit({
        "batch_id": bid,
        "sample_id": sid,
        "filename": s["filename"],
        "kind": kind,
        "op": payload.get("op"),
        "recipe_id": payload.get("recipe_id"),
        "before": old,
        "after": stages,
        "note": payload.get("note", ""),
    })
    return {"ok": True, "edit_count": int(s.get("edit_count", 0)) + 1}


@app.post("/api/batches/{bid}/samples/{sid}/accept")
def accept(bid: str, sid: str, payload: dict = Body(default={})):
    """人工核验通过 —— 这是整个流程里唯一的"真值"来源。"""
    s = store.get_sample(bid, sid)
    if not s:
        raise HTTPException(404, "样本不存在")
    patch = {"accepted": True, "accepted_at": time.time()}
    if payload.get("stages"):
        patch["stages"] = payload["stages"]
    store.update_sample(bid, sid, patch)
    store.log_edit({
        "batch_id": bid, "sample_id": sid, "filename": s["filename"],
        "kind": "accept", "recipe_id": payload.get("recipe_id") or s.get("selected_recipe"),
        "note": payload.get("note", ""),
    })
    return {"ok": True}


@app.post("/api/batches/{bid}/samples/{sid}/switch")
def switch_candidate(bid: str, sid: str, payload: dict = Body(...)):
    """在已算出的候选之间直接切换（相当于底部的 1/2/3 按钮）。

    和 /stages 的区别：这是"选一个已有方案"，不是"改方案"，
    所以不增加 edit_count、不标 edited —— 否则每切一次都算一次人工修改，
    "改了几处"这个指标就失去意义了。
    """
    s = store.get_sample(bid, sid)
    if not s:
        raise HTTPException(404, "样本不存在")

    rid = payload.get("recipe_id")
    rounds = [r for r in (s.get("rounds") or []) if r.get("ok") and r.get("stages")]
    r = next((x for x in rounds if x.get("recipe_id") == rid), None)
    if r is None:
        raise HTTPException(404, "候选不存在或已失效，请重新生成方案")

    patch = {
        "stages": r["stages"],
        "selected_recipe": rid,
        "phase": r.get("phase", "preset"),
        "rationale": r.get("rationale", ""),
    }
    # 换了一条链，之前"核验通过"就失效了 —— 核验针对的是具体某条链
    if s.get("accepted"):
        patch["accepted"] = False

    # 切到被闸门拒绝的链，不拦，但必须留痕：
    # 之前切过去什么都没发生，那条被拒的链会安静地成为该诊断组的唯一方案被导出，
    # 配置单里只有逐张明细里一个"拒绝"字样，配流程的人不会注意到。
    verdict = (r.get("audit") or {}).get("verdict")
    warning = ""
    if verdict == "reject":
        patch["review_required"] = True
        patch["note"] = "当前候选未通过伪影闸门，导出前需人工复核"
        warning = "这条候选未通过伪影闸门（噪声或截断超限），已按你的选择切换，导出前请人工确认"
    else:
        # 切回通过/告警的链，把"闸门全灭需复核"的标记撤掉（标记描述的是当前这条链）
        patch["review_required"] = False
        if (s.get("note") or "").startswith(("闸门全灭", "当前候选未通过")):
            patch["note"] = ""

    store.update_sample(bid, sid, patch)
    store.log_edit({
        "batch_id": bid, "sample_id": sid, "filename": s["filename"],
        "kind": "switch", "recipe_id": rid,
        "note": f"候选切换至 {rid}" + (f"（{verdict}，需复核）" if verdict == "reject" else ""),
    })
    return {
        "ok": True,
        "recipe_id": rid,
        "score": r.get("score"),
        "audit": r.get("audit"),
        "verdict": verdict,
        "warning": warning,
        "stages": r["stages"],
        "phase": r.get("phase", "preset"),
        "accepted_cleared": bool(s.get("accepted")),
    }


@app.post("/api/batches/{bid}/samples/{sid}/reject")
def reject(bid: str, sid: str, payload: dict = Body(default={})):
    """核验不通过。前端拿到 next 后自动切到该候选，这就是"回环第 2 步"。"""
    s = store.get_sample(bid, sid)
    if not s:
        raise HTTPException(404, "样本不存在")
    store.log_edit({
        "batch_id": bid, "sample_id": sid, "filename": s["filename"],
        "kind": "reject", "op": payload.get("reason"), "note": payload.get("note", ""),
    })
    rounds = [r for r in (s.get("rounds") or []) if r.get("ok")]

    # 找下一条还没被拒过的候选。
    # nxt 必须先置 None：候选耗尽时循环一次都不执行，
    # 原来直接 `if nxt:` 会抛 UnboundLocalError，前端看到的就是 500 —— 连点两次就"坏掉"。
    rejected = set(payload.get("rejected") or [])
    nxt = None
    for r in rounds:
        if r["recipe_id"] in rejected:
            continue
        if r["recipe_id"] == s.get("selected_recipe"):
            continue
        nxt = r
        break

    if nxt:
        store.update_sample(bid, sid, {"stages": nxt["stages"], "selected_recipe": nxt["recipe_id"]})
        return {"ok": True, "next": {"recipe_id": nxt["recipe_id"], "desc": nxt["desc"],
                                     "stages": nxt["stages"], "score": nxt["score"],
                                     "audit": nxt["audit"]}}
    return {"ok": True, "next": None, "message": "候选已用尽，请人工指定算子链"}


# ---------------------------------------------------------------- 批量

@app.post("/api/batches/{bid}/run_all")
def run_all(bid: str, payload: dict = Body(default={})):
    """整批跑一遍诊断 + 迭代。每张都留诊断与轮次记录，但只落最终方案。"""
    b = store.get_batch(bid)
    if not b:
        raise HTTPException(404, "批次不存在")
    c = cfg()
    provider = payload.get("provider") or None
    results = []
    timing_jobs = []

    for s in b["samples"]:
        img = store.load_image(s)
        if img is None:
            results.append({"id": s["id"], "ok": False, "error": "原图无法读取"})
            continue
        work = AG._downscale(img, 1600)
        base_m = MT.measure(work)
        diag = VLM.diagnose(work, c, base_m=base_m, force_provider=provider)
        res = AG.run_agent(work, diag, cfg=c, max_rounds=3, max_side=1600,
                           enable_compose=bool(payload.get("compose", True)),
                           enable_tune=bool(payload.get("tune", True)))
        rec = res["recommendation"]
        if rec:
            patch = {
                "diagnosis": diag,
                "rounds": [{k: v for k, v in r.items() if k != "steps"} for r in res["rounds"]],
                "stages": rec["stages"],
                "selected_recipe": rec["recipe_id"],
                # 和单张分析保持一致：整批重出方案也会换掉链，
                # 改过标记与计数要一起重置，否则两张图的"已改"口径不同
                "edited": False,
                "edit_count": 0,
                "review_required": bool(rec.get("review_required")),
                "note": "闸门全灭，需人工复核" if rec.get("review_required") else "",
            }
        else:
            patch = {"diagnosis": diag,
                     "rounds": [{k: v for k, v in r.items() if k != "steps"} for r in res["rounds"]],
                     "stages": RECIPE_BY_ID["r01_pct_stretch"]["stages"],
                     "selected_recipe": "r01_pct_stretch",
                     "edited": False,
                     "edit_count": 0,
                     "review_required": True,
                     "note": "全部候选执行失败"}
        # 整批分析同样要量一次链耗时，否则界面上的「纯算子 xx ms」一直是空的
        timing_jobs.append((s["id"], work, patch.get("stages") or []))
        store.update_sample(bid, s["id"], patch)
        results.append({
            "id": s["id"], "filename": s["filename"], "ok": True,
            "recipe_id": patch["selected_recipe"],
            "score": (rec or {}).get("score", 0.0),
            "verdict": (rec or {}).get("audit", {}).get("verdict", "reject"),
            "diagnoses": diag["diagnoses"],
        })

    # 耗时统一放到整批跑完之后再量。
    # 紧跟在调参/拼链后面测，CPU 还在满负荷，实测虚高约 1.7 倍：
    # 同一张图界面显示 1000 ms，而配置单里是 580 ms，两个数对不上，
    # 站上讲台被问"到底多少毫秒"就说不清了。
    for sid_, tw, tstages in timing_jobs:
        try:
            ms = EX.time_chain(tw, tstages) if tstages else 0.0
        except Exception:
            ms = 0.0
        store.update_sample(bid, sid_, {"chain_ms": ms})

    return {"results": results, "count": len(results)}


@app.post("/api/batches/{bid}/aggregate")
def aggregate(bid: str):
    b = store.get_batch(bid)
    if not b:
        raise HTTPException(404, "批次不存在")
    items = []
    for s in b["samples"]:
        items.append({
            "sample_id": s["id"],
            "filename": s["filename"],
            "diagnosis": (s.get("diagnosis") or {}).get("diagnoses") or ["normal"],
            "recipe_id": s.get("selected_recipe") or "r01_pct_stretch",
            "score": next((r.get("score") for r in (s.get("rounds") or [])
                           if r.get("recipe_id") == s.get("selected_recipe")), 0.0),
            "accepted": s.get("accepted"),
        })
    agg = AG.aggregate_batch(items)
    b["aggregate"] = agg
    b["plan"] = agg["groups"]
    store.save_batch(b)
    return {"aggregate": agg}


# ---------------------------------------------------------------- 导出

@app.post("/api/batches/{bid}/export")
def export_batch(bid: str, payload: dict = Body(default={})):
    b = store.get_batch(bid)
    if not b:
        raise HTTPException(404, "批次不存在")
    if not b["samples"]:
        raise HTTPException(400, "批次为空")

    only_accepted = bool(payload.get("only_accepted"))
    only_native = bool(payload.get("only_native"))
    with_json = bool(payload.get("with_json"))
    if only_accepted and not any(s.get("accepted") for s in b["samples"]):
        raise HTTPException(400, "还没有通过核验的样本，无法只导出已核验的图")
    # 每次都重算聚合。
    # 缓存过一次"聚合成方案"之后就复用旧结果，会出这种事故：
    # 聚合 → 又改了两张图的链 → 导出，包里用的是改之前的分组，
    # 而且从文档上看不出来。12 张图重算只要几百毫秒，没必要缓存。
    agg_items = []
    for s in b["samples"]:
        agg_items.append({
            "sample_id": s["id"], "filename": s["filename"],
            "diagnosis": (s.get("diagnosis") or {}).get("diagnoses") or ["normal"],
            "recipe_id": s.get("selected_recipe") or "r01_pct_stretch",
        })
    b["aggregate"] = AG.aggregate_batch(agg_items)
    store.save_batch(b)

    t0 = time.time()
    manifest = EX.build_export(store, b, store.exports_dir,
                               only_accepted=only_accepted, only_native=only_native,
                               with_json=with_json)
    zpath = EX.zip_export(store.exports_dir, bid)

    pipelines = manifest.get("pipeline") or []
    script_nodes = sum(len(p.get("script_nodes") or []) for p in pipelines)
    base_ms = [p["timing"]["base_ms"] for p in pipelines if p.get("timing")]
    return {
        "ok": True,
        "zip": os.path.basename(zpath),
        "download": f"/api/batches/{bid}/export/download",
        "items": len(manifest["items"]),
        "pipelines": len(pipelines),
        "script_nodes": script_nodes,
        "slowest_ms": max(base_ms) if base_ms else 0,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "only_accepted": only_accepted,
        "only_native": only_native,
        "with_json": with_json,
        "native_swaps": sum(len(p.get("native_swaps") or []) for p in pipelines),
        # 样本级计数：和前端勾选项里那句"N 处脚本算子"口径一致（按张数算）
        "native_swaps_items": sum(len(it.get("native_swaps") or [])
                                  for it in (manifest.get("items") or [])),
        "accepted": sum(1 for s in b["samples"] if s.get("accepted")),
        "skipped_unverified": (manifest.get("export_filter") or {}).get("skipped_unverified", 0),
    }


@app.get("/api/batches/{bid}/export/download")
def download(bid: str):
    zpath = os.path.join(store.exports_dir, f"{bid}.zip")
    if not os.path.exists(zpath):
        raise HTTPException(404, "还没有导出过，请先点导出")
    return FileResponse(zpath, filename=f"preproc_{bid}.zip", media_type="application/zip")


# ---------------------------------------------------------------- 算子准入表 / 统计

@app.post("/api/preflight")
def preflight(payload: dict = Body(...)):
    """所有算子链在若干张图上跑一遍伪影审计，产出算子准入表。

    不依赖 VLM、不依赖人工 —— 这是唯一能纯自动生成的结论表。
    """
    bid = payload.get("batch_id")
    limit = int(payload.get("limit") or 4)
    imgs = []
    if bid:
        b = store.get_batch(bid)
        if not b:
            raise HTTPException(404, "批次不存在")
        for s in b["samples"][:limit]:
            im = store.load_image(s)
            if im is not None:
                imgs.append(im)
    else:
        for p in (payload.get("paths") or [])[:limit]:
            im = imread_unicode(p)
            if im is not None:
                imgs.append(im)
    if not imgs:
        raise HTTPException(400, "没有可用图像")
    return AG.preflight(imgs, max_side=int(payload.get("max_side") or 900))


@app.get("/api/batches/{bid}/stats")
def stats(bid: str):
    b = store.get_batch(bid)
    if not b:
        raise HTTPException(404, "批次不存在")
    ss = b["samples"]
    total = len(ss)
    diag_count: dict[str, int] = {}
    verdict_count = {"pass": 0, "warn": 0, "reject": 0}
    for s in ss:
        for d in (s.get("diagnosis") or {}).get("diagnoses", []):
            diag_count[d] = diag_count.get(d, 0) + 1
        r = next((x for x in (s.get("rounds") or [])
                  if x.get("recipe_id") == s.get("selected_recipe")),
                 (s.get("rounds") or [{}])[0])
        v = ((r or {}).get("audit") or {}).get("verdict")
        if v in verdict_count:
            verdict_count[v] += 1
    return {
        "total": total,
        "accepted": sum(1 for s in ss if s.get("accepted")),
        "edited": sum(1 for s in ss if s.get("edited")),
        "analyzed": sum(1 for s in ss if s.get("diagnosis")),
        "by_diagnosis": diag_count,
        "by_verdict": verdict_count,
        "edits": store.edit_stats(bid),
    }


@app.get("/api/edits")
def edits(batch_id: str | None = None, limit: int = 200):
    return {"edits": store.read_edits(limit=limit, batch_id=batch_id)}


@app.get("/api/op_default")
def op_default(op: str):
    if op not in SCHEMA:
        raise HTTPException(404, "算子不存在")
    return {"op": op, "params": default_params(op), "schema": SCHEMA[op]}
