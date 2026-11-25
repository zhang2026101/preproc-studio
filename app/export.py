"""导出：流程配置 + 调试样本 + 对比图

重心变了
--------
原来按"标注辅助"设计：增强图给标注员看，标签落原图。
现在按"流程搭建"设计：算子链要真的配进海康 VisionMaster，所以导出物是

  流程配置.md     一页纸，按节点列出算子、参数、VM 对应模块、耗时
  pipeline.json   机读版配置单，含多分辨率耗时估算、必须写脚本的节点
  debug_samples/  增强图，导入流程软件做验证用（不是给标注员看的）
  compare/        原图|增强图并排，一眼看效果
  manifest.json   逐张对照索引，留作追溯

几何不变依然重要，但理由变了：不是为了让标注框对齐，
而是流程里配的算子必须和调试时看到的完全一致——中间偷偷 resize 一次，产线就配不出来。

耗时
----
流程搭建最关心节拍。这里实测每张链的执行时间，换算成 ms/百万像素，
再按常见产线分辨率外推。样本分辨率与产线不同时，外推值会标注「估算」。
"""

from __future__ import annotations

import json
import os
import time
import zipfile

import cv2
import numpy as np

from . import metrics as MT
from . import __version__
from . import platforms as PF
from .ops import apply_recipe, apply_with_stages
from .recipes import DIAGNOSES, RECIPE_BY_ID
from .store import imwrite_unicode


def _stages_of(sample: dict) -> list[dict]:
    if sample.get("stages"):
        return sample["stages"]
    rid = sample.get("selected_recipe")
    if rid and rid in RECIPE_BY_ID:
        return RECIPE_BY_ID[rid]["stages"]
    return []


def _stage_sig(stages: list[dict]) -> str:
    """算子链签名（算子 + 参数，键序无关）。

    AI 自主拼接的样本 recipe_id 都叫 ai_composed，参数却各不相同，
    只按 id 归类会把几条不同的链写成一个参数表。
    """
    return json.dumps(stages or [], sort_keys=True, ensure_ascii=False)


def time_chain(img: np.ndarray, stages: list[dict], repeat: int = 5) -> float:
    """测一条链在给定图像上的耗时（毫秒）。

    两个坑都踩过：
    1. 首次执行有 OpenCV 缓冲分配开销，直接计时会虚高 2-3 倍 —— 先预热一次。
    2. 如果调用方刚跑完一堆重计算（比如 analyze 里的三候选 + 拼链 + 精修），
       CPU 还在被抢占，测出来能到真实值的十几倍。
       所以这里多次采样取最小值，并让调用方在测之前先停一下。
    """
    recipe = {"stages": stages}
    try:
        apply_recipe(img, recipe)
    except Exception:
        return 0.0

    time.sleep(0.04)   # 给调度器一点时间，把刚才的重负载消化掉
    best = float("inf")
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        apply_recipe(img, recipe)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return round(best, 2)


def _fit_h(img: np.ndarray, h: int) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    s = h / ih
    return cv2.resize(img, (max(1, int(iw * s)), h), interpolation=cv2.INTER_AREA)


def make_compare(orig: np.ndarray, enh: np.ndarray, dst: str, h: int = 640) -> bool:
    """原图 | 增强图 并排，中间留白缝。

    流程搭建时要拿它跟 VM 里跑出来的结果对眼睛核对，
    所以两张必须同高、边缘对齐，不能各自缩放后错位。
    """
    try:
        a, b = _fit_h(orig, h), _fit_h(enh, h)
        gap = 14
        canvas = np.full((h, a.shape[1] + b.shape[1] + gap, 3), 245, np.uint8)
        canvas[:, :a.shape[1]] = a
        canvas[:, a.shape[1] + gap:] = b
        for x, label in ((14, "ORIGINAL"), (a.shape[1] + gap + 14, "ENHANCED")):
            cv2.putText(canvas, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (250, 250, 250), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (30, 30, 30), 1, cv2.LINE_AA)
        return imwrite_unicode(dst, canvas)
    except Exception:
        return False


def _mp(w: int, h: int) -> float:
    return max(w * h / 1e6, 1e-6)


def timing_table(ms: float, w: int, h: int) -> dict:
    """把实测耗时换算成 ms/MP，再按常见分辨率外推。

    小图（<0.2 MP）的 ms/MP 不能用：固定开销（内存分配、首次缓存）占比过大，
    192×192 上测出的 0.4 ms 折成 11 ms/MP，而 1280×960 上同族链是 57 ms/MP，
    直接外推会把 4K 耗时低估一个数量级。这种情况给一条 note，让看的人知道别照抄。
    """
    per_mp = ms / _mp(w, h)
    rows = []
    for name, rw, rh in PF.RESOLUTIONS:
        est = per_mp * _mp(rw, rh)
        same = (rw == w and rh == h)
        rows.append({
            "label": name,
            "w": rw,
            "h": rh,
            "ms": round(ms if same else est, 1),
            "measured": same,
        })
    small = (w * h) < 200000
    return {
        "base_w": w,
        "base_h": h,
        "base_ms": round(ms, 1),
        "ms_per_mp": round(per_mp, 2),
        "rows": rows,
        "note": (f"基准图为 {w}×{h} 的小尺寸样本，ms/百万像素 受固定开销影响偏大，"
                 f"其余分辨率的预估仅供粗估" if small else ""),
    }


def _pick_timing(rows: list[tuple[float, int, int]]) -> tuple[float, int, int] | None:
    """从同一方案的若干样本里挑一个耗时基准，返回 (中位耗时 ms, w, h)。

    为什么不能只用"该方案第一张样本"：
    1. 混合分辨率批次里，两条方案的基准图不同，耗时表就不能横向比较；
    2. 小图（如 192×192）的 ms/百万像素 被固定开销与缓存效应放大，
       直接外推到 4K 会明显偏，所以按面积优先取最大的一档。
    同一档内取中位数，避免个别抖动值拉偏（原实现取的就是中位数，保留这个口径）。
    """
    usable = [(ms, w, h) for ms, w, h in rows if ms and ms > 0 and w and h]
    if not usable:
        return None
    groups: dict[tuple[int, int], list[float]] = {}
    for ms, w, h in usable:
        groups.setdefault((w, h), []).append(float(ms))
    (w, h), times = max(groups.items(),
                        key=lambda kv: (kv[0][0] * kv[0][1], len(kv[1])))
    times.sort()
    return (times[len(times) // 2], w, h)


def build_export(store, batch: dict, out_dir: str, only_accepted: bool = False,
                 only_native: bool = False, with_json: bool = False) -> dict:
    """产出流程配置包。

    only_accepted=True 时只导出已通过人工核验的样本——
    未核验的链配进产线是有风险的，所以给调用方一个"只交确认过的"选项。

    only_native=True 时把 retinex / homomorphic / bandpass 换成海康原生组合，
    保证整条链在 VM 里没有脚本节点。换的是近似，替换明细会写进配置单。

    with_json=False（默认）只产出人要看的东西：配置单 + 对照图 + 增强图。
    JSON 是给脚本用的，交付时默认不给，免得拿包的人先被五个文件劝退。
    """
    import shutil

    bid = batch["id"]
    base = os.path.join(out_dir, bid)

    # 清掉上一版留下的旧目录，否则 enhanced/ 和 debug_samples/ 会同时存在，
    # 流程搭建的人不知道该用哪个
    for name in ("enhanced", "debug_samples", "compare"):
        d = os.path.join(base, name)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    stale_md = os.path.join(base, "方案说明.md")
    if os.path.exists(stale_md):
        os.remove(stale_md)

    enh_dir = os.path.join(base, "debug_samples")
    cmp_dir = os.path.join(base, "compare")
    os.makedirs(enh_dir, exist_ok=True)
    os.makedirs(cmp_dir, exist_ok=True)

    native_notes: list[dict] = []

    items = []
    used: dict[str, int] = {}
    # 每条链在每张样本上的耗时与分辨率。按样本累计，不按"第一条"定基准，
    # 否则混合分辨率批次里两条方案的耗时基准会不可比（详见 _pick_timing）。
    # 归组的键是「recipe_id + 算子序列」，参数不参与：
    # 参数精修会给每张图微调出不同参数，若按参数归组，12 张图能出 6 个方案，
    # 产线不可能每张图配一套参数。同一算子序列只算一个方案，参数取组内多数那套，
    # 个别被精修过的样本在逐张明细里标出来。
    chain_rows: dict[tuple, list[tuple[float, int, int]]] = {}
    chain_cover: dict[tuple, int] = {}
    chain_variants: dict[tuple, dict[str, list]] = {}
    ckey_by_sample: dict[str, tuple] = {}
    psig_by_sample: dict[str, str] = {}

    all_samples = batch.get("samples", [])
    samples = [x for x in all_samples if x.get("accepted")] if only_accepted else all_samples
    skipped = len(all_samples) - len(samples)

    for s in samples:
        src = store.load_image(s)
        if src is None:
            items.append({"sample_id": s["id"], "filename": s["filename"],
                          "error": "原图无法读取"})
            continue

        raw_stages = _stages_of(s)
        stages = raw_stages
        swaps: list[dict] = []
        if only_native:
            stages, swaps = PF.native_rewrite(raw_stages)
        recipe = {"id": s.get("selected_recipe") or "custom", "stages": stages}
        try:
            out, step_list = apply_with_stages(src, recipe)
            err = None
        except Exception as e:
            out, step_list, err = src, [], f"{type(e).__name__}: {e}"

        rid = s.get("selected_recipe") or "custom"
        # 归组键用改写前的原始链：改写只是交付形式，不该影响"这是不是同一条链"
        ops = ">".join(str(st.get("op")) for st in raw_stages)
        ckey = (rid, ops)
        psig = _stage_sig(raw_stages)
        chain_cover[ckey] = chain_cover.get(ckey, 0) + 1
        slot = chain_variants.setdefault(ckey, {}).setdefault(psig, [0, raw_stages])
        slot[0] += 1
        ckey_by_sample[s["id"]] = ckey
        psig_by_sample[s["id"]] = psig
        ms = time_chain(src, stages) if stages else 0.0
        chain_rows.setdefault(ckey, []).append((ms, src.shape[1], src.shape[0]))

        stem, ext = os.path.splitext(s["filename"])
        name = f"{stem}{ext if ext else '.png'}"
        if name in used:
            used[name] += 1
            name = f"{stem}_{used[name]}{ext if ext else '.png'}"
        else:
            used[name] = 0

        imwrite_unicode(os.path.join(enh_dir, name), out)
        make_compare(src, out, os.path.join(cmp_dir, f"{stem}.png"))

        m = MT.measure(out)
        base_m = MT.measure(src)
        audit = MT.audit_artifact(m, base_m)
        score, _ = MT.score(m, base_m)

        items.append({
            "sample_id": s["id"],
            "filename": s["filename"],
            "enhanced_file": name,
            "compare_file": f"{stem}.png",
            "recipe_id": rid,
            "stages": stages,
            "diagnosis": (s.get("diagnosis") or {}).get("diagnoses", []),
            "diagnosis_observation": (s.get("diagnosis") or {}).get("observation", ""),
            "provider": (s.get("diagnosis") or {}).get("provider", ""),
            "accepted": bool(s.get("accepted")),
            "edited": bool(s.get("edited")),
            "edit_count": s.get("edit_count", 0),
            "native_swaps": swaps,
            "score": round(score, 4),
            "audit": audit,
            "ms": ms,
            "resolution": [src.shape[1], src.shape[0]],
            "error": err,
        })

    # ---------------- 方案清单 ----------------
    # 两类方案：
    #   group  按诊断分组的投票 winner（一批一份的主方案）
    #   solo   分组覆盖不到的实际链：人工切换、手改、或 AI 自主拼接的链
    # 只出 group 会有这种事故：逐张明细里写着 r01_pct_stretch，
    # 方案章节里找它的节点参数却找不到，配流程的人只能猜。
    #
    # 一节对应一条「算子序列」，参数取组内多数样本那套：
    # AI 自主拼接的样本 recipe_id 都叫 ai_composed，算子序列各不相同，
    # 按 id 归成一节会把几条不同的链写成一个参数表，配出来只有一条是对的。
    pipelines: list[dict] = []
    plan_groups = (batch.get("aggregate") or {}).get("groups", [])
    used_keys: set[tuple] = set()
    rep_sig: dict[tuple, str] = {}

    def _add_pipeline(key: tuple, desc: str, consistency: float, kind: str) -> None:
        rid, _ops = key
        variants = chain_variants.get(key) or {}
        stages: list[dict] = []
        if variants:
            # 出现次数最多的那套参数作为方案基准；并列时取先出现的
            sig, (cnt, rep) = max(variants.items(), key=lambda kv: kv[1][0])
            stages, rep_sig[key] = rep, sig
        covers = chain_cover.get(key, 0)
        p_swaps: list[dict] = []
        if only_native:
            stages, p_swaps = PF.native_rewrite(stages)
            for c in p_swaps:
                if c not in native_notes:
                    native_notes.append(c)
        t = _pick_timing(chain_rows.get(key) or [])
        pipelines.append({
            "recipe_id": rid,
            "desc": desc,
            "kind": kind,
            "covers": covers,
            "consistency": consistency,
            "stages": stages,
            "nodes": PF.chain_plan(stages),
            "script_nodes": PF.script_only(stages),
            "timing": timing_table(*t) if t else None,
            "native_swaps": p_swaps,
        })

    for g in plan_groups:
        rid = g.get("recipe_id")
        keys = [k for k in chain_cover if k[0] == rid and k not in used_keys]
        if not keys:
            # 该分组的代表链一张都没导出（只导已核验时常见），跳过
            continue
        # 同 id 下样本最多的那条实际链，才是这个分组「代表链」
        key = max(keys, key=lambda k: chain_cover[k])
        used_keys.add(key)
        rec = RECIPE_BY_ID.get(rid)
        _add_pipeline(
            key,
            g.get("diagnosis_label", "") + (" · " + rec["desc"] if rec else " · AI 自主拼接"),
            g.get("consistency", 0), "group",
        )

    # 剩下没被任何分组代表的实际链，按样本出现顺序补在后面
    for it in items:
        if it.get("error"):
            continue
        # 用样本的链签名取键：items 里的 stages 可能已被全原生改写过，
        # 拿改写后的链去匹配改写前的签名会漏样本
        key = ckey_by_sample.get(it.get("sample_id"))
        if key is None or key in used_keys or key not in chain_cover:
            continue
        used_keys.add(key)
        rid = key[0]
        rec = RECIPE_BY_ID.get(rid)
        ds = it.get("diagnosis") or []
        label = DIAGNOSES.get(ds[0], {}).get("label", "") if ds else ""
        _add_pipeline(
            key,
            "样本专用" + ((" · " + label) if label else "")
            + (" · " + rec["desc"] if rec else " · AI 自主拼接"),
            1.0, "solo",
        )

    # 参数与方案基准不同的样本（精修产生）。逐张明细里要标出来，
    # 否则"按方案配完发现这张图对不上"就无从解释。
    for it in items:
        key = ckey_by_sample.get(it.get("sample_id"))
        base_sig = rep_sig.get(key) if key else None
        it["params_differ"] = bool(base_sig and psig_by_sample.get(it.get("sample_id")) != base_sig)

    manifest = {
        "batch_id": bid,
        "batch_name": batch.get("name", ""),
        "note": batch.get("note", ""),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_platform": "海康 VisionMaster",
        "geometry_invariant": True,
        "usage": {
            "debug_samples": "导入流程软件做验证，或作为流程调试的测试图",
            "compare": "原图|增强图并排，用于核对 VM 里跑出来的结果是否一致",
            "pipeline": "算子链参数，按节点顺序配进流程；script_nodes 里的必须写脚本模块",
        },
        "export_filter": {
            "only_accepted": bool(only_accepted),
            "total": len(all_samples),
            "exported": len(items),
            "skipped_unverified": skipped,
            "only_native": bool(only_native),
        },
        "pipeline": pipelines,
        "items": items,
    }

    manifest["export_filter"]["with_json"] = bool(with_json)

    # 上一次导出可能带了 JSON，这次不带就删掉，否则 zip 里会残留旧文件
    for stale in ("manifest.json", "pipeline.json", "edits.jsonl"):
        sp = os.path.join(base, stale)
        if os.path.exists(sp):
            os.remove(sp)

    if with_json:
        with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        with open(os.path.join(base, "pipeline.json"), "w", encoding="utf-8") as f:
            json.dump({
                "target_platform": "海康 VisionMaster",
                "only_native": bool(only_native),
                "exported_at": manifest["exported_at"],
                "batch": batch.get("name", ""),
                "pipelines": pipelines,
            }, f, ensure_ascii=False, indent=2)

        edits = store.read_edits(limit=100000, batch_id=bid)
        with open(os.path.join(base, "edits.jsonl"), "w", encoding="utf-8") as f:
            for e in edits:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(os.path.join(base, "流程配置.md"), "w", encoding="utf-8") as f:
        f.write(render_pipeline_md(batch, pipelines, items, manifest["export_filter"],
                                   native_notes))

    return manifest


def _cell(v) -> str:
    """表格单元格的转义：竖线会把列冲散，换行会把行冲散。"""
    return str(v if v is not None else "").replace("|", "\\|").replace("\n", " ").strip()


def render_pipeline_md(batch: dict, pipelines: list[dict], items: list[dict],
                       export_filter: dict | None = None,
                       native_notes: list[dict] | None = None) -> str:
    """给流程搭建的人看的一页纸。

    只讲三件事：这条链怎么配、配完多快、哪张图有问题。
    算法原理不写在这里——那是界面上「指标 / 伪影审计」的说明弹窗干的事。
    """
    L = []
    total_script = sum(len(p["script_nodes"]) for p in pipelines)
    n_acc = sum(1 for it in items if it.get("accepted"))
    L.append(f"# 流程配置 · {batch.get('name', '')}")
    L.append("")
    L.append(f"目标平台：**海康 VisionMaster**　"
             f"导出时间：{time.strftime('%Y-%m-%d %H:%M')}　"
             f"生成工具：Preproc Studio v{__version__}")
    L.append("")
    # 一段话交代完整：多少图、核没核验、几个方案、要不要写脚本。
    # 原来这四项散在三行里，还要翻到"先看这里"才知道有没有脚本节点。
    summary = [f"**{len(items)}** 张样本", f"**{n_acc}** 张已核验",
               f"**{len(pipelines)}** 个方案"]
    summary.append(f"**{total_script}** 个节点需脚本" if total_script
                   else "全部节点可原生配置")
    L.append("　·　".join(summary))
    if export_filter and export_filter.get("skipped_unverified"):
        L.append("")
        L.append(f"> 本次只导出了**已通过核验**的样本，"
                 f"另有 {export_filter['skipped_unverified']} 张未核验的图没有导出。")
    L.append("")

    if not pipelines:
        L.append("> 尚未聚合成方案。先在界面点「聚合成方案」，再导出。")
        return "\n".join(L)

    L.append("## 先看这里")
    L.append("")
    L.append("- **怎么配**：按下面每个方案的节点顺序，在 VM 里逐个选模块、填参数。"
             "标「需脚本」的节点用脚本模块，参考代码附在该方案末尾。")
    L.append("- **耗时**：以样本实测为准，其他分辨率按 ms/百万像素线性外推，是估算值，"
             "上产线前请在实际分辨率复测一次。")
    L.append("- **怎么核对**：拿 `compare/` 里的原图｜增强图对照，确认 VM 跑出来和这里一致。")
    L.append("")

    # 导出链上的审计是在整幅图上、按最终要配的链重算的，
    # 和界面上的结论可能不同（全原生替换是近似）。有拒绝就必须说，
    # 否则"界面通过、包里拒绝"这种事只有翻到最后一页才看得见。
    rejected = [it for it in items
                if (it.get("audit") or {}).get("verdict") == "reject"]
    if rejected:
        names = "、".join(it["filename"] for it in rejected[:5])
        more = f" 等 {len(rejected)} 张" if len(rejected) > 5 else ""
        L.append(f"> **{len(rejected)} 张样本在导出链上未通过伪影闸门**：{names}{more}。"
                 f"多为全原生替换后效果变化所致，这几张建议回界面换一条链再导。")
        L.append("")

    if native_notes:
        L.append("## 已转换为海康原生模块")
        L.append("")
        L.append("整条链没有脚本节点，全部可原生配置。但**替换的是近似，不是等价**——"
                 "配完之后请拿 `compare/` 里的对照图核一次，确认效果没跑偏。")
        L.append("")
        L.append("| 原算子 | 换成 | 参数 | 差异在哪 |")
        L.append("|---|---|---|---|")
        for c in native_notes:
            L.append(f"| `{c['from']}` | {c['to']} | {c['detail']} | {c['note']} |")
        L.append("")

    for pi, p in enumerate(pipelines, start=1):
        L.append(f"## 方案 {pi}：{p['desc']}")
        L.append("")
        # 一行交代完整：覆盖多少、一致率、几个节点、跑多快。
        # 原来这些分散在三处，配置的人要跳着读。
        head = [f"覆盖 **{p['covers']}** 张图"]
        head.append("人工指定，未参与分组投票" if p.get("kind") == "solo"
                    else f"组内一致率 {p['consistency'] * 100:.0f}%")
        head.append(f"{len(p['stages'])} 个节点")
        t = p.get("timing")
        if t:
            head.append(f"实测 **{t['base_ms']} ms** @ {t['base_w']}×{t['base_h']}")
        L.append("　·　".join(head))
        L.append("")

        if t and t.get("note"):
            L.append(f"> {t['note']}")
            L.append("")

        if not p["nodes"]:
            # 空链会渲染成一张只有表头的表，看着像生成失败，得说清楚是什么情况
            L.append("> 这条链没有节点（样本未分析，或链被清空），配不出流程——"
                     "回界面给它补一条链再导。")
            L.append("")
        else:
            L.append("| # | 算子 | VM 模块 | 参数 | 说明 |")
            L.append("|---|---|---|---|---|")
            for n in p["nodes"]:
                flag = {"native": "原生", "combo": "组合", "script": "**需脚本**"}[n["how"]]
                params = "、".join(f"`{k}={v}`" for k, v in n["params"].items()) or "—"
                desc = n["semantics"] or "—"
                if n["note"]:
                    desc = f"{desc}；{n['note']}"
                mod = n["vm_module"] or "无（需脚本实现）"
                L.append(f"| {n['node']} | `{n['op']}` {flag} | {_cell(mod)} | "
                         f"{_cell(params)} | {_cell(desc)} |")
            L.append("")

        if t:
            est = [f"{r['label']} ≈ {r['ms']} ms" for r in t["rows"] if not r["measured"]]
            if est:
                L.append(f"耗时外推（估算，非实测；{t['ms_per_mp']} ms/百万像素）："
                         + "　".join(est))
                L.append("")

        # 只有真需要脚本的节点才贴代码。原先是只要算子带参考实现就贴，
        # 原生节点也挂一段 python，一页纸里一半是配流程时根本用不上的东西。
        scripts = [n for n in p["nodes"] if n["how"] == "script" and n.get("script")]
        if scripts:
            L.append("### 需要脚本的节点")
            L.append("")
            for n in scripts:
                L.append(f"**{n['node']}. `{n['op']}`**")
                L.append("")
                L.append("```python")
                L.append(n["script"])
                L.append("```")
                L.append("")

    # 逐张明细与核验状态合并成一张表：原本两张表每张都是"一行一张图"，
    # 字段只差核验/改动/得分，配置的人要来回对着看才拼出完整状态。
    L.append("## 逐张明细")
    L.append("")
    L.append("| 文件 | 方案 | 分辨率 | 耗时 | 审计 | 核验 | 改动 | 得分 |")
    L.append("|---|---|---|---|---|---|---|---|")
    any_differ = False
    for it in items:
        if it.get("error"):
            L.append(f"| {it['filename']} | — | — | — | 失败：{_cell(it['error'])} "
                     f"| — | — | — |")
            continue
        v = {"pass": "通过", "warn": "警告", "reject": "拒绝"}.get(
            it["audit"]["verdict"], "—")
        plan_cell = it["recipe_id"] + ("·参数微调" if it.get("params_differ") else "")
        if not it.get("stages"):
            plan_cell += "·空链"
        any_differ = any_differ or bool(it.get("params_differ"))
        acc = "已核验" if it.get("accepted") else "未核验"
        edt = (f"改过 {it['edit_count']} 处" if it.get("edit_count")
               else ("改过" if it.get("edited") else "未改"))
        sc = it.get("score")
        sc_txt = f"{sc:+.3f}" if sc is not None else "—"
        L.append(f"| {it['filename']} | {_cell(plan_cell)} | "
                 f"{it['resolution'][0]}×{it['resolution'][1]} | {it['ms']} ms | {v} | "
                 f"{acc} | {edt} | {sc_txt} |")
    L.append("")
    if any_differ:
        L.append("> 标「参数微调」的样本，实际参数与该方案的基准参数略有差异（参数精修产生）。"
                 "这类样本按 `manifest.json` 里记录的实际参数配，或直接用它的增强图核对。")
        L.append("")

    L.append("## 包内有什么")
    L.append("")
    L.append("- `debug_samples/`　增强图，文件名与原图一致，可直接导入 VM 跑流程验证")
    L.append("- `compare/`　原图｜增强图并排，配完流程拿它核对 VM 输出是否一致")
    with_json = bool((export_filter or {}).get("with_json"))
    if with_json:
        L.append("- `pipeline.json` / `manifest.json` / `edits.jsonl`　机读配置单、"
                 "逐张索引与人工改动流水（给脚本用，手工配流程不需要）")
    else:
        L.append("")
        L.append("> 本次未包含 JSON。需要脚本化生成流程模板或做回归核对时，"
                 "重新导出时勾选「同时导出机器可读配置」。")
    L.append("")
    return "\n".join(L)


def render_markdown(batch: dict, items: list[dict], plan_groups: list[dict]) -> str:
    """保留旧接口，内部转调新的流程配置渲染。"""
    pipelines = []
    for g in plan_groups:
        rec = RECIPE_BY_ID.get(g.get("recipe_id"))
        pipelines.append({
            "desc": g.get("diagnosis_label", ""),
            "covers": covers,
            "consistency": g.get("consistency", 0),
            "stages": rec["stages"] if rec else [],
            "nodes": PF.chain_plan(rec["stages"] if rec else []),
            "script_nodes": PF.script_only(rec["stages"] if rec else []),
            "timing": None,
        })
    return render_pipeline_md(batch, pipelines, items)


def zip_export(out_dir: str, bid: str) -> str:
    base = os.path.join(out_dir, bid)
    zpath = os.path.join(out_dir, f"{bid}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(base):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, base))
    return zpath


def render_preview(img: np.ndarray, max_side: int = 520, quality: int = 82) -> str:
    """转 base64 jpeg 供前端内联显示。"""
    import base64

    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
