"""准备一个演示批次：导入 -> 逐张分析 -> 模拟两人核验 -> 聚合

目的是让界面打开时就是满的，而不是空白。
不联网、不调 API，纯本地算子。

    python scripts/prepare_demo.py
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import agent as AG      # noqa: E402
from app import metrics as MT    # noqa: E402
from app import vlm as VLM       # noqa: E402
from app.store import Store, imread_unicode  # noqa: E402

SRC_DIRS = [
    "samples",
    "samples_b",
]
EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

store = Store(os.path.join(ROOT, "data"))
cfg = VLM.load_config(store.config_path)

# 已有同名批次就先删掉，避免重复堆叠
for b in store.list_batches():
    if b["name"].startswith("铜线 · 低照度"):
        store.delete_batch(b["id"])
        print("清理旧批次", b["id"])

batch = store.create_batch("铜线 · 低照度批次", "素材自带样本，用于演示标注辅助视图的完整流程")
bid = batch["id"]

files = []
for d in SRC_DIRS:
    if os.path.isdir(d):
        files += [os.path.join(d, n) for n in sorted(os.listdir(d)) if n.lower().endswith(EXT)]
print(f"发现 {len(files)} 张图")

t0 = time.time()
for p in files:
    img = imread_unicode(p)
    if img is None:
        continue
    store.add_sample(bid, os.path.basename(p), img)
print(f"导入完成 {len(files)} 张，{time.time() - t0:.1f}s")

batch = store.get_batch(bid)
print("逐张分析…")
for i, s in enumerate(batch["samples"], start=1):
    img = store.load_image(s)
    work = AG._downscale(img, 1600)
    base_m = MT.measure(work)
    diag = VLM.diagnose(work, cfg, base_m=base_m)
    res = AG.run_agent(work, diag, max_rounds=3)
    rec = res["recommendation"]
    patch = {
        "diagnosis": diag,
        "rounds": [{k: v for k, v in r.items() if k != "steps"} for r in res["rounds"]],
        "review_required": bool(rec and rec.get("review_required")),
    }
    if rec:
        patch["stages"] = rec["stages"]
        patch["selected_recipe"] = rec["recipe_id"]
    store.update_sample(bid, s["id"], patch)
    tag = rec["recipe_id"] if rec else "无"
    print(f"  {i:>2}/{len(batch['samples'])} {s['filename'][:26]:<28} "
          f"{'+' if (rec and rec['score'] > 0) else ''}{rec['score']:.3f}  {tag}")

# 模拟"两个样本已经被人核验过"，其中一个还改了参数
batch = store.get_batch(bid)
samples = batch["samples"]
if len(samples) >= 2:
    s1 = samples[0]
    store.update_sample(bid, s1["id"], {"accepted": True, "accepted_at": time.time()})
    store.log_edit({
        "batch_id": bid, "sample_id": s1["id"], "filename": s1["filename"],
        "kind": "accept", "recipe_id": s1.get("selected_recipe"), "note": "",
    })

    s2 = samples[1]
    stages = s2.get("stages") or []
    if stages:
        for st in stages:
            for k, v in (st.get("p") or {}).items():
                if isinstance(v, (int, float)) and st["op"] == "normalize":
                    st["p"][k] = round(float(v) + 0.3, 2)
                    break
        store.update_sample(bid, s2["id"], {"stages": stages, "edited": True, "edit_count": 1})
        store.log_edit({
            "batch_id": bid, "sample_id": s2["id"], "filename": s2["filename"],
            "kind": "param", "op": "normalize",
            "before": s2.get("stages"), "after": stages,
            "note": "人工放宽百分位，让暗部更透",
        })
        print("已模拟 1 张核验通过、1 张改参数")

# 聚合
batch = store.get_batch(bid)
items = [{
    "sample_id": s["id"], "filename": s["filename"],
    "diagnosis": (s.get("diagnosis") or {}).get("diagnoses") or ["normal"],
    "recipe_id": s.get("selected_recipe") or "r01_pct_stretch",
} for s in batch["samples"]]
agg = AG.aggregate_batch(items)
batch["aggregate"] = agg
batch["plan"] = agg["groups"]
store.save_batch(batch)

print(f"\n聚合：{agg['groups_count']} 组 / {agg['total']} 张，需人工 {agg['needs_manual_count']}")
for g in agg["groups"]:
    print(f"  {g['diagnosis_label']:<14} -> {g['recipe_id']:<18} "
          f"{g['member_count']} 张 一致率 {g['consistency'] * 100:.0f}%")
print(f"\n批次 id = {bid}   总耗时 {time.time() - t0:.1f}s")
