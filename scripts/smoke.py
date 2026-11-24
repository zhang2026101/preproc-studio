"""端到端冒烟测试：不启服务，直接跑通核心链路

    python scripts/smoke.py [图片目录]

验证顺序与界面一致：建批次 -> 导入 -> 诊断 -> 三次迭代 -> 逐节预览 -> 聚合 -> 导出
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import agent as AG          # noqa: E402
from app import export as EX         # noqa: E402
from app import metrics as MT        # noqa: E402
from app import vlm as VLM           # noqa: E402
from app.ops import apply_with_stages  # noqa: E402
from app.store import Store, imread_unicode  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "samples"
N = 3

store = Store(os.path.join(ROOT, "data"))

print("=" * 62)
print("1. 建批次 + 导入")
b = store.create_batch("冒烟批次", "smoke test")
files = [n for n in sorted(os.listdir(SRC)) if n.lower().endswith((".png", ".jpg", ".bmp", ".tif"))][:N]
for n in files:
    img = imread_unicode(os.path.join(SRC, n))
    if img is None:
        print("  跳过（读不了）:", n)
        continue
    s = store.add_sample(b["id"], n, img)
    print(f"  + {n}  {img.shape[1]}x{img.shape[0]}")

b = store.get_batch(b["id"])
if not b["samples"]:
    print("没有可用图片，退出")
    sys.exit(1)

print("\n2. 诊断 + 三次迭代（规则兜底，不联网）")
cfg = VLM.load_config(store.config_path)
for s in b["samples"]:
    img = store.load_image(s)
    work = AG._downscale(img, 1600)
    base_m = MT.measure(work)
    diag = VLM.diagnose(work, cfg, base_m=base_m)
    t0 = time.time()
    res = AG.run_agent(work, diag, max_rounds=3)
    dt = int((time.time() - t0) * 1000)
    rec = res["recommendation"]
    print(f"\n  {s['filename']}")
    print(f"    诊断 {diag['diagnoses']} ({diag['provider']})")
    print(f"    轮次 {res['candidates_tried']} 用时 {dt}ms 停止原因「{res['stop_reason']}」")
    for r in res["rounds"]:
        if not r.get("ok"):
            print(f"      R{r['round']} {r.get('recipe_id')} FAILED {r.get('error')}")
            continue
        a = r["audit"]
        print(f"      R{r['round']} {r['recipe_id']:<18} score={r['score']:+.3f} "
              f"verdict={a['verdict']:<6} risk={a['risk']:.2f} "
              f"clip={a['clip_abs']*100:.1f}% noise_abs={a["noise_abs"]*255:.1f} ratio={a["noise_ratio"]:.1f}x")

    if rec:
        store.update_sample(b["id"], s["id"], {
            "diagnosis": diag,
            "rounds": [{k: v for k, v in r.items() if k != "steps"} for r in res["rounds"]],
            "stages": rec["stages"],
            "selected_recipe": rec["recipe_id"],
        })
    else:
        print("    !! 无可用推荐")

print("\n3. 逐节预览")
b = store.get_batch(b["id"])
s0 = b["samples"][0]
img = store.load_image(s0)
work = AG._downscale(img, 900)
base_m = MT.measure(work)
out, steps = apply_with_stages(work, {"stages": s0["stages"]})
print(f"  {s0['filename']} 共 {len(steps)} 态")
for i, op, im in steps:
    m = MT.measure(im)
    sc, _ = MT.score(m, base_m)
    print(f"    step {i} {op:<14} {im.shape} score={sc:+.3f}")

print("\n4. 聚合")
items = [{
    "sample_id": s["id"], "filename": s["filename"],
    "diagnosis": (s.get("diagnosis") or {}).get("diagnoses") or ["normal"],
    "recipe_id": s.get("selected_recipe"),
} for s in b["samples"]]
agg = AG.aggregate_batch(items)
print(f"  {agg['groups_count']} 组, {agg['total']} 张, 需人工 {agg['needs_manual_count']}")
for g in agg["groups"]:
    print(f"    {g['diagnosis_label']:<14} -> {g['recipe_id']:<18} "
          f"{g['member_count']} 张 一致率 {g['consistency']*100:.0f}%")

print("\n5. 导出")
b["aggregate"] = agg
b["plan"] = agg["groups"]
store.save_batch(b)
manifest = EX.build_export(store, b, store.exports_dir)
z = EX.zip_export(store.exports_dir, b["id"])
print(f"  {len(manifest['items'])} 张增强图 -> {os.path.basename(z)} "
      f"({os.path.getsize(z)/1024:.0f} KB)")
md = os.path.join(store.exports_dir, b["id"], "方案说明.md")
print(f"  说明文档 {os.path.getsize(md)} 字节")

print("\n6. 配方准入表（全部配方 × 前 4 张）")
imgs = [store.load_image(s) for s in b["samples"][:4]]
pf = AG.preflight(imgs, max_side=1000)
print(f"  {pf['admitted_count']}/{pf['total']} 条准入")
for r in pf["rows"][:6]:
    print(f"    {r['recipe_id']:<18} 拒绝率 {r['reject_rate']*100:>4.0f}%  "
          f"噪声 {r['mean_noise_ratio']:.2f}x  均分 {r['mean_score']:+.3f}  "
          f"{'准入' if r['admitted'] else '禁用'}")

print("\n" + "=" * 62)
print(f"全部通过。批次 id = {b['id']}")
