"""消费导出包：把 JSON 变成能直接用的东西。

两个命令：

  checklist   把 pipeline.json 摊成一张核对表（CSV），配流程时逐行打钩
  verify      把某套"外部实现"的输出和我们的产出逐像素比，给差异数字

用法：
  python scripts/use_export.py checklist data/exports/<批次ID>
  python scripts/use_export.py verify data/exports/<批次ID>
  python scripts/use_export.py verify data/exports/<批次ID> --vm-dir D:/vm_out

verify 不带 --vm-dir 时是自检：拿 stages 重跑一遍，和 debug_samples 里的图比，
差值应该为 0（证明导出的图确实是按配置单跑出来的）。
带 --vm-dir 时是跨实现核对：那个目录放 VM 跑出来的同名图，
数字越小说明 VM 配置得越接近分析时的效果。
"""
from __future__ import annotations

import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.ops import apply_recipe  # noqa: E402
from app.store import Store, imread_unicode  # noqa: E402


def _load(pkg_dir: str) -> tuple[dict, dict]:
    need = [n for n in ("pipeline.json", "manifest.json")
            if not os.path.exists(os.path.join(pkg_dir, n))]
    if need:
        # 默认导出不含 JSON，这个提示比 "No such file" 有用
        raise SystemExit(
            "这个包里没有机器可读配置（缺 %s）。\n"
            "在界面里重新导出，勾选「同时导出机器可读配置」，再跑本工具。"
            % "、".join(need))

    with open(os.path.join(pkg_dir, "pipeline.json"), encoding="utf-8") as f:
        pipe = json.load(f)
    with open(os.path.join(pkg_dir, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    return pipe, man


def cmd_checklist(pkg_dir: str) -> None:
    pipe, man = _load(pkg_dir)
    out = os.path.join(pkg_dir, "配置核对表.csv")

    rows = []
    for i, p in enumerate(pipe.get("pipelines") or [], start=1):
        for n in p.get("nodes") or []:
            params = "；".join(f"{k}={v}" for k, v in (n.get("params") or {}).items())
            rows.append({
                "方案": f"{i}. {p.get('desc', '')}",
                "覆盖张数": p.get("covers", 0),
                "节点序号": n.get("node"),
                "算子": n.get("op"),
                "VM模块": n.get("vm_module") or "",
                "配置方式": {"native": "原生", "combo": "原生(组合)", "script": "脚本"}.get(
                    n.get("how"), n.get("how") or ""),
                "参数": params,
                "算法语义": n.get("semantics") or "",
                "备注": n.get("note") or "",
                "配好了": "",          # 留给配流程的人打钩
                "实测耗时ms": "",
            })

    if not rows:
        print("这个包里没有方案节点，先聚合成方案再导出。")
        return

    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    script_n = sum(1 for r in rows if r["配置方式"] == "脚本")
    print(f"已生成 {out}")
    print(f"  {len(pipe['pipelines'])} 条方案，{len(rows)} 个节点，其中脚本节点 {script_n} 个")
    print(f"  覆盖样本 {sum(p.get('covers', 0) for p in pipe['pipelines'])} 张")


def _diff(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if x.ndim == 3 else x
    ga, gb = gray(a), gray(b)
    d = np.abs(ga.astype(np.int16) - gb.astype(np.int16))
    mse = float((d.astype(np.float32) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    return {
        "mean_abs": float(d.mean()),
        "p99": float(np.percentile(d, 99)),
        "max": int(d.max()),
        "psnr": psnr,
        "same_ratio": float((d <= 2).mean()),
    }


def cmd_verify(pkg_dir: str, vm_dir: str | None = None) -> None:
    _pipe, man = _load(pkg_dir)
    items = [it for it in man.get("items") or [] if not it.get("error")]
    if not items:
        print("manifest 里没有可用样本。")
        return

    store = Store(os.path.join(ROOT, "data"))
    mode = "跨实现核对（外部输出 vs 我们的产出）" if vm_dir else "自检（重跑 stages vs 导出图）"
    print(f"模式：{mode}\n")

    ok = 0
    bad = []
    for it in items:
        name = it.get("enhanced_file") or it.get("filename")
        ours_p = os.path.join(pkg_dir, "debug_samples", name)
        ours = imread_unicode(ours_p)
        if ours is None:
            print(f"  跳过 {name}：导出图读不到")
            continue

        if vm_dir:
            other = imread_unicode(os.path.join(vm_dir, name))
            if other is None:
                bad.append((name, "外部目录里没有同名图"))
                continue
        else:
            # 原图从数据目录里按 sample_id 取，和导出时用的是同一份
            sample = store.get_sample(man.get("batch_id"), it.get("sample_id"))
            src = store.load_image(sample) if sample else None
            if src is None:
                bad.append((name, "找不到原图，无法重跑"))
                continue
            try:
                other, _ = apply_recipe(src, {"stages": it.get("stages") or []})
            except Exception as e:
                bad.append((name, f"重跑失败 {type(e).__name__}: {e}"))
                continue

        d = _diff(ours, other)
        verdict = "一致" if d["mean_abs"] <= 1.0 else ("接近" if d["mean_abs"] <= 4.0 else "偏离")
        if verdict == "一致":
            ok += 1
        else:
            bad.append((name, f"{verdict} mean={d['mean_abs']:.2f}"))
        print(f"  {name:<26} 均值差 {d['mean_abs']:6.2f}  P99 {d['p99']:5.1f}  "
              f"PSNR {d['psnr']:5.1f}dB  同像素 {d['same_ratio'] * 100:5.1f}%  {verdict}")

    print()
    print(f"共 {len(items)} 张，完全一致 {ok} 张，其余 {len(bad)} 张")
    if bad:
        print("需要看的：")
        for name, why in bad[:12]:
            print(f"  - {name}: {why}")
    if vm_dir:
        print("\n判读：均值差 <1 基本等价；1~4 属实现差异（截断/取整/参数定义不同）；")
        print("      >4 说明模块选错或参数没对上，要回到流程里改。")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return
    cmd, pkg = sys.argv[1], sys.argv[2]
    vm_dir = None
    if "--vm-dir" in sys.argv:
        i = sys.argv.index("--vm-dir")
        if i + 1 < len(sys.argv):
            vm_dir = sys.argv[i + 1]

    if not os.path.isdir(pkg):
        print(f"目录不存在：{pkg}")
        return

    if cmd == "checklist":
        cmd_checklist(pkg)
    elif cmd == "verify":
        cmd_verify(pkg, vm_dir)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
