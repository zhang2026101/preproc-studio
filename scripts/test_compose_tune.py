"""验证「AI 自主拼链」+「参数精修」两条新链路

    python scripts/test_compose_tune.py

用本地假端点冒充模型，返回一条**故意有问题**的链：
  - 混入不存在的算子 pseudo_hdr
  - clahe.clip=99 远超上限
  - 链首不是通道算子
看校验层能不能把这些收拾干净，再看精修层能不能真的提分。
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import agent as AG          # noqa: E402
from app import metrics as MT       # noqa: E402
from app import vlm as VLM          # noqa: E402
from app.recipes import RECIPE_BY_ID  # noqa: E402
from app.store import imread_unicode  # noqa: E402

PORT = 8901
ok = True
SEEN = {"compose": 0, "diagnose": 0}

MOCK_CHAIN = {
    "stages": [
        {"op": "retinex", "p": {"sigmas": [15, 80, 250]}},
        {"op": "pseudo_hdr", "p": {}},
        {"op": "clahe", "p": {"clip": 99, "grid": 8}},
        {"op": "normalize", "p": {"low": 1, "high": 99}},
    ],
    "rationale": "先用 Retinex 提暗部，再用 CLAHE 拉局部对比，最后拉伸定标",
    "confidence": 0.78,
}


def check(name, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        ok = False


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        msgs = body.get("messages") or []
        text = json.dumps(msgs, ensure_ascii=False)

        if "工业视觉预处理工程师" in text:
            SEEN["compose"] += 1
            content = json.dumps(MOCK_CHAIN, ensure_ascii=False)
        else:
            SEEN["diagnose"] += 1
            content = json.dumps({
                "diagnoses": ["underexposed", "low_contrast"],
                "observation": "mock: 整体偏暗，细节压在暗端",
                "confidence": 0.9,
            }, ensure_ascii=False)

        data = json.dumps({"choices": [{"message": {"content": content}}]},
                          ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


srv = HTTPServer(("127.0.0.1", PORT), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"mock 已起于 :{PORT}\n")

img = imread_unicode(os.path.join("samples", "sample.png"))
if img is None:
    print("测试图读不到，退出")
    sys.exit(1)

cfg = {
    "provider": "cloud", "base_url": f"http://127.0.0.1:{PORT}/v1",
    "model": "deepseek-flash", "api_key": "sk-test", "timeout": 20,
    "detail": "low", "image_style": "data_url",
}

print("1. 诊断（mock）")
VLM._CACHE.clear()
diag = VLM.diagnose(img, cfg, force_provider="cloud")
check("诊断走模型", diag["provider"] == "cloud", diag["provider"])
check("标签合法", diag["diagnoses"] == ["underexposed", "low_contrast"], diag["diagnoses"])

print("\n2. 完整三阶段流程")
res = AG.run_agent(img, diag, cfg=cfg, max_side=900)
phases = [r.get("phase") for r in res["rounds"] if r.get("ok")]
check("有查表阶段", phases.count("preset") >= 2, str(phases))
check("有 AI 自拟阶段", "compose" in phases, str(phases))
check("调用了拼链接口", SEEN["compose"] == 1, f"compose 调用 {SEEN['compose']} 次")

comp = res.get("compose") or {}
check("拼链成功", comp.get("ok") is True, str(comp.get("error"))[:60])

crow = next((r for r in res["rounds"] if r.get("phase") == "compose"), None)
if crow:
    ops = [s["op"] for s in crow["stages"]]
    check("非法算子被丢弃", "pseudo_hdr" not in ops, str(ops))
    check("链首自动补 gray", ops[0] == "gray", str(ops))
    clip = None
    for s in crow["stages"]:
        if s["op"] == "clahe":
            clip = s["p"].get("clip")
    check("超界参数被钳制", clip is not None and clip <= 8.0, f"clip={clip}")
    check("有 rationale", bool(crow.get("rationale")), crow.get("rationale", "")[:40])
    print(f"       链: {'→'.join(ops)}  得分 {crow['score']:+.3f}")

print("\n3. 参数精修")
tun = res.get("tuning") or {}
runs = tun.get("runs") or []
check("精修已执行", tun.get("evaluations", 0) > 0, f"评估 {tun.get('evaluations')} 次")
check("有尝试记录", runs and len(runs[0].get("trials") or []) > 0)
for run in runs:
    acc = [t for t in run["trials"] if t["accepted"]]
    print(f"       链路 {run['target']:<16} {run['base_score']:+.3f} → {run['score']:+.3f}"
          f"（{run['gain']:+.3f}，采纳 {len(acc)}/{len(run['trials'])}）")
    for t in acc[:3]:
        print(f"         ✓ {t['op']}.{t['param']} {t['from']} → {t['to']}  ({t['score']:+.3f} {t['verdict']})")

print("\n4. 最终结论")
rec = res.get("recommendation") or {}
check("有推荐", bool(rec), str(rec.get("recipe_id")))
check("推荐可追溯到来源", rec.get("phase") in ("preset", "compose", "tune"), rec.get("phase"))
print(f"       推荐 {rec.get('recipe_id')} ({rec.get('phase')}) 得分 {rec.get('score'):+.3f}")
print(f"       停止原因：{res.get('stop_reason')}")

print("\n5. 关掉拼链与精修，应只剩查表")
res2 = AG.run_agent(img, diag, cfg=cfg, max_side=900,
                    enable_compose=False, enable_tune=False)
p2 = [r.get("phase") for r in res2["rounds"] if r.get("ok")]
check("只剩查表阶段", set(p2) == {"preset"}, str(p2))
check("拼链条目为空", res2.get("compose") is None)
check("精修条目为空", res2.get("tuning") is None)

srv.shutdown()
print("\n" + "=" * 56)
print("全部通过" if ok else "有失败项")
sys.exit(0 if ok else 1)
