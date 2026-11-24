"""用本地 mock 服务验证视觉接口的请求结构

没有 API key 也能验：起一个假的 OpenAI 兼容端点，把 vlm.py 的请求拦下来，
逐字段检查 model / image_url 格式 / detail / response_format / 回退重试是否都对。

    python scripts/mock_test_vision.py
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import vlm as VLM  # noqa: E402

PORT = 8899
LAST = {"body": None, "count": 0}
MODE = {"reject_first": False}

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        ok = False


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        LAST["body"] = body
        LAST["count"] += 1

        if MODE["reject_first"] and LAST["count"] == 1:
            data = json.dumps({"error": {"message": "response_format is not supported"}}).encode()
            self.send_response(400)
        else:
            resp = {"choices": [{"message": {"content": json.dumps({
                "diagnoses": ["underexposed", "low_contrast", "不存在的标签"],
                "observation": "mock: 图像整体偏暗，有效码值约在 12-58 区间",
                "confidence": 0.86,
            }, ensure_ascii=False)}}]}
            data = json.dumps(resp, ensure_ascii=False).encode()
            self.send_response(200)

        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


srv = HTTPServer(("127.0.0.1", PORT), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"mock 服务已起于 :{PORT}")

img = np.full((480, 640, 3), 90, np.uint8)

print("\n1. DeepSeek 预设（data URL + detail=low）")
VLM._CACHE.clear()
MODE["reject_first"] = False
cfg = {
    "provider": "cloud", "preset": "deepseek",
    "base_url": f"http://127.0.0.1:{PORT}/v1", "model": "deepseek-flash",
    "api_key": "sk-test", "timeout": 10, "detail": "low",
    "image_style": "data_url", "image_max_side": 1024,
}
r = VLM.diagnose(img, cfg, force_provider="cloud")
b = LAST["body"] or {}

check("请求到达 mock", LAST["count"] >= 1, f"次数 {LAST['count']}")
check("model 正确", b.get("model") == "deepseek-flash", b.get("model"))
check("带 response_format", (b.get("response_format") or {}).get("type") == "json_object")

content = (b.get("messages") or [{}, {}])[1].get("content") or []
img_block = next((c for c in content if c.get("type") == "image_url"), None)
check("有 image_url 块", img_block is not None)
if img_block:
    url = img_block["image_url"].get("url", "")
    check("图片是 data URL", url.startswith("data:image/jpeg;base64,"), url[:34] + "…")
    check("detail=low", img_block["image_url"].get("detail") == "low",
          img_block["image_url"].get("detail"))
    check("图片被压到 ≤1024", len(url) < 400000, f"base64 长度 {len(url)}")

check("诊断结果已解析", r["diagnoses"] == ["underexposed", "low_contrast"], r["diagnoses"])
check("非法标签被丢弃", "不存在的标签" not in r["diagnoses"])
check("observation 保留", "12-58" in r.get("observation", ""), r.get("observation"))

print("\n2. 智谱预设（裸 base64，不带 detail）")
VLM._CACHE.clear()
cfg2 = dict(cfg, preset="zhipu", model="glm-4v-flash",
            image_style="raw_base64", detail="")
img2 = np.full((300, 400, 3), 120, np.uint8)
VLM.diagnose(img2, cfg2, force_provider="cloud")
b2 = LAST["body"] or {}
c2 = (b2.get("messages") or [{}, {}])[1].get("content") or []
ib2 = next((c for c in c2 if c.get("type") == "image_url"), None)
if ib2:
    u2 = ib2["image_url"]["url"]
    check("裸 base64 无前缀", not u2.startswith("data:"), u2[:24] + "…")
    check("不带 detail 字段", "detail" not in ib2["image_url"])

print("\n3. 端点不支持 response_format 时自动回退")
VLM._CACHE.clear()
MODE["reject_first"] = True
LAST["count"] = 0
img3 = np.full((200, 200, 3), 60, np.uint8)
r3 = VLM.diagnose(img3, cfg, force_provider="cloud")
check("重试后成功", LAST["count"] == 2, f"请求次数 {LAST['count']}")
check("回退后仍拿到结果", r3["diagnoses"] == ["underexposed", "low_contrast"], r3["diagnoses"])

print("\n4. 接口不可达时降级到规则")
cfg_bad = dict(cfg, base_url="http://127.0.0.1:9/v1", timeout=2)
r4 = VLM.diagnose(np.full((256, 256, 3), 30, np.uint8), cfg_bad, force_provider="cloud")
check("自动降级 rules", r4["provider"] == "rules", r4["provider"])
check("记录降级原因", bool(r4.get("fallback_reason")), (r4.get("fallback_reason") or "")[:70])
check("仍有可用诊断", bool(r4["diagnoses"]), str(r4["diagnoses"]))

print("\n5. 401 错误提示可读")
VLM._CACHE.clear()
MODE["reject_first"] = False
cfg401 = dict(cfg, base_url="http://127.0.0.1:9/v1", timeout=2)
import urllib.error
try:
    raise RuntimeError(VLM._friendly_http_error(401, '{"error":"invalid key"}'))
except RuntimeError as e:
    check("401 提示含中文说明", "Key 无效" in str(e), str(e)[:60])

print("\n6. /api/test_vision 用表单值而不是已存配置")
from app.main import test_vision  # noqa: E402

MODE["reject_first"] = False
VLM._CACHE.clear()
r6 = test_vision({
    "provider": "cloud",
    "base_url": f"http://127.0.0.1:{PORT}/v1",
    "model": "deepseek-flash",
    "detail": "low",
    "image_style": "data_url",
    "api_key": "sk-form",
    "timeout": 10,
})
check("未保存也能测通", r6["ok"] is True, str(r6.get("reason"))[:60])
check("回显模型名", r6.get("model") == "deepseek-flash", r6.get("model"))

r7 = test_vision({"provider": "rules"})
check("rules 时给明确引导", r7["ok"] is False and r7.get("hint") == "provider=rules",
      (r7.get("reason") or "")[:40])

srv.shutdown()
print("\n" + "=" * 54)
print("全部通过" if ok else "有失败项")
sys.exit(0 if ok else 1)
