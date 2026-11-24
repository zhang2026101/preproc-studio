"""HTTP 接口验收：模拟界面会发的每一个请求

    python scripts/api_test.py [base_url] [图片目录]

依赖服务已启动（python run.py）。
"""

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8770"
SRC = sys.argv[2] if len(sys.argv) > 2 else "samples"

ok_all = True


def step(name, r, check=None):
    global ok_all
    good = r.status_code < 400
    if good and check:
        try:
            good = bool(check(r.json()))
        except Exception:
            good = False
    mark = "PASS" if good else "FAIL"
    if not good:
        ok_all = False
    print(f"  [{mark}] {name}  {r.status_code}")
    if not good:
        print("        ", r.text[:220])
    return r.json() if good else {}


class Client:
    """标准库实现的极简客户端，避免为跑测试再装一个包。"""

    def __init__(self, base, timeout=180):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def _req(self, method, path, body=None, files=None):
        url = self.base + path
        headers = {}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                return Resp(r.status, raw)
        except urllib.error.HTTPError as e:
            return Resp(e.code, e.read())
        except Exception as e:
            return Resp(0, json.dumps({"detail": f"{type(e).__name__}: {e}"}).encode())

    def get(self, p, **kw): return self._req("GET", p)
    def post(self, p, json=None, **kw): return self._req("POST", p, body=json)
    def delete(self, p, **kw): return self._req("DELETE", p)

    def __enter__(self): return self
    def __exit__(self, *a): return False


class Resp:
    def __init__(self, status, raw):
        self.status_code = status
        self._raw = raw

    @property
    def content(self):
        return self._raw

    @property
    def text(self):
        return self._raw.decode("utf-8", "ignore")

    def json(self):
        return json.loads(self._raw.decode("utf-8"))


with Client(BASE) as c:
    print("1. 基础")
    step("health", c.get("/api/health"), lambda j: j["ops"] == 16)
    sc = step("schema", c.get("/api/schema"), lambda j: len(j["ops"]) == 16 and len(j["recipes"]) == 12)
    print(f"        算子 {len(sc.get('ops', []))} 个，分组 {sc.get('groups')}")

    print("\n2. 批次")
    b = step("create batch", c.post("/api/batches", json={"name": "接口验收", "note": "api test"}))
    bid = b["batch"]["id"]
    imp = step("import dir", c.post(f"/api/batches/{bid}/import_dir",
                                   json={"path": SRC, "limit": 2}),
               lambda j: j["count"] > 0)
    print(f"        导入 {imp.get('count')} 张")
    detail = c.get(f"/api/batches/{bid}").json()
    sid = detail["batch"]["samples"][0]["id"]
    fn = detail["batch"]["samples"][0]["filename"]
    print(f"        首个样本 {fn}")

    print("\n3. 诊断 + 三次迭代")
    a = step("analyze", c.post(f"/api/batches/{bid}/samples/{sid}/analyze", json={}),
             lambda j: j["diagnosis"] and j["rounds"] and j["panels"])
    print(f"        诊断 {a['diagnosis']['diagnoses']} ({a['diagnosis']['provider']})")
    print(f"        试跑 {a['candidates_tried']} 条，{a['elapsed_ms']} ms，停止：{a['stop_reason']}")
    for r in a["rounds"]:
        if r.get("ok"):
            print(f"          R{r['round']} {r['recipe_id']:<18} {r['score']:+.3f} "
                  f"{r['audit']['verdict']:<6} steps={len(r['steps'])}")
    print(f"        推荐 {a['recommendation']['recipe_id'] if a['recommendation'] else '无'}")
    print(f"        面板 {len(a['panels'])} 个")

    print("\n4. 逐节预览 + 人工改参数")
    stages = a["stages"]
    pv = step("preview", c.post(f"/api/batches/{bid}/samples/{sid}/preview",
                                json={"stages": stages}), lambda j: len(j["steps"]) == len(stages) + 1)
    print(f"        {len(pv['steps'])} 个中间态，末态得分 {pv['final']['score']:+.3f}")
    for s in pv["steps"]:
        print(f"          step {s['step']} {s['op']:<14} score={s['score']:+.3f}")

    # 改一个参数（retinex 的 sigmas 或第一个 float 参数）
    edited = json.loads(json.dumps(stages))
    touched = None
    for st in edited:
        if st.get("p"):
            for k, v in st["p"].items():
                if isinstance(v, (int, float)):
                    st["p"][k] = round(float(v) * 1.2, 3)
                    touched = (st["op"], k, st["p"][k])
                    break
            if touched:
                break
    pv2 = step("preview after edit", c.post(f"/api/batches/{bid}/samples/{sid}/preview",
                                            json={"stages": edited}))
    print(f"        改动 {touched} -> 得分 {pv2['final']['score']:+.3f} "
          f"(原 {pv['final']['score']:+.3f})")

    print("\n5. 保存修改 + 核验")
    step("save stages", c.post(f"/api/batches/{bid}/samples/{sid}/stages",
                               json={"stages": edited, "kind": "param", "op": touched[0] if touched else None}))
    step("accept", c.post(f"/api/batches/{bid}/samples/{sid}/accept", json={"stages": edited}))
    st = c.get(f"/api/batches/{bid}/stats").json()
    print(f"        已接受 {st['accepted']}/{st['total']}，改动 {st['edited']}，"
          f"流水 {st['edits']['total']} 条")

    print("\n6. 整批 + 聚合 + 导出")
    ra = step("run_all", c.post(f"/api/batches/{bid}/run_all", json={}), lambda j: j["count"] > 0)
    print(f"        整批 {ra.get('count')} 张")
    ag = step("aggregate", c.post(f"/api/batches/{bid}/aggregate", json={}),
              lambda j: j["aggregate"]["groups"])
    for g in ag["aggregate"]["groups"]:
        print(f"        {g['diagnosis_label']:<14} -> {g['recipe_id']:<18} "
              f"{g['member_count']} 张 {g['consistency'] * 100:.0f}%")
    ex = step("export", c.post(f"/api/batches/{bid}/export", json={}), lambda j: j["items"] > 0)
    print(f"        导出 {ex.get('items')} 张，{ex.get('elapsed_ms')} ms")
    dl = c.get(f"/api/batches/{bid}/export/download")
    zip_ok = dl.status_code == 200 and dl.content[:2] == b"PK"
    print(f"  [{'PASS' if zip_ok else 'FAIL'}] download zip  {dl.status_code}")
    if not zip_ok:
        ok_all = False
    print(f"        zip 大小 {len(dl.content) / 1024:.0f} KB")

    print("\n7. 准入表")
    pf = step("preflight", c.post("/api/preflight", json={"batch_id": bid, "limit": 4}),
              lambda j: j["rows"])
    print(f"        {pf.get('admitted_count')}/{pf.get('total')} 条准入")
    for r in pf["rows"][:4]:
        print(f"          {r['recipe_id']:<18} 拒绝 {r['reject_rate'] * 100:>3.0f}%  "
              f"均分 {r['mean_score']:+.3f}  {'准入' if r['admitted'] else '禁用'}")

    print("\n8. 清理")
    step("delete batch", c.delete(f"/api/batches/{bid}"))

print("\n" + ("=" * 56))
print("全部接口通过" if ok_all else "有失败项，见上")
sys.exit(0 if ok_all else 1)
