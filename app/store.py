"""轻量持久化

用 JSON 文件而不是数据库，理由很实在：
1. 一天内要交，SQLite 的迁移/并发问题不值当
2. 答辩时"每一步都留痕"最直观的证据就是能打开的 JSON 和 jsonl
3. 导出时直接把 meta.json 塞进导出包，评审能自己翻

目录约定
--------
data/uploads/<batch_id>/<sample_id>.png     原图
data/batches/<batch_id>.json                批次元信息 + 所有样本状态
data/edits.jsonl                            人工改动流水（append-only）
data/config.json                            VLM 配置
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid

import cv2
import numpy as np

LOCK = threading.RLock()


def now() -> float:
    return time.time()


def imread_unicode(path: str):
    """Windows 下 cv2.imread 读不了中文路径，统一走 imdecode。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, img: np.ndarray, params=None) -> bool:
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(path)
    return True


class Store:
    def __init__(self, root: str):
        self.root = root
        self.batches_dir = os.path.join(root, "batches")
        self.uploads_dir = os.path.join(root, "uploads")
        self.exports_dir = os.path.join(root, "exports")
        self.edits_path = os.path.join(root, "edits.jsonl")
        self.config_path = os.path.join(root, "config.json")
        for d in (self.batches_dir, self.uploads_dir, self.exports_dir):
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------ 批次

    def _bpath(self, bid: str) -> str:
        return os.path.join(self.batches_dir, f"{bid}.json")

    def create_batch(self, name: str, note: str = "") -> dict:
        bid = uuid.uuid4().hex[:12]
        b = {
            "id": bid,
            "name": name or "未命名批次",
            "note": note,
            "created": now(),
            "updated": now(),
            "samples": [],
            "plan": None,
            "aggregate": None,
        }
        with LOCK:
            self._write_json(self._bpath(bid), b)
            os.makedirs(os.path.join(self.uploads_dir, bid), exist_ok=True)
        return b

    def list_batches(self) -> list[dict]:
        out = []
        for fn in sorted(os.listdir(self.batches_dir), reverse=True):
            if not fn.endswith(".json"):
                continue
            b = self._read_json(os.path.join(self.batches_dir, fn))
            if not b:
                continue
            out.append({
                "id": b["id"],
                "name": b["name"],
                "note": b.get("note", ""),
                "created": b.get("created"),
                "updated": b.get("updated"),
                "count": len(b.get("samples", [])),
                "accepted": sum(1 for s in b.get("samples", []) if s.get("accepted")),
                "with_plan": bool(b.get("plan")),
            })
        out.sort(key=lambda x: x.get("updated") or 0, reverse=True)
        return out

    def get_batch(self, bid: str) -> dict | None:
        return self._read_json(self._bpath(bid))

    def save_batch(self, b: dict) -> None:
        b["updated"] = now()
        with LOCK:
            self._write_json(self._bpath(b["id"]), b)

    def delete_batch(self, bid: str) -> bool:
        with LOCK:
            p = self._bpath(bid)
            if os.path.exists(p):
                os.remove(p)
            d = os.path.join(self.uploads_dir, bid)
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    try:
                        os.remove(os.path.join(d, fn))
                    except OSError:
                        pass
                try:
                    os.rmdir(d)
                except OSError:
                    pass
            # 导出包一起清掉：留着旧 zip 会让人以为批次还在，
            # 而且一个 12 张的批次光对照图就十几 MB
            exp = os.path.join(self.exports_dir, bid)
            if os.path.isdir(exp):
                shutil.rmtree(exp, ignore_errors=True)
            zpath = os.path.join(self.exports_dir, f"{bid}.zip")
            if os.path.exists(zpath):
                try:
                    os.remove(zpath)
                except OSError:
                    pass
            return True

    # ------------------------------------------------------------ 样本

    def add_sample(self, bid: str, filename: str, img: np.ndarray) -> dict:
        sid = uuid.uuid4().hex[:10]
        d = os.path.join(self.uploads_dir, bid)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{sid}.png")
        imwrite_unicode(path, img)
        b = self.get_batch(bid)
        if b is None:
            raise KeyError(f"批次不存在: {bid}")
        sample = {
            "id": sid,
            "filename": filename,
            "path": path,
            "w": int(img.shape[1]),
            "h": int(img.shape[0]),
            "added": now(),
            "diagnosis": None,
            "rounds": [],
            "stages": None,
            "selected_recipe": None,
            "accepted": False,
            "accepted_at": None,
            "edited": False,
            "edit_count": 0,
            "note": "",
        }
        b["samples"].append(sample)
        self.save_batch(b)
        return sample

    def get_sample(self, bid: str, sid: str):
        b = self.get_batch(bid)
        if not b:
            return None
        for s in b["samples"]:
            if s["id"] == sid:
                return s
        return None

    def update_sample(self, bid: str, sid: str, patch: dict) -> dict:
        with LOCK:
            b = self.get_batch(bid)
            if not b:
                raise KeyError(f"批次不存在: {bid}")
            for s in b["samples"]:
                if s["id"] == sid:
                    s.update(patch)
                    self.save_batch(b)
                    return s
        raise KeyError(f"样本不存在: {sid}")

    def load_image(self, sample: dict) -> np.ndarray | None:
        return imread_unicode(sample["path"])

    # ------------------------------------------------------------ 编辑流水

    def log_edit(self, entry: dict) -> None:
        """人工改动流水。这是"AI 越用越准"的唯一语料来源，append-only。"""
        entry = dict(entry)
        entry["ts"] = now()
        with LOCK:
            with open(self.edits_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_edits(self, limit: int = 200, batch_id: str | None = None) -> list[dict]:
        if not os.path.exists(self.edits_path):
            return []
        out = []
        with open(self.edits_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if batch_id and e.get("batch_id") != batch_id:
                    continue
                out.append(e)
        return out[-limit:]

    def edit_stats(self, batch_id: str | None = None) -> dict:
        """给界面和答辩看的一张小表：AI 首版被人改了多少。"""
        edits = self.read_edits(limit=100000, batch_id=batch_id)
        by_kind: dict[str, int] = {}
        by_op: dict[str, int] = {}
        for e in edits:
            by_kind[e.get("kind", "?")] = by_kind.get(e.get("kind", "?"), 0) + 1
            op = e.get("op")
            if op:
                by_op[op] = by_op.get(op, 0) + 1
        return {
            "total": len(edits),
            "by_kind": by_kind,
            "by_op": by_op,
            "recent": edits[-20:][::-1],
        }

    # ------------------------------------------------------------ 工具

    @staticmethod
    def _read_json(path: str):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _write_json(path: str, obj) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
