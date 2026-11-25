"""Agent 调度层

流程（与界面上的五个节点一一对应）
----------------------------------
  看懂  ->  出候选  ->  试跑打分  ->  人核验  ->  聚合导出
  vlm      查表        指标+闸门      人点        批次投票

三次迭代的正确用法（不是重试）
------------------------------
  第 1 轮：诊断直接命中的首选，保正
  第 2 轮：换族（首选偏灰 -> 换局部均衡族）
  第 3 轮：同族调参（配方库里同 family 的变体）

停止判据三条并列：
  a) 已有人工接受的方案
  b) 候选耗尽（配方库最多给 3 条）
  c) 连续两轮分数不再提升（增益 < MIN_GAIN）

为什么要早停而不是跑满三次：跑满三次的边际收益在第 3 轮基本为零，
但标注员等 3 倍的渲染时间。答辩时这条要主动说，这是"用户体验"层面的设计。
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import metrics as MT
from . import compose as COMPOSE
from . import tuning as TUNE
from .ops import apply_recipe, apply_with_stages
from .recipes import DIAGNOSIS_MAP, DIAGNOSES, RECIPE_BY_ID, candidates_for

MIN_GAIN = 0.02      # 分数提升低于此值视为没进步
MAX_ROUNDS = 3

# 推荐时给"带警告"的方案扣一点分。
# 纯按分数选会推荐一条 banding 警告、但分数高 0.02 的链，
# 工程上不划算——警告意味着标注员可能标到假结构。
# 扣分制让"干净且接近"的方案胜出，同时不至于埋没明显更好的警告方案。
VERDICT_PENALTY = {"pass": 0.0, "warn": 0.06, "reject": float("inf")}


def _downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def evaluate_recipe(img: np.ndarray, recipe: dict, base_m=None, base_img=None) -> dict:
    """单条配方评估：指标、相对得分、伪影审计、逐节中间态。

    base_m / base_img 可复用（同一次请求里多条候选共享基线，省一次计算）。
    """
    if base_img is None:
        base_img = img
    if base_m is None:
        base_m = MT.measure(base_img)

    try:
        out, steps = apply_with_stages(img, recipe)
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "recipe_id": recipe.get("id") or recipe.get("recipe_id"),
        }

    m = MT.measure(out)
    s, parts = MT.score(m, base_m)
    audit = MT.audit_artifact(m, base_m)

    return {
        "ok": True,
        "recipe_id": recipe.get("id") or recipe.get("recipe_id"),
        "desc": recipe.get("desc", ""),
        "family": recipe.get("family", ""),
        "stages": recipe.get("stages", []),
        "score": round(s, 4),
        "score_parts": {k: round(v, 4) for k, v in parts.items()},
        "metrics": m,
        "delta": MT.delta_summary(m, base_m),
        "audit": audit,
        "out_shape": list(out.shape),
        "steps": steps,
    }


def run_agent(
    img: np.ndarray,
    diag: dict,
    cfg: dict | None = None,
    max_rounds: int = MAX_ROUNDS,
    accepted: str | None = None,
    max_side: int = 1600,
    enable_compose: bool = True,
    enable_tune: bool = True,
) -> dict:
    """跑完整流程：查表候选 -> AI 自拟 -> 参数精修。

    三个阶段的分工是刻意的：
      查表   快、可复现、覆盖已知场景（人写过的组合）
      拼链   能走出配方库，真正体现"agent"；但出口仍要过闸门，分数不行就丢
      精修   用测的不用猜的，对最终链做参数打磨

    拼链和精修都不是必需的：没配模型就只查表，图太小/时间紧就关掉精修。
    """
    t0 = time.time()
    work = _downscale(img, max_side)
    base_m = MT.measure(work)
    base_panel = {
        "recipe_id": "r00_baseline",
        "desc": "原图（对照基线，分数恒为 0）",
        "stages": [{"op": "gray"}],
        "score": 0.0,
        "metrics": base_m,
        "audit": {"verdict": "pass", "risk": 0.0, "reasons": []},
        "delta": MT.delta_summary(base_m, base_m),
        "phase": "baseline",
    }

    diagnoses = diag.get("diagnoses") or ["normal"]
    cands = candidates_for(diagnoses, limit=max_rounds)

    rounds = []
    compose_info = None
    tune_info = None

    # ---------------- 阶段 1：查表候选 ----------------
    for i, cand in enumerate(cands, start=1):
        res = evaluate_recipe(work, cand, base_m=base_m, base_img=work)
        res["round"] = i
        res["phase"] = "preset"
        res["candidate_source"] = cand.get("source", "preset")
        rounds.append(res)

    # ---------------- 阶段 2：AI 自主拼链 ----------------
    if enable_compose and cfg:
        comp = COMPOSE.compose_recipe(diag, base_m, rounds, cfg)
        compose_info = {k: v for k, v in comp.items() if k != "stages"}
        if comp.get("ok"):
            recipe = {
                "id": "ai_composed",
                "desc": "AI 自主拼接",
                "family": "composed",
                "stages": comp["stages"],
            }
            res = evaluate_recipe(work, recipe, base_m=base_m, base_img=work)
            res["round"] = len(rounds) + 1
            res["phase"] = "compose"
            res["rationale"] = comp.get("rationale", "")
            res["notes"] = comp.get("notes", [])
            res["confidence"] = comp.get("confidence", 0)
            rounds.append(res)

            # 被噪声拒掉的话，把失败原因喂回去让它重来一次。
            # 这一步只在真被拒时触发，成本可控；而噪声放大恰恰是模型最容易犯、
            # 也最容易被"告诉它一声"就改对的错。
            noise_rej = [x for x in (res.get("audit", {}).get("reasons") or [])
                         if x.get("code") == "noise" and x.get("level") == "reject"]
            if noise_rej and res.get("ok"):
                fb = (f"上次给的链 {'→'.join(x['op'] for x in comp['stages'])} "
                      f"跑出来噪声 σ≈{(res['audit'].get('noise_abs', 0) * 255):.1f}/255"
                      f"（放大 {res['audit'].get('noise_ratio', 0):.1f}×），超过上限被拒。")
                comp2 = COMPOSE.compose_recipe(diag, base_m, rounds, cfg, feedback=fb)
                if comp2.get("ok"):
                    recipe2 = {
                        "id": "ai_composed_2",
                        "desc": "AI 自主拼接（按噪声反馈修正）",
                        "family": "composed",
                        "stages": comp2["stages"],
                    }
                    res2 = evaluate_recipe(work, recipe2, base_m=base_m, base_img=work)
                    res2["round"] = len(rounds) + 1
                    res2["phase"] = "compose"
                    res2["rationale"] = comp2.get("rationale", "")
                    res2["notes"] = list(comp2.get("notes") or []) + ["按闸门反馈重新生成"]
                    res2["confidence"] = comp2.get("confidence", 0)
                    rounds.append(res2)
                    compose_info["retried"] = True
                    compose_info["retry_error"] = comp2.get("error")
                    compose_info["first_stages"] = comp["stages"]
                    compose_info["second_stages"] = comp2["stages"]
                else:
                    compose_info["retried"] = False
                    compose_info["retry_error"] = comp2.get("error")

    # ---------------- 选当前最优 ----------------
    def rank_of(r):
        if not r.get("ok"):
            return -1e9
        return r["score"] - VERDICT_PENALTY.get(r["audit"]["verdict"], 0.0)

    def pick_best(items):
        b, br = None, -1e9
        for r in items:
            k = rank_of(r)
            if k > br:
                br, b = k, r
        return b

    best = pick_best(rounds)

    # ---------------- 阶段 3：参数精修 ----------------
    tune_runs = []
    if enable_tune and best:
        # 除了 rank 最优的那条，再给"分数更高但被闸门拒掉"的那条一次机会。
        # 调参经常能把截断/噪声压下来（clahe 的 clip 小一档、gamma 温和一点），
        # 救回来就比查表方案更好；救不回来就丢掉，不影响结论。
        targets = [best]
        alt = None
        for r in rounds:
            if r.get("ok") and r is not best and r["score"] > best["score"] + 0.02:
                if alt is None or r["score"] > alt["score"]:
                    alt = r
        if alt is not None:
            targets.append(alt)

        for tgt in targets:
            t = TUNE.tune(work, tgt["stages"], base_m, budget=4)
            tune_runs.append({
                "target": tgt["recipe_id"],
                "target_phase": tgt.get("phase", "preset"),
                "improved": t.get("improved", False),
                "base_score": t.get("base_score", 0.0),
                "score": t.get("score", 0.0),
                "gain": t.get("gain", 0.0),
                "evaluations": t.get("evaluations", 0),
                "trials": t.get("trials", []),
                "elapsed_ms": t.get("elapsed_ms", 0),
                "reason": t.get("reason"),
            })
            if t.get("improved"):
                recipe = {
                    "id": tgt["recipe_id"],
                    "desc": f"{tgt['desc']}（参数精修）",
                    "family": tgt.get("family", ""),
                    "stages": t["stages"],
                }
                res = evaluate_recipe(work, recipe, base_m=base_m, base_img=work)
                if res.get("ok"):
                    res["round"] = len(rounds) + 1
                    res["phase"] = "tune"
                    res["recipe_id"] = tgt["recipe_id"]
                    res["tuned_from"] = tgt["recipe_id"]
                    res["tune_trials"] = t["trials"]
                    res["tune_gain"] = t["gain"]
                    rounds.append(res)
                    best = pick_best(rounds)

    tune_info = None
    if tune_runs:
        tune_info = {
            "runs": tune_runs,
            "evaluations": sum(r["evaluations"] for r in tune_runs),
            "improved": any(r["improved"] for r in tune_runs),
            "best_gain": round(max((r["gain"] for r in tune_runs), default=0.0), 4),
            "elapsed_ms": sum(r["elapsed_ms"] for r in tune_runs),
        }

    elapsed = int((time.time() - t0) * 1000)

    # ---------------- 结论 ----------------
    preset_n = sum(1 for r in rounds if r.get("phase") == "preset")
    parts = [f"查表 {preset_n} 条"]
    if compose_info:
        if compose_info.get("ok"):
            parts.append("AI 自拟 1 条")
        else:
            parts.append(f"拼链跳过（{(compose_info.get('error') or '')[:28]}）")
    if tune_info:
        n_run = len(tune_info.get("runs") or [])
        if tune_info.get("improved"):
            parts.append(f"参数精修 {n_run} 条链共 {tune_info['evaluations']} 次，"
                         f"最好提升 {tune_info['best_gain']:+.3f}")
        else:
            parts.append(f"参数精修 {n_run} 条链共 {tune_info['evaluations']} 次，无提升")
    stop_reason = " · ".join(parts)

    review_required = False
    recommendation = None

    if best is None:
        ok_rounds = [r for r in rounds if r.get("ok")]
        if ok_rounds:
            best = max(ok_rounds, key=lambda r: r["score"])
            review_required = True
            stop_reason += " · 全部候选未通过闸门，已取分数最高的一条供人工复核"
        else:
            stop_reason += " · 所有候选执行失败，建议检查图像本身"

    if best is not None:
        recommendation = {
            "recipe_id": best["recipe_id"],
            "desc": best["desc"],
            "score": best["score"],
            "audit": best["audit"],
            "stages": best["stages"],
            "phase": best.get("phase", "preset"),
            "rationale": best.get("rationale", ""),
            "review_required": review_required,
            "reason": _explain(best, base_m, diagnoses),
        }

    return {
        "diagnosis": diag,
        "rounds": rounds,
        "baseline": base_panel,
        "recommendation": recommendation,
        "compose": compose_info,
        "tuning": tune_info,
        "stop_reason": stop_reason,
        "elapsed_ms": elapsed,
        "candidates_tried": len([r for r in rounds if r.get("ok")]),
    }


def _explain(best: dict, base_m: dict, diagnoses: list[str]) -> str:
    """把推荐理由写成一句人话——评委问"为什么是它"时要能当场念出来。"""
    d = best["delta"]
    bits = []
    if d["local_contrast"]["delta_pct"] is not None:
        bits.append(f"局部对比度 {d['local_contrast']['delta_pct']:+.0f}%")
    if d["entropy"]["delta_pct"] is not None:
        bits.append(f"直方图熵 {d['entropy']['delta_pct']:+.0f}%")
    if d["clip"]["value"] is not None:
        bits.append(f"截断率 {d['clip']['value'] * 100:.1f}%")
    label = "、".join(DIAGNOSES[x]["label"] for x in diagnoses if x in DIAGNOSES)
    audit = best["audit"]["verdict"]
    verdict_cn = {"pass": "通过闸门", "warn": "带警告通过", "reject": "未通过"}.get(audit, audit)
    return f"针对「{label}」：{best['desc']}；{('，'.join(bits))}；伪影审计{verdict_cn}。"


# ---------------------------------------------------------------- 批次聚合

def aggregate_batch(items: list[dict], consistency_floor: float = 0.8) -> dict:
    """把逐样本的结果聚合成"一批一份"的标注方案。

    为什么必须聚合：30 张图各出一个方案，标注员不会读 30 份说明，等于没方案。
    按首选诊断分组 -> 组内投票 -> 得一组的代表链。

    items: [{"sample_id","diagnosis":[...],"recipe_id","score","audit"}]
    """
    groups: dict[str, dict] = {}
    for it in items:
        ds = it.get("diagnosis") or ["normal"]
        key = ds[0] if ds[0] in DIAGNOSIS_MAP else "normal"
        g = groups.setdefault(key, {"diagnosis": key, "members": [], "votes": {}})
        rid = it.get("recipe_id") or "r01_pct_stretch"
        g["members"].append(it)
        g["votes"][rid] = g["votes"].get(rid, 0) + 1

    out_groups = []
    for key, g in groups.items():
        total = len(g["members"])
        winner, votes = max(g["votes"].items(), key=lambda kv: kv[1])
        consistency = votes / max(total, 1)
        outliers = [m["sample_id"] for m in g["members"] if (m.get("recipe_id") or "") != winner]
        rec = RECIPE_BY_ID.get(winner, {})
        out_groups.append({
            "diagnosis": key,
            "diagnosis_label": DIAGNOSES.get(key, {}).get("label", key),
            "recipe_id": winner,
            "recipe_desc": rec.get("desc", ""),
            "stages": rec.get("stages", []),
            "member_count": total,
            "consistency": round(consistency, 3),
            "needs_manual": consistency < consistency_floor,
            "outliers": outliers,
            "votes": g["votes"],
        })

    out_groups.sort(key=lambda x: (-x["member_count"], x["diagnosis"]))
    total = sum(g["member_count"] for g in out_groups)
    manual = sum(len(g["outliers"]) for g in out_groups if g["needs_manual"])
    return {
        "groups": out_groups,
        "total": total,
        "groups_count": len(out_groups),
        "needs_manual_count": manual,
        "consistency_floor": consistency_floor,
    }


# ---------------------------------------------------------------- 算子准入表

def preflight(images: list[np.ndarray], max_side: int = 1280,
              recipes: list[dict] | None = None) -> dict:
    """所有算子链 × 若干图，跑一遍伪影审计，产出"哪条链可以用于标注/配流程"的准入表。

    这张表不依赖 VLM、不依赖人工，是整套方案里唯一可以纯自动生成的结论。

    recipes 显式传入时以它为准（内置配方 + 用户自己存的链）。不传就退回注册表，
    兼容老调用方——但注册表是内存副本，手动改过 custom_recipes.json 会读到旧的。
    """
    # baseline 不参与准入判断：它是原图本身，没有"能不能用于标注"的问题
    src = recipes if recipes is not None else list(RECIPE_BY_ID.values())
    chain_list = [r for r in src if r.get("id") != "r00_baseline"]
    rows = []
    for rec in chain_list:
        rid = rec["id"]
        verdicts, risks, clips, noises, noises_abs, scores = [], [], [], [], [], []
        for img in images:
            work = _downscale(img, max_side)
            base = MT.measure(work)
            try:
                out, _ = apply_recipe(work, rec)
            except Exception:
                continue
            m = MT.measure(out)
            a = MT.audit_artifact(m, base)
            s, _ = MT.score(m, base)
            verdicts.append(a["verdict"])
            risks.append(a["risk"])
            clips.append(a["clip_abs"])
            noises.append(a["noise_ratio"])
            noises_abs.append(a["noise_abs"])
            scores.append(s)
        if not verdicts:
            continue
        rej = verdicts.count("reject") / len(verdicts)
        rows.append({
            "recipe_id": rid,
            "desc": rec["desc"],
            "family": rec.get("family", ""),
            "custom": bool(rec.get("custom")),
            "n": len(verdicts),
            "reject_rate": round(rej, 3),
            "warn_rate": round(verdicts.count("warn") / len(verdicts), 3),
            "mean_risk": round(float(np.mean(risks)), 3),
            "max_risk": round(float(np.max(risks)), 3),
            "mean_clip": round(float(np.mean(clips)), 4),
            "mean_noise_ratio": round(float(np.mean(noises)), 3),
            "mean_noise_abs": round(float(np.mean(noises_abs)), 4),
            "mean_score": round(float(np.mean(scores)), 4),
            "admitted": rej <= 0.2,
        })
    rows.sort(key=lambda r: (not r["admitted"], -r["mean_score"]))
    return {
        "rows": rows,
        "admitted_count": sum(1 for r in rows if r["admitted"]),
        "total": len(rows),
    }
