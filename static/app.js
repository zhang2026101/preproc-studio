/* Preproc Studio —— 前端逻辑
 *
 * 一条原则贯穿全局：**只有结构性改动才重绘算子链**。
 * 拖滑杆时如果重绘 DOM，滑杆会失焦，手感直接废掉。
 * 所以参数变化只更新数值标签、缩略图和右侧面板。
 */

const S = {
  schema: null,
  batches: [],
  batch: null,
  batchId: null,
  sample: null,
  sampleId: null,
  analysis: null,
  stages: [],
  steps: [],
  activeStep: -1,
  viewMode: "final",
  rejected: [],
  busy: false,
  previewToken: 0,
  previewTimer: null,
  aiStages: null,
  phase: "preset",
  rationale: "",
  stageTimer: null,
  chainMs: 0,
  pending: false,
  lightboxToken: 0,
  lb: null,
  roundsCache: null,
  metricsCache: null,
  auditCache: null,
  roundsOpen: false,
  metricsOpen: false,
  obsOpen: false,
  auditOpen: false,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fmt(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v !== "number") return String(v);
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(digits);
}

function pct(v) {
  if (v === null || v === undefined) return "—";
  return (v > 0 ? "+" : "") + v.toFixed(0) + "%";
}

function toast(msg, isErr) {
  // 放进统一的容器里竖着叠：原来每条都绝对定位在同一个位置，
  // 连续操作时两条会互相盖住（连点换一条 + 通过核验就是这种情况）
  let root = document.getElementById("toastRoot");
  if (!root) {
    root = document.createElement("div");
    root.id = "toastRoot";
    document.body.appendChild(root);
  }
  const el = document.createElement("div");
  el.className = "toast" + (isErr ? " err" : "");
  el.textContent = msg;
  root.appendChild(el);
  setTimeout(() => el.remove(), isErr ? 4200 : 2400);
}

function status(text, meta) {
  $("#statusText").textContent = text;
  if (meta !== undefined) $("#statusMeta").textContent = meta;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const ct = res.headers.get("content-type") || "";
  if (!res.ok) {
    let detail = res.statusText;
    if (ct.includes("json")) {
      const j = await res.json().catch(() => null);
      detail = (j && (j.detail || j.message)) || detail;
    }
    throw new Error(detail);
  }
  if (ct.includes("json")) return res.json();
  return res;
}

function postJson(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
}

/* ---------------------------------------------------------------- 顶栏/批次 */

async function loadHealth() {
  try {
    const h = await api("/api/health");
    const dot = $("#health .dot");
    const label = $("#health span:last-child");
    const providerCn = { cloud: "线上模型", local: "本地模型", rules: "本地规则" }[h.provider] || h.provider;
    dot.className = "dot " + (h.provider === "rules" ? "warn" : "ok");
    const tail = h.provider !== "rules" && h.model ? ` · ${h.model}` : "";
    label.textContent = `${providerCn}${tail} · ${h.ops} 算子 · ${h.batches} 批次`;
    $("#health").title = h.has_key
      ? `模型 ${h.model}  已配置 key`
      : "未配置 API key，使用本地规则诊断。点「设置」可接 DeepSeek / 百炼 / 本地模型";
  } catch (e) {
    $("#health .dot").className = "dot bad";
    $("#health span:last-child").textContent = "后端未连接";
  }
}

async function loadBatches(selectId) {
  const d = await api("/api/batches");
  S.batches = d.batches || [];
  const sel = $("#batchSelect");
  if (!S.batches.length) {
    sel.innerHTML = `<option value="">（无批次）</option>`;
    // 删掉最后一个批次后要把界面清干净：不清的话左栏还留着已删批次的样本，
    // 点一下就是 404，看起来像"删了但没删掉"
    S.batchId = null;
    S.batch = null;
    S.sampleId = null;
    S.sample = null;
    S.analysis = null;
    S.stages = [];
    S.steps = [];
    S.activeStep = -1;
    renderSampleList();
    clearCenter();
    status("还没有批次", "新建批次，或把图片拖进来");
    return;
  }
  sel.innerHTML = S.batches.map((b) =>
    `<option value="${b.id}">${esc(b.name)} · ${b.count} 张</option>`
  ).join("");
  const target = selectId || S.batchId || S.batches[0].id;
  sel.value = target;
  await selectBatch(target);
}

/* ---------------------------------------------------------------- 删除批次 */

function deleteBatchDialog() {
  if (!S.batchId || !S.batch) return toast("先选中一个批次", true);
  const list = S.batch.samples || [];
  const acc = list.filter((s) => s.accepted).length;
  openModal("删除批次", `
    <p style="margin:0 0 10px">要删除批次 <b>${esc(S.batch.name)}</b>
      （${list.length} 张样本${acc ? `，其中 ${acc} 张已通过核验` : ""}）吗？</p>
    <div class="reason">
      这批的原图、批次记录和已导出的配置包会一起删掉，<b>不可恢复</b>。
      ${acc ? "已核验的结论也会一起消失——还要用的话，先导出配置包再删。" : ""}
    </div>`,
    `<button onclick="closeModal()">取消</button>
     <button class="danger" onclick="doDeleteBatch()">删除</button>`);
}

async function doDeleteBatch() {
  const id = S.batchId;
  const name = S.batch ? S.batch.name : id;
  closeModal();
  try {
    await api(`/api/batches/${id}`, { method: "DELETE" });
    toast(`已删除批次「${name}」`);
    S.batchId = null;
    S.batch = null;
    await loadBatches();
    await loadHealth();
  } catch (e) {
    toast("删除失败：" + e.message, true);
  }
}

window.deleteBatchDialog = deleteBatchDialog;
window.doDeleteBatch = doDeleteBatch;

async function selectBatch(id) {
  if (!id) return;
  S.batchId = id;
  const d = await api(`/api/batches/${id}`);
  S.batch = d.batch;
  S.sampleId = null;
  S.sample = null;
  S.analysis = null;
  S.stages = [];
  S.steps = [];
  S.activeStep = -1;
  renderSampleList();
  clearCenter();
  status(`已加载批次「${S.batch.name}」`, `${S.batch.samples.length} 个样本`);
  await loadHealth();
}

function renderSampleList() {
  const box = $("#sampleList");
  const list = (S.batch && S.batch.samples) || [];
  $("#sampleCount").textContent = list.length;
  if (!list.length) {
    box.innerHTML = `<div class="empty-hint"><div class="empty-figure"></div>批次为空<br>拖图片进来，或从目录导入</div>`;
    return;
  }
  box.innerHTML = list.map((s) => {
    const flags = [];
    if (s.accepted) flags.push('<i class="flag accepted" title="已通过核验"></i>');
    if (s.edited) flags.push('<i class="flag edited" title="人工改过"></i>');
    if (s.review_required && !s.accepted) flags.push('<i class="flag warn" title="闸门全灭，需人工复核"></i>');
    const diag = (s.diagnosis && s.diagnosis.diagnoses) ? s.diagnosis.diagnoses[0] : "";
    const sub = diag
      ? `${esc(labelOfDiagnosis(diag))} · ${esc(s.selected_recipe || "")}`
      : (s.w ? `${s.w}×${s.h}` : "");
    return `<div class="sample-item ${s.id === S.sampleId ? "active" : ""}" data-sid="${s.id}">
      <img class="thumb" loading="lazy" src="/api/batches/${S.batchId}/samples/${s.id}/image?max_side=160&raw=1" alt="">
      <div class="sample-meta">
        <div class="sample-name" title="${esc(s.filename)}">${esc(s.filename)}</div>
        <div class="sample-sub">${sub}</div>
      </div>
      <div class="sample-flags">${flags.join("")}</div>
    </div>`;
  }).join("");
}

function labelOfDiagnosis(key) {
  const d = S.schema && S.schema.diagnoses[key];
  return d ? d.label : key;
}

/* ---------------------------------------------------------------- 样本 */

async function selectSample(sid) {
  S.sampleId = sid;
  S.sample = (S.batch.samples || []).find((x) => x.id === sid) || null;
  S.analysis = null;
  S.rejected = [];
  S.roundsOpen = false;
  S.metricsOpen = false;
  S.obsOpen = false;
  renderSampleList();

  const s = S.sample;
  if (!s) return;

  S.stages = JSON.parse(JSON.stringify(s.stages || []));
  S.steps = [];
  S.activeStep = -1;
  renderCandidateSwitcher();

  const matchRound = (s.rounds || []).find((r) => r.ok && r.stages &&
    JSON.stringify(r.stages) === JSON.stringify(S.stages));
  S.chainMs = s.chain_ms || 0;
  S.phase = matchRound ? (matchRound.phase || "preset") : "preset";
  S.rationale = matchRound ? (matchRound.rationale || "") : "";

  if (s.diagnosis && S.stages.length) {
    // 恢复已有状态：诊断 + 链，预览重算一遍（不花模型 token）
    renderDiagnosis(s.diagnosis, null);
    renderRounds(s.rounds || [], "");
    renderChain();
    await previewChain(true);
  } else {
    clearCenter(true);
    await showOriginal();
  }
  status(`已选中 ${s.filename}`, s.accepted ? "已通过核验" : "");
}

async function showOriginal() {
  if (!S.sampleId) return;
  try {
    const d = await api(`/api/batches/${S.batchId}/samples/${S.sampleId}/image?max_side=900`);
    $("#viewer").innerHTML = `<img src="data:image/jpeg;base64,${d.image}" alt="">`;
    $("#viewerLabel").textContent = "原图";
    S.steps = [{ step: 0, op: "原图", label: "原图", img: d.image, score: 0, delta: null, audit: null }];
    S.activeStep = 0;
    renderStepStrip();
  } catch (e) {
    $("#viewer").innerHTML = `<div class="empty-hint">读图失败：${esc(e.message)}</div>`;
  }
}

function clearCenter(keepViewer) {
  $("#diagBody").innerHTML = `<div class="empty-hint">选中样本后分析，或从迭代记录里挑一条链</div>`;
  $("#diagProvider").textContent = "";
  $("#chain").innerHTML = `<div class="empty-hint">暂无算子链</div>`;
  S.roundsCache = null;
  S.metricsCache = null;
  S.auditCache = null;
  syncEntryButtons();
  $("#stepStrip").innerHTML = "";
  if (!keepViewer) {
    $("#viewer").innerHTML = `<div class="empty-hint">暂无图像</div>`;
    $("#viewerLabel").textContent = "预览";
  }
}

/* ---------------------------------------------------------------- AI 分析 */

async function analyzeCurrent() {
  if (!S.sampleId || S.busy) return;
  S.busy = true;
  const btn = $("#aiAgainBtn");
  btn.disabled = true;
  S.roundsOpen = false;
  S.obsOpen = false;
  status("AI 正在诊断并试跑候选…", "最多 3 轮");

  // 全流程 8-25 秒，耗时主要取决于模型思考长度，波动很大。
  // 不报进度的话，用户会以为卡死。这里按阶段轮播文案（不假装精确百分比）。
  const STEPS = [
    "正在看图诊断成像问题…",
    "试跑配方候选…",
    "正在生成候选算子链…",
    "参数精修搜索…",
    "整理结果…",
  ];
  let stepIdx = 0;
  const paint = () => {
    const b = $("#diagBody");
    if (b && b.querySelector(".spin")) {
      b.innerHTML = `<div class="empty-hint"><span class="spin"></span> ${STEPS[stepIdx]}</div>`;
    }
  };
  paint();
  S.stageTimer = setInterval(() => {
    stepIdx = Math.min(stepIdx + 1, STEPS.length - 1);
    paint();
  }, 3800);

  try {
    const d = await postJson(`/api/batches/${S.batchId}/samples/${S.sampleId}/analyze`, {});
    S.analysis = d;
    S.stages = JSON.parse(JSON.stringify(d.stages || []));
    S.aiStages = JSON.parse(JSON.stringify(d.stages || []));
    S.rejected = [];

    renderDiagnosis(d.diagnosis, d);
    renderRounds(d.rounds || [], d.stop_reason);
    renderChain();
    renderCandidateSwitcher();

    const rec = d.recommendation;
    S.chainMs = d.chain_ms || 0;
    S.phase = rec ? rec.phase : "preset";
    S.rationale = rec ? (rec.rationale || "") : "";
    const panel = rec ? (d.panels || []).find((p) => p.recipe_id === rec.recipe_id) : null;
    S.steps = rec
      ? (d.rounds.find((r) => r.recipe_id === rec.recipe_id) || {}).steps || []
      : [];
    if (S.steps.length) {
      S.activeStep = S.steps.length - 1;
      renderViewer();
      renderStepStrip();
      renderMetrics();
      renderAudit();
    } else if (panel) {
      $("#viewer").innerHTML = `<img src="data:image/jpeg;base64,${panel.img}" alt="">`;
    }

    status(`分析完成 · 试了 ${d.candidates_tried} 条 · ${d.elapsed_ms} ms`, d.stop_reason);
    await refreshSampleFlags();
  } catch (e) {
    toast("分析失败：" + e.message, true);
    status("分析失败");
    renderDiagnosis(S.sample && S.sample.diagnosis, null);
  } finally {
    if (S.stageTimer) { clearInterval(S.stageTimer); S.stageTimer = null; }
    S.busy = false;
    btn.disabled = false;
  }
}

async function refreshSampleFlags() {
  const d = await api(`/api/batches/${S.batchId}`);
  S.batch = d.batch;
  S.sample = (S.batch.samples || []).find((x) => x.id === S.sampleId) || S.sample;
  renderSampleList();
}

/* ---------------------------------------------------------------- 诊断卡 */

function renderDiagnosis(diag, full) {
  if (!diag) {
    $("#diagBody").innerHTML = `<div class="empty-hint">未诊断</div>`;
    return;
  }
  const ds = diag.diagnoses || [];
  const tags = ds.map((k) => {
    const info = (S.schema && S.schema.diagnoses[k]) || { label: k };
    return `<span class="diag-tag" title="${esc(info.desc || "")}">${esc(info.label)}</span>`;
  }).join("");

  const providerCn = { cloud: "线上模型", local: "本地模型", rules: "本地规则" }[diag.provider] || diag.provider;
  $("#diagProvider").textContent = providerCn + (diag.cached ? " · 缓存" : "");

  let html = `<div class="diag-tags">${tags || '<span class="diag-tag weak">无标签</span>'}</div>`;
  if (diag.observation) {
    const long = diag.observation.length > 40;
    html += `<div class="diag-obs ${long ? (S.obsOpen ? "clamp expanded" : "clamp") : ""}"
      onclick="toggleObs()" title="${long ? "点击展开/收起" : ""}">${esc(diag.observation)}</div>`;
  }

  const rows = [];
  const conf = diag.confidence;
  rows.push(`<span>置信度 <b>${conf > 0 ? (conf * 100).toFixed(0) + "%" : "—（规则判定无置信度）"}</b></span>`);
  // 命中缓存时 elapsed 是 0，显示"耗时 0 ms"只会让人以为是 bug
  if (diag.elapsed_ms > 0) rows.push(`<span>耗时 <b>${diag.elapsed_ms} ms</b></span>`);
  if (full && full.candidates_tried) rows.push(`<span>试跑 <b>${full.candidates_tried} 条</b></span>`);
  if (full && full.compose) {
    const okc = full.compose.ok;
    rows.push(`<span>AI 拼链 <b style="color:${okc ? "var(--ok)" : "var(--text-3)"}">${okc ? "成功" : "跳过"}</b></span>`);
  }
  if (full && full.tuning) {
    const im = full.tuning.improved;
    rows.push(`<span>参数精修 <b>${full.tuning.evaluations} 次${im ? "，有提升" : ""}</b></span>`);
  }
  if (full && full.recommendation && full.recommendation.review_required) {
    rows.push(`<span style="color:var(--warn)">闸门全灭，需人工复核</span>`);
  }
  html += `<div class="diag-row">${rows.join("")}</div>`;

  if (diag.fallback_reason) {
    html += `<div class="reason warn">视觉接口不可用，已降级为规则诊断：${esc(diag.fallback_reason)}</div>`;
  }
  $("#diagBody").innerHTML = html;
}

/* ---------------------------------------------------------------- 算子链 */

function paramControl(opSpec, key, spec, value, i) {
  if (spec.type === "enum") {
    const opts = spec.options.map((o) =>
      `<option value="${esc(o)}" ${String(value) === String(o) ? "selected" : ""}>${esc(o)}</option>`
    ).join("");
    return `<label class="param" title="${esc(spec.label)}">
      <span>${esc(key)}</span>
      <select data-i="${i}" data-k="${key}">${opts}</select>
    </label>`;
  }
  if (spec.type === "floats") {
    const v = Array.isArray(value) ? value.join(",") : String(value || "");
    return `<label class="param" title="逗号分隔">
      <span>${esc(key)}</span>
      <input type="text" data-i="${i}" data-k="${key}" data-t="floats" value="${esc(v)}" style="width:74px">
    </label>`;
  }
  const isInt = spec.type === "int";
  const step = spec.step || (isInt ? 1 : 0.1);
  const shown = isInt ? Math.round(Number(value)) : Number(value).toFixed(2);
  return `<label class="param" title="${esc(spec.label)}">
    <span>${esc(key)}</span>
    <input type="range" data-i="${i}" data-k="${key}" data-t="${spec.type}"
      min="${spec.min}" max="${spec.max}" step="${step}" value="${value}">
    <span class="param-val" data-val-for="${i}-${key}">${shown}</span>
  </label>`;
}

function renderChain() {
  const src = {
    preset: "来自配方库 · 可复现",
    compose: "AI 自主拼接 · 已过校验与闸门",
    tune: "参数精修后 · 搜索得来",
  }[S.phase] || "自上而下执行";
  const hint = $("#chainHint");
  if (hint) hint.textContent = src;

  const box = $("#chain");
  if (!S.stages.length) {
    box.innerHTML = `<div class="empty-hint"><div class="empty-figure"></div>暂无算子链<br><span class="small">可添加算子，或点「生成方案」</span></div>`;
    return;
  }
  const schemaFor = (op) => (S.schema.ops.find((o) => o.op === op) || { label: op, params: {}, hint: "" });

  // 行高与缩略图尺寸都由 CSS 固定（--op-h / --thumb-h / --thumb-w），
  // 不再按"容器高 ÷ 步数"反推：那样同一条链在 2 步和 6 步时会变大变小，
  // 而且行高被内容反向撑出来时还会循环依赖。

  const cards = S.stages.map((st, i) => {
    const spec = schemaFor(st.op);
    const params = Object.entries(st.p || {}).map(([k, v]) => {
      const ps = spec.params[k] || { type: "float", min: 0, max: 100, step: 1, label: k };
      return paramControl(spec, k, ps, v, i);
    }).join("");

    const step = S.steps[i + 1];
    const thumb = step
      ? `<img class="op-thumb" data-act="thumb" data-step="${i + 1}" src="data:image/jpeg;base64,${step.img}" title="点看该步输出">`
      : `<div class="op-thumb op-thumb-empty">—</div>`;

    // 整条链的来源已写在标题上，逐行再标"AI"是冗余信息。
    // 只有人工动过的地方值得单独提示。
    const isEdited = S.aiStages && JSON.stringify(S.stages) !== JSON.stringify(S.aiStages);
    const badge = isEdited ? `<span class="badge edited">已改</span>` : "";

    return `<div class="op-card ${isEdited ? "edited" : ""} ${S.activeStep === i + 1 ? "selected" : ""}" data-card="${i}">
        <span class="op-idx">${String(i + 1).padStart(2, "0")}</span>
        ${thumb}
        <div class="op-main">
          <div class="op-head">
            <span class="op-name" title="${esc(spec.hint || "")}">${esc(st.op)}</span>
            <span class="muted small">${esc(spec.label)}</span>
            ${badge}
          </div>
          <div class="op-params">${params || '<span class="no-param">该步不调参</span>'}</div>
        </div>
        <div class="op-tools">
          <button class="icon-btn" data-act="up" data-i="${i}" title="上移" ${i === 0 ? "disabled" : ""}>↑</button>
          <button class="icon-btn" data-act="down" data-i="${i}" title="下移" ${i === S.stages.length - 1 ? "disabled" : ""}>↓</button>
          <button class="icon-btn" data-act="del" data-i="${i}" title="删除">×</button>
        </div>
      </div>`;
  }).join("");

  // 链尾留一条虚线占位行，作为固定的最后一行（48px），
  // 在视线落点处给出「加一步」的入口，不用再回标题栏找。
  // 它放在 .chain-body 外面：body 承载两条竖直分列线，
  // 线只该穿过算子卡片，不该切到"添加算子"的虚框里。
  box.innerHTML = `<div class="chain-body">${cards}</div>`
    + `<div class="chain-add" data-act="add-op">＋ 添加算子</div>`;
  requestAnimationFrame(fitChainThumbs);
}

/* 行高固定后，缩略图尺寸由 CSS 变量（--thumb-h / --thumb-w）定死。
   原来按行高反推，会让同一张卡在 2 步和 6 步时忽大忽小。
   函数保留为空实现：调用点（首次渲染、窗口 resize）不用改。 */
function fitChainThumbs() {}

let thumbTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(thumbTimer);
  thumbTimer = setTimeout(() => {
    fitChainThumbs();
    positionCompareTags();
  }, 120);
});

/* ---------------------------------------------------------------- 轮次 */

function roundsHTML(rounds) {
  if (!rounds || !rounds.length) {
    return `<div class="empty-hint">还没有跑过</div>`;
  }
  const PHASE = {
    preset: { label: "查表", cls: "p-preset" },
    compose: { label: "AI 自拟", cls: "p-compose" },
    tune: { label: "参数精修", cls: "p-tune" },
  };
  const picked = S.sample ? S.sample.selected_recipe : null;

  return rounds.map((r) => {
    const ph = PHASE[r.phase] || PHASE.preset;

    if (!r.ok) {
      return `<div class="round-row"><span class="phase-tag ${ph.cls}">${ph.label}</span>
        <span class="round-desc">${esc(r.recipe_id || "")} 执行失败</span>
        <span class="round-score">—</span><span></span></div>`;
    }

    const v = r.audit ? r.audit.verdict : "pass";
    const badge = v === "reject" ? `<span class="badge danger">拒绝</span>`
      : v === "warn" ? `<span class="badge warn">告警</span>`
        : `<span class="badge ok">可过</span>`;

    const ops = (r.stages || []).map((x) => x.op).join("→");
    const sub = r.phase === "compose" ? ops : `${esc(r.recipe_id)} · ${esc(r.desc || "")}`;
    const active = r.recipe_id === picked ? "active" : "";

    let extra = "";
    if (r.rationale) extra += `<div class="round-note">理由：${esc(r.rationale)}</div>`;
    if (r.notes && r.notes.length) {
      extra += `<div class="round-note muted">校验：${esc(r.notes.join("；"))}</div>`;
    }
    if (r.audit && r.audit.reasons && r.audit.reasons.length) {
      const rs = r.audit.reasons[0];
      extra += `<div class="round-note gate ${rs.level === "reject" ? "bad" : ""}">闸门：${esc(rs.text)}</div>`;
    }
    if (r.tune_trials && r.tune_trials.length) {
      const acc = r.tune_trials.filter((t) => t.accepted);
      const txt = acc.length
        ? acc.map((t) => `${t.op}.${t.param} ${t.from}→${t.to}`).join("，")
        : `${r.tune_trials.length} 次尝试均未提升`;
      extra += `<div class="round-note">精修：${esc(txt)}</div>`;
    }

    return `<div class="round-row ${active}" data-round="${esc(r.recipe_id)}"
      data-stages='${esc(JSON.stringify(r.stages))}'>
      <span class="phase-tag ${ph.cls}">${ph.label}</span>
      <span class="round-desc" title="${esc(sub)}">${sub}</span>
      <span class="round-score">${r.score > 0 ? "+" : ""}${fmt(r.score, 3)}</span>
      <span style="text-align:right">${badge}</span>
    </div>${extra}`;
  }).join("");
}

function renderRounds(rounds, stopReason) {
  const from = {
    preset: (rounds || []).filter((r) => r.phase === "preset").length,
    compose: (rounds || []).filter((r) => r.phase === "compose").length,
    tune: (rounds || []).filter((r) => r.phase === "tune").length,
  };
  const parts = [];
  if (from.preset) parts.push(`查表 ${from.preset}`);
  if (from.compose) parts.push(`自拟 ${from.compose}`);
  if (from.tune) parts.push(`精修 ${from.tune}`);
  S.roundsCache = { rounds: rounds || [], summary: parts.join(" · "), stopReason: stopReason || "" };
  syncEntryButtons();
}

function openRoundsModal() {
  const c = S.roundsCache;
  if (!c || !c.rounds.length) return toast("还没有跑过迭代", true);
  openModal(`迭代记录　${c.summary}`,
    `<div class="modal-table">${roundsHTML(c.rounds)}</div>
     ${c.stopReason ? `<p class="muted" style="margin:12px 0 0;font-size:13px">${esc(c.stopReason)}</p>` : ""}`,
    `<button onclick="closeModal()">关闭</button>`);

  $("#modalRoot .modal-table").addEventListener("click", (e) => {
    const row = e.target.closest("[data-round]");
    if (!row) return;
    const r = c.rounds.find((x) => x.recipe_id === row.dataset.round);
    if (r && r.stages) {
      S.stages = JSON.parse(JSON.stringify(r.stages));
      S.phase = r.phase || "preset";
      closeModal();
      renderChain();
      previewChain(true);
      toast("已切到 " + r.recipe_id);
    }
  });
}
window.openRoundsModal = openRoundsModal;

/* ---------------------------------------------------------------- 预览 */

function debouncePreview() {
  clearTimeout(S.previewTimer);
  S.previewTimer = setTimeout(() => previewChain(false), 260);
}

async function previewChain(immediate) {
  if (!S.sampleId || !S.stages.length) return;
  if (immediate) clearTimeout(S.previewTimer);

  const token = ++S.previewToken;
  status("重算预览…");
  try {
    const d = await postJson(`/api/batches/${S.batchId}/samples/${S.sampleId}/preview`, {
      stages: S.stages,
    });
    if (token !== S.previewToken) return;   // 旧请求不覆盖新结果
    S.steps = d.steps;
    if (S.activeStep < 0 || S.activeStep >= S.steps.length) {
      S.activeStep = S.steps.length - 1;
    }
    updateThumbs();
    renderViewer();
    renderStepStrip();
    renderMetrics();
    renderAudit();
    status("预览已更新", `得分 ${d.final.score > 0 ? "+" : ""}${fmt(d.final.score)}`);
  } catch (e) {
    toast("预览失败：" + e.message, true);
    status("预览失败");
  }
}

function updateThumbs() {
  S.stages.forEach((st, i) => {
    const step = S.steps[i + 1];
    const card = document.querySelector(`[data-card="${i}"]`);
    if (!card || !step) return;
    let img = card.querySelector(".op-thumb");
    if (img && img.tagName === "IMG") {
      img.src = "data:image/jpeg;base64," + step.img;
    } else if (img) {
      const nw = document.createElement("img");
      nw.className = "op-thumb";
      nw.dataset.act = "thumb";
      nw.dataset.step = String(i + 1);
      img.replaceWith(nw);
      nw.src = "data:image/jpeg;base64," + step.img;
    }
  });
}

/* ---------------------------------------------------------------- 视图 */

function renderViewer() {
  const box = $("#viewer");
  if (!S.steps.length) {
    box.innerHTML = `<div class="empty-hint">暂无图像</div>`;
    return;
  }
  const cur = S.steps[Math.max(0, Math.min(S.activeStep, S.steps.length - 1))];
  const base = S.steps[0];

  const curName = cur.step === 0 ? "原图" : `第 ${cur.step} 步 · ${cur.label || cur.op}`;

  if (S.viewMode === "compare") {
    // 名称做圆角标签贴在各自图片的左上角。
    // 关键：标签要贴着"图片"而不是"格子"——contain 会在格子里留出灰边，
    // 贴着格子放就会飘进灰边里。所以外面套一层只包住图片的 figure。
    box.innerHTML = `<div class="compare-wrap">
      <div class="compare-cell">
        <span class="compare-tag">原图</span>
        <div class="compare-figure">
          <img src="data:image/jpeg;base64,${base.img}" alt=""
               data-zoom="原图" data-step="0" title="点击看大图">
        </div>
      </div>
      <div class="compare-divider"></div>
      <div class="compare-cell">
        <span class="compare-tag">${esc(cur.label || cur.op)}</span>
        <div class="compare-figure">
          <img src="data:image/jpeg;base64,${cur.img}" alt=""
               data-zoom="${esc(curName)}" data-step="${cur.step}" title="点击看大图">
        </div>
      </div>
    </div>`;
    $("#viewerLabel").textContent = "对比";
    const imgs = box.querySelectorAll(".compare-cell img");
    let left = imgs.length;
    imgs.forEach((im) => {
      if (im.complete) { left -= 1; }
      else im.addEventListener("load", () => { if (--left === 0) positionCompareTags(); });
    });
    if (left === 0) requestAnimationFrame(positionCompareTags);
  } else {
    box.innerHTML = `<img src="data:image/jpeg;base64,${cur.img}" alt=""
      data-zoom="${esc(curName)}" data-step="${cur.step}" title="点击看大图">`;
    $("#viewerLabel").textContent = curName;
  }
}

/* 大图预览 + 两台对比。
   逐节预览的图只有 420px，放大必然糊——所以先显示它（不等待），
   同时按原分辨率重算，回来再替换。对比模式下原图和当前步一起重算。 */
function lbSrc(step) {
  const st = S.steps[step];
  return st ? "data:image/jpeg;base64," + st.img : "";
}

function renderLightbox() {
  const lb = S.lb;
  if (!lb) return;
  const stage = $("#lightboxStage");
  const lowCls = (low) => (low ? "is-lowres" : "");

  if (!lb.pair) {
    stage.className = "lightbox-stage single";
    stage.innerHTML = `<img id="lbImg" class="${lowCls(lb.curLow)}" src="${lb.curSrc}" alt="">`;
    $("#lightboxLabel").textContent = `${lb.curLabel}${lb.curDim ? "　" + lb.curDim : ""}`;
  } else {
    stage.className = "lightbox-stage pair";
    stage.innerHTML = `
      <div class="lb-cell">
        <div class="lb-cap">原图${lb.baseDim ? "　" + lb.baseDim : ""}</div>
        <img class="${lowCls(lb.baseLow)}" src="${lb.baseSrc}" alt="">
      </div>
      <div class="lb-cell">
        <div class="lb-cap">${esc(lb.cmpLabel)}${lb.cmpDim ? "　" + lb.cmpDim : ""}</div>
        <img class="${lowCls(lb.cmpLow)}" src="${lb.cmpSrc}" alt="">
      </div>`;
    $("#lightboxLabel").textContent = `${lb.curLabel}${lb.curDim ? "　" + lb.curDim : ""}`;
  }

  const btn = $("#lightboxPairBtn");
  btn.hidden = !lb.canPair;
  btn.textContent = lb.pair ? "只看当前" : "两张对比";
}

function stepLabel(i) {
  const st = S.steps[i];
  if (!st) return `第 ${i} 步`;
  return st.step === 0 ? "原图" : `第 ${st.step} 步 · ${st.label || st.op}`;
}

async function fetchLightboxHD() {
  const lb = S.lb;
  if (!lb || !S.sampleId || !S.stages.length) return;
  const token = (S.lightboxToken = (S.lightboxToken || 0) + 1);

  const want = lb.pair ? [lb.curStep, 0, lb.cmpStep] : [lb.curStep];
  const uniq = [...new Set(want)];

  try {
    const res = await Promise.all(uniq.map((st) =>
      postJson(`/api/batches/${S.batchId}/samples/${S.sampleId}/full`,
        { stages: S.stages, step: st, max_side: 1600 })));
    if (token !== S.lightboxToken || $("#lightbox").hidden) return;

    res.forEach((d, i) => {
      const step = uniq[i];
      const src = "data:image/jpeg;base64," + d.img;
      const dim = `${d.w}×${d.h}`;
      if (step === lb.curStep) { lb.curSrc = src; lb.curLow = false; lb.curDim = dim; }
      if (step === 0) { lb.baseSrc = src; lb.baseLow = false; lb.baseDim = dim; }
      if (step === lb.cmpStep) { lb.cmpSrc = src; lb.cmpLow = false; lb.cmpDim = dim; }
    });
    renderLightbox();
  } catch (e) {
    if (!lb) return;
    lb.curLow = lb.baseLow = lb.cmpLow = false;   // 高清失败就保留缩略图，不打断
    renderLightbox();
  }
}

/* 打开灯箱：永远先给单图。
   要对比时再点右上角切换——大图首屏看的是"这一张的细节"，
   直接摊成并排会把图压掉一半。 */
async function openLightbox(src, label, step, pair) {
  const box = $("#lightbox");
  if (!box || !src) return;

  const cur = step == null ? S.steps.length - 1 : step;
  // 对比对象：当前步不是原图就用当前步，否则用主界面正在看的那一步
  const cmp = cur > 0 ? cur : (S.activeStep > 0 ? S.activeStep : cur);

  S.lb = {
    curStep: cur,
    curLabel: label || stepLabel(cur),
    curSrc: src, curLow: true, curDim: "",
    baseSrc: lbSrc(0), baseLow: true, baseDim: "",
    cmpStep: cmp,
    cmpLabel: stepLabel(cmp),
    cmpSrc: lbSrc(cmp), cmpLow: true, cmpDim: "",
    pair: !!pair,
    canPair: cmp > 0,
  };

  renderLightbox();
  box.hidden = false;
  document.body.classList.add("lightbox-open");
  fetchLightboxHD();
}
window.openLightbox = openLightbox;

function toggleLightboxPair() {
  if (!S.lb || !S.lb.canPair) return;
  S.lb.pair = !S.lb.pair;
  renderLightbox();
  if (S.lb.pair) fetchLightboxHD();
}
window.toggleLightboxPair = toggleLightboxPair;

function closeLightbox() {
  const box = $("#lightbox");
  if (!box || box.hidden) return;
  box.hidden = true;
  $("#lightboxStage").innerHTML = "";
  S.lb = null;
  document.body.classList.remove("lightbox-open");
}
window.closeLightbox = closeLightbox;

/* 把名称标签贴到"图片内容"的左上角，而不是格子的左上角。
   object-fit: contain 会在格子里留出灰边，纯 CSS 拿不到内容框，
   所以按图片实际宽高比算一次；窗口变化时重算。 */
function positionCompareTags() {
  document.querySelectorAll(".compare-cell").forEach((cell) => {
    const img = cell.querySelector("img");
    const tag = cell.querySelector(".compare-tag");
    if (!img || !tag || !img.naturalWidth) return;

    const fig = cell.querySelector(".compare-figure");
    const r = img.getBoundingClientRect();
    const c = cell.getBoundingClientRect();
    if (!fig || !r.width) return;

    // 把 figure 收到图片内容的大小
    const scale = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
    const w = Math.round(img.naturalWidth * scale);
    const h = Math.round(img.naturalHeight * scale);
    fig.style.width = w + "px";
    fig.style.height = h + "px";

    // 标签在图片左侧外，垂直与图片顶边对齐（略低 8px）
    const figTop = fig.getBoundingClientRect().top - c.top;
    const figLeft = fig.getBoundingClientRect().left - c.left;
    tag.style.top = Math.round(figTop + 8) + "px";
    tag.style.left = Math.max(4, Math.round(figLeft - tag.offsetWidth - 12)) + "px";
  });
}

function renderStepStrip() {
  const strip = $("#stepStrip");
  if (!S.steps.length) {
    strip.innerHTML = "";
    return;
  }
  strip.innerHTML = S.steps.map((s, i) => `
    <div class="step-cell ${i === S.activeStep ? "active" : ""}" data-step-idx="${i}" title="${esc(s.label || s.op)}">
      <img src="data:image/jpeg;base64,${s.img}" alt="">
      <span>${s.step === 0 ? "原图" : String(s.step).padStart(2, "0") + " " + esc((s.op || "").slice(0, 7))}</span>
    </div>`).join("");
}

/* ---------------------------------------------------------------- 指标 */

// neutral: 越高越好 / 越低越好 / 无方向
// 平均亮度是"合适才好"，调亮调暗都可能对，所以不给红绿；
// 截断率反过来，越低越好，单独按阈值判。
const METRIC_DEFS = [
  { key: "local_contrast", name: "局部对比度", dir: "high" },
  { key: "entropy", name: "直方图熵", dir: "high" },
  { key: "grad_energy", name: "梯度能量", dir: "high" },
  { key: "range_used", name: "动态范围", dir: "high" },
  { key: "mean", name: "平均亮度", dir: "none" },
  { key: "clip", name: "截断率", dir: "low" },
];

function metricsHTML(step) {
  if (!step || !step.delta) return `<div class="empty-hint">—</div>`;

  const rows = METRIC_DEFS.map((def) => {
    const d = step.delta[def.key] || {};
    const dp = d.delta_pct;
    let cls = "", width = 0, txt = "—";

    if (def.dir === "low") {
      const v = (d.value || 0) * 100;
      cls = v > 10 ? "neg" : v > 5 ? "" : "pos";
      width = Math.min(100, v * 5);
      txt = v.toFixed(1) + "%";
    } else if (def.dir === "none" || dp === null || dp === undefined) {
      cls = "neu";
      txt = def.dir === "none" && dp !== null && dp !== undefined ? pct(dp) : fmt(d.value);
    } else {
      cls = dp > 1 ? "pos" : dp < -1 ? "neg" : "neu";
      width = Math.min(100, Math.abs(dp));
      txt = pct(dp);
    }

    return `<div class="metric-row">
      <span class="metric-name">${def.name}</span>
      <span class="metric-bar"><i class="${cls}" style="width:${width}%"></i></span>
      <span class="metric-val ${cls === "neu" ? "" : cls}">${txt}</span>
    </div>`;
  }).join("");

  const score = step.score;
  const scoreCls = score > 0.02 ? "pos" : score < -0.02 ? "neg" : "";
  return `<div class="score-row">
      <span class="score-label">综合得分</span>
      <span class="score-value ${scoreCls}">${score > 0 ? "+" : ""}${fmt(score)}</span>
      <span class="score-unit">${S.chainMs ? `纯算子 ${Math.round(S.chainMs)} ms` : "相对原图"}</span>
    </div>${rows}`;
}

function renderMetrics() {
  const step = S.steps[Math.max(0, Math.min(S.activeStep, S.steps.length - 1))];
  S.metricsCache = (step && step.delta) ? step : null;
  syncEntryButtons();
}

function openMetricsModal() {
  const step = S.metricsCache;
  if (!step) return toast("还没有可显示的指标", true);
  const at = step.step === 0 ? "原图" : `第 ${step.step} 步 · ${step.label || step.op}`;
  openModal(`指标　${at}`,
    `<div class="modal-table">${metricsHTML(step)}</div>
     <p class="muted" style="margin:12px 0 0;font-size:13px">
       所有数值相对原图计算。正向提升为绿，下降为红，无方向性的指标（平均亮度）用中性色。</p>`,
    `<button onclick="closeModal()">关闭</button>`);
}
window.openMetricsModal = openMetricsModal;

function auditHTML(step) {
  if (!step || !step.audit) return `<div class="empty-hint">—</div>`;
  const a = step.audit;
  const map = { pass: ["通过", "ok"], warn: ["告警", "warn"], reject: ["拒绝", "danger"] };
  const [cn, cls] = map[a.verdict] || ["—", ""];
  const riskPct = Math.round((a.risk || 0) * 100);
  const riskLv = a.verdict === "reject" ? "lv-bad" : a.verdict === "warn" ? "lv-warn" : "lv-ok";

  let html = `<div class="audit-head">
      <span class="badge ${cls}">${cn}</span>
      <span class="muted small">风险</span>
      <span class="risk-bar"><i class="${riskLv}" style="width:${riskPct}%"></i></span>
      <span class="metric-val">${riskPct}</span>
    </div>`;

  html += `<div class="audit-detail open">
      <span>截断 ${((a.clip_abs || 0) * 100).toFixed(1)}%</span>
      <span>噪声 σ${((a.noise_abs || 0) * 255).toFixed(1)} / ${fmt(a.noise_ratio, 1)}×</span>
      <span>台阶 ${a.banding_delta != null ? (a.banding_delta > 0 ? "+" : "") + fmt(a.banding_delta, 2) : "—"}</span>
    </div>`;

  if (a.reasons && a.reasons.length) {
    html += a.reasons.map((r) => `<div class="reason ${r.level}">${esc(r.text)}</div>`).join("");
  } else {
    html += `<div class="reason">未触发任何闸门，可直接用于流程配置。</div>`;
  }
  return html;
}

function renderAudit() {
  const step = S.steps[Math.max(0, Math.min(S.activeStep, S.steps.length - 1))];
  S.auditCache = (step && step.audit) ? step : null;
  syncEntryButtons();
}

function openAuditModal() {
  const step = S.auditCache;
  if (!step) return toast("还没有可审计的结果", true);
  const v = step.audit.verdict;
  const cn = { pass: "通过", warn: "告警", reject: "拒绝" }[v] || "—";
  openModal(`伪影审计　${cn}`,
    `<div class="modal-table">${auditHTML(step)}</div>
     <p class="muted" style="margin:12px 0 0;font-size:13px">
       闸门判据：截断率 &gt;10%，或噪声 σ 超 7.6/255 且放大 &gt;2.5× 即判拒绝——
       这类链会把噪声标成缺陷，配进流程后会误检。</p>`,
    `<button onclick="closeModal()">关闭</button>`);
}
window.openAuditModal = openAuditModal;

/* 三个入口按钮上直接带关键值，不点开也知道结论 */
function syncEntryButtons() {
  const rb = $("#roundsBtn");
  if (rb) {
    const n = S.roundsCache ? S.roundsCache.rounds.length : 0;
    rb.textContent = n ? `迭代记录 ${n}` : "迭代记录";
    rb.disabled = !n;
  }

  const mb = $("#metricsBtn");
  if (mb) {
    const st = S.metricsCache;
    mb.textContent = st ? `指标 ${st.score > 0 ? "+" : ""}${fmt(st.score)}` : "指标";
    mb.disabled = !st;
  }

  const ab = $("#auditBtn");
  if (ab) {
    const st = S.auditCache;
    const v = st && st.audit ? st.audit.verdict : null;
    const cn = { pass: "通过", warn: "告警", reject: "拒绝" }[v];
    ab.textContent = cn ? `伪影 ${cn}` : "伪影审计";
    ab.className = "ghost" + (v === "reject" ? " is-bad" : v === "warn" ? " is-warn" : "");
    ab.disabled = !st;
  }
}

/* ---------------------------------------------------------------- 编辑动作 */

function markEdited(kind, op) {
  S.phase = "preset";   // 人动过之后就不再是"AI 自拟"了，徽章要诚实
  saveStages(kind, op);
}

async function saveStages(kind, op, note) {
  if (!S.sampleId) return;
  try {
    await postJson(`/api/batches/${S.batchId}/samples/${S.sampleId}/stages`, {
      stages: S.stages,
      kind: kind || "param",
      op: op || null,
      recipe_id: S.sample ? S.sample.selected_recipe : null,
      note: note || "",
    });
    S.sample.edited = true;
    S.sample.edit_count = (S.sample.edit_count || 0) + 1;
    renderSampleList();
  } catch (e) {
    toast("保存失败：" + e.message, true);
  }
}

function moveStage(i, dir) {
  const j = i + dir;
  if (j < 0 || j >= S.stages.length) return;
  const t = S.stages[i];
  S.stages[i] = S.stages[j];
  S.stages[j] = t;
  renderChain();
  markEdited("move", S.stages[j].op);
  previewChain(true);
}

function deleteStage(i) {
  const op = S.stages[i].op;
  S.stages.splice(i, 1);
  S.activeStep = Math.min(S.activeStep, S.stages.length);
  renderChain();
  markEdited("remove", op);
  previewChain(true);
}

async function addStage(op) {
  const d = await api(`/api/op_default?op=${encodeURIComponent(op)}`);
  S.stages.push({ op, p: d.params });
  renderChain();
  markEdited("add", op);
  await previewChain(true);
  toast(`已添加 ${op}`);
}

function applyRecipeById(rid) {
  const r = S.schema.recipes.find((x) => x.id === rid);
  if (!r) return;
  S.stages = JSON.parse(JSON.stringify(r.stages));
  renderChain();
  markEdited("reset", null, "套用配方 " + rid);
  previewChain(true);
}

/* ---------------------------------------------------------------- 核验 */

async function acceptCurrent() {
  if (!S.sampleId || !S.stages.length) {
    toast("没有可提交的算子链", true);
    return;
  }
  if (S.pending) return;
  S.pending = true;
  const btn = $("#acceptBtn");
  if (btn) btn.disabled = true;
  try {
    await postJson(`/api/batches/${S.batchId}/samples/${S.sampleId}/accept`, {
      stages: S.stages,
      recipe_id: S.sample ? S.sample.selected_recipe : null,
    });
    toast("已通过核验，这张图定稿");
    await refreshSampleFlags();
    status("已通过核验", S.sample ? S.sample.filename : "");
  } catch (e) {
    toast("提交失败：" + e.message, true);
  } finally {
    S.pending = false;
    if (btn) btn.disabled = false;
  }
}

/* 候选切换：底部 1/2/3 直接跳到某一条已算好的方案。
   不重新算、不调模型，就是在这批候选里选。 */
function candidateList() {
  const rounds = (S.analysis && S.analysis.rounds) ||
                 (S.sample && S.sample.rounds) || [];
  const seen = new Set();
  const out = [];
  for (const r of rounds) {
    if (!r || !r.ok || !r.stages || !r.stages.length) continue;
    if (seen.has(r.recipe_id)) continue;
    seen.add(r.recipe_id);
    out.push(r);
  }
  return out.slice(0, 5);
}

const VERDICT_CN = { pass: "可过", warn: "告警", reject: "拒绝" };

function renderCandidateSwitcher() {
  const box = $("#candSwitcher");
  if (!box) return;

  const cands = candidateList();
  if (!cands.length) {
    box.innerHTML = `<span class="muted small">分析后可切换候选</span>`;
    return;
  }

  const cur = S.sample ? S.sample.selected_recipe : null;
  box.innerHTML = cands.map((c, i) => {
    const v = (c.audit || {}).verdict || "pass";
    const active = c.recipe_id === cur ? "active" : "";
    const ops = (c.stages || []).map((x) => x.op).join("→");
    const tip = `候选 ${i + 1}：${c.recipe_id}
得分 ${c.score > 0 ? "+" : ""}${fmt(c.score, 3)}　${VERDICT_CN[v] || v}
${ops}`;
    return `<button class="cand-btn ${active}" data-rid="${esc(c.recipe_id)}"
      title="${esc(tip)}">${i + 1}<i class="cand-dot ${v}"></i></button>`;
  }).join("");
}

async function switchCandidate(rid) {
  if (!S.sampleId || S.pending) return;
  const cands = candidateList();
  const idx = cands.findIndex((c) => c.recipe_id === rid);
  if (idx < 0) return;

  S.pending = true;
  try {
    const d = await postJson(`/api/batches/${S.batchId}/samples/${S.sampleId}/switch`,
      { recipe_id: rid });
    S.stages = JSON.parse(JSON.stringify(d.stages || []));
    S.aiStages = JSON.parse(JSON.stringify(d.stages || []));
    S.phase = d.phase || "preset";
    S.rationale = "";
    renderChain();
    await previewChain(true);
    await refreshSampleFlags();
    renderCandidateSwitcher();
    if (d.warning) {
      // 切到被闸门拒绝的链：切换成功，但必须当场告诉人，别等导出时才发现
      toast(d.warning, true);
    } else {
      toast(d.accepted_cleared
        ? `已切到候选 ${idx + 1}，原核验标记已撤销`
        : `已切到候选 ${idx + 1}：${rid}`);
    }
  } catch (e) {
    toast("切换失败：" + e.message, true);
  } finally {
    S.pending = false;
  }
}

/* ---------------------------------------------------------------- 批量 */

async function runAll() {
  if (!S.batchId) return;
  const n = (S.batch.samples || []).length;
  if (!n) return toast("批次为空", true);

  openModal("整批分析", `
    <p style="margin-top:0;color:var(--text-2)">
      将对 <b>${n}</b> 张图逐张诊断并生成方案。生成候选链要等模型返回，是耗时大头。
    </p>
    <div class="field">
      <label>分析模式</label>
      <select id="runMode">
        <option value="preset">最简 · 只查表（约 1 秒/张，${Math.ceil(n * 1)} 秒）</option>
        <option value="fast">快速 · 查表 + 参数精修（约 2 秒/张，${Math.ceil(n * 2)} 秒）</option>
        <option value="std" selected>标准 · 含 AI 拼链（约 20 秒/张，约 ${Math.ceil(n * 20 / 60)} 分钟）</option>
      </select>
    </div>
    <div class="reason" id="runTip">
      标准模式每张会额外调一次模型自己拼链，遇到噪声超标还会重试一次。
      赶时间就选「快速」——方案质量差距不大，但快十几倍。
    </div>`,
    `<button onclick="closeModal()">取消</button>
     <button class="primary" onclick="doRunAll()">开始</button>`);

  $("#runMode").addEventListener("change", (e) => {
    const tip = $("#runTip");
    if (!tip) return;
    tip.textContent = {
      preset: "只走配方库，不调模型。适合先粗筛一遍看诊断对不对。",
      fast: "查表后会做参数扫描，能把最优链再打磨一点，全程不调模型。",
      std: "每张会多调一次模型拼链；被噪声拒掉时会带反馈再试一次。耗时主要在这里。",
    }[e.target.value];
  });
}

async function doRunAll() {
  const mode = $("#runMode").value;
  const opt = {
    preset: { compose: false, tune: false, label: "只查表" },
    fast: { compose: false, tune: true, label: "查表+精修" },
    std: { compose: true, tune: true, label: "查表+拼链+精修" },
  }[mode];

  closeModal();
  const n = (S.batch.samples || []).length;
  status(`整批分析中（${opt.label}）…`, `${n} 张`);

  try {
    const d = await postJson(`/api/batches/${S.batchId}/run_all`,
      { compose: opt.compose, tune: opt.tune });
    toast(`整批完成：${d.count} 张`);
    await loadBatchRefresh();
    const composed = (d.results || []).filter((r) => (r.recipe_id || "").startsWith("ai_composed")).length;
    status(`整批完成（${opt.label}）`,
      `${d.count} 张${composed ? ` · ${composed} 张采用 AI 自拟` : ""}`);
  } catch (e) {
    toast("整批失败：" + e.message, true);
    status("整批失败");
  }
}
window.doRunAll = doRunAll;

async function loadBatchRefresh() {
  const d = await api(`/api/batches/${S.batchId}`);
  S.batch = d.batch;
  renderSampleList();
  if (S.sampleId) {
    S.sample = (S.batch.samples || []).find((x) => x.id === S.sampleId) || S.sample;
  }
}

/* 导出前先问一句：未核验的样本要不要一起交出去。
   流程是要配进产线的，把没确认过的链混进去是实打实的风险。 */
async function exportBatch() {
  if (!S.batchId) return;
  let st, agg;
  try {
    const [a, b] = await Promise.all([
      api(`/api/batches/${S.batchId}/stats`),
      postJson(`/api/batches/${S.batchId}/aggregate`, {}),
    ]);
    st = a;
    agg = (b.aggregate || {});
  } catch (e) {
    return toast("读取批次状态失败：" + e.message, true);
  }

  const total = st.total || 0;
  const acc = st.accepted || 0;

  // 这条包里还有多少算子在海康里没有原生模块
  const SCRIPT_OPS = ["retinex", "homomorphic", "bandpass"];
  const scriptCount = (S.batch.samples || []).reduce(
    (n, s) => n + (s.stages || []).filter((x) => SCRIPT_OPS.includes(x.op)).length, 0);

  const groups = agg.groups || [];
  const needManual = groups.filter((g) => g.needs_manual);
  const unanalyzed = Math.max(0, (st.total || 0) - (st.analyzed || 0));

  const groupRows = groups.map((g) => `<tr>
      <td class="txt">${esc(g.diagnosis_label)}</td>
      <td>${esc(g.recipe_id)}</td>
      <td>${g.member_count}</td>
      <td>${Math.round(g.consistency * 100)}%</td>
      <td class="txt">${g.needs_manual ? "需人工确认" : "可直接用"}</td>
    </tr>`).join("");

  const groupBox = groups.length
    ? `<div class="group-summary">
         <div class="group-line">
           本批会分成 <b>${groups.length}</b> 组方案，
           ${needManual.length
             ? `<span class="warn-text">${needManual.length} 组组内一致率低于 80%，需人工确认</span>`
             : "组内一致率均达标"}
           ${unanalyzed ? `<span class="muted small">（${unanalyzed} 张尚未分析，分组结果仅供参考）</span>` : ""}
         </div>
         <div class="group-table">
           <table class="grid">
             <thead><tr><th>成像问题</th><th>处理链</th><th>样本</th><th>一致性</th><th>结论</th></tr></thead>
             <tbody>${groupRows}</tbody>
           </table>
         </div>
       </div>`
    : "";

  const jsonBox = `<label class="check-line">
      <input type="checkbox" id="withJson">
      <span>同时导出机器可读配置
        <em class="muted small">（pipeline.json / manifest.json / edits.jsonl，
        给脚本做批量生成或回归核对用；手工配流程不需要）</em>
      </span>
    </label>`;

  const nativeBox = scriptCount
    ? `<label class="check-line">
         <input type="checkbox" id="onlyNative" checked>
         <span>全部转成海康原生模块
           <em class="muted small">（${scriptCount} 处脚本算子将换成原生组合，替换明细写进配置单）</em>
         </span>
       </label>`
    : `<div class="reason" style="margin-top:10px">
         这批链里没有脚本算子，直接按参数配即可。</div>`;

  if (!acc) {
    openModal("导出流程配置", `
      <p style="margin-top:0">这批 <b>${total}</b> 张图还没有一张通过核验。</p>
      <div class="reason warn">
        核验是「这条链我确认过了」的凭据。没核验过的方案，导进产线等于没经过人工把关。
        可以先去逐张点「通过核验」，也可以现在就全部导出。
      </div>
      ${groupBox}
      ${nativeBox}
      ${jsonBox}`,
      `<button onclick="closeModal()">去核验</button>
       <button class="primary" onclick="doExport(false)">全部导出（${total} 张）</button>`, "wide");
    return;
  }

  openModal("导出流程配置", `
    <p style="margin-top:0">
      共 <b>${total}</b> 张，其中 <b>${acc}</b> 张已通过人工核验，${total - acc} 张未核验。
    </p>
    <table class="grid" style="margin-top:4px">
      <thead><tr><th>选项</th><th>导出的内容</th></tr></thead>
      <tbody>
        <tr><td>只导已核验</td>
          <td class="txt">${acc} 张已确认的图，交给流程搭建最稳妥（推荐）</td></tr>
        <tr><td>全部导出</td>
          <td class="txt">${total} 张图，含 ${total - acc} 张未确认的链，仅适合内部先看效果</td></tr>
      </tbody>
    </table>
    ${groupBox}
    ${nativeBox}
    ${jsonBox}`,
    `<button onclick="closeModal()">取消</button>
     <button class="primary" onclick="doExport(true)">只导已核验（${acc} 张）</button>
     <button onclick="doExport(false)">全部导出（${total} 张）</button>`);
}
window.exportBatch = exportBatch;

async function doExport(onlyAccepted) {
  const cb = $("#onlyNative");
  const onlyNative = cb ? cb.checked : false;
  const jb = $("#withJson");
  const withJson = jb ? jb.checked : false;
  closeModal();
  status("生成导出包…", onlyAccepted ? "仅已核验" : "全部样本");
  try {
    const d = await postJson(`/api/batches/${S.batchId}/export`,
      { only_accepted: onlyAccepted, only_native: onlyNative, with_json: withJson });

    const warn = d.script_nodes > 0
      ? `<div class="reason warn" style="margin-bottom:12px">
           ${d.script_nodes} 个节点在 VisionMaster 里没有原生模块，必须用脚本模块实现，
           <code>流程配置.md</code> 里已附参考代码。</div>`
      : `<div class="reason" style="margin-bottom:12px">
           所有节点在 VisionMaster 里都有原生模块，可直接按参数配置。</div>`;

    const nativeNote = d.only_native
      ? (d.native_swaps
        ? `<div class="reason" style="margin-bottom:12px">
             已把 ${d.native_swaps_items} 张图上的脚本算子换成原生组合
             （涉及 ${d.native_swaps} 条方案）。<b>换的是近似</b>，
             配完请拿 <code>compare/</code> 的对照图核一次。</div>`
        : `<div class="reason" style="margin-bottom:12px">
             这批链本来就没有脚本算子，未做替换。</div>`)
      : "";

    const skipped = d.skipped_unverified
      ? `<div class="reason warn" style="margin-bottom:12px">
           ${d.skipped_unverified} 张未核验的样本没有导出。核验后再导一次即可补齐。</div>`
      : "";

    openModal("导出流程配置", `
      ${warn}
      ${nativeNote}
      ${skipped}
      <p style="margin-top:0">
        ${d.pipelines} 条流程方案，${d.items} 张调试样本${d.only_accepted ? "（全部已核验）" : `（其中 ${d.accepted} 张已核验）`}。
        最慢一条链实测 <b>${d.slowest_ms} ms</b>（按样本分辨率）。
      </p>
      <table class="grid" style="margin-top:8px">
        <thead><tr><th>文件</th><th>用途</th></tr></thead>
        <tbody>
          <tr><td>流程配置.md</td><td class="txt">打开这个：节点顺序、参数、VM 模块、多分辨率耗时、核验状态</td></tr>
          <tr><td>compare/</td><td class="txt">原图\|增强图并排，配完对照核对</td></tr>
          <tr><td>debug_samples/</td><td class="txt">增强图本身，可直接导入 VM 试跑</td></tr>
          ${d.with_json ? `<tr><td>pipeline.json</td><td class="txt">机读配置单（给脚本）</td></tr>
          <tr><td>manifest.json</td><td class="txt">逐张对照索引（给脚本）</td></tr>
          <tr><td>edits.jsonl</td><td class="txt">改动流水（给脚本）</td></tr>` : ""}
        </tbody>
      </table>`,
      `<a href="${d.download}" download><button class="primary">下载 zip</button></a>
       <button onclick="closeModal()">关闭</button>`, "wide");
    status("导出完成", `${d.pipelines} 条方案 · ${d.items} 张样本 · ${d.elapsed_ms} ms`);
  } catch (e) {
    toast("导出失败：" + e.message, true);
    status("导出失败");
  }
}
window.doExport = doExport;

/* ---------------------------------------------------------------- 弹窗 */

function openModal(title, bodyHTML, footHTML, extraCls) {
  const root = $("#modalRoot");
  root.innerHTML = `<div class="modal-mask" onclick="if(event.target===this)closeModal()">
    <div class="modal ${extraCls || ""}">
      <div class="modal-head"><span>${esc(title)}</span>
        <button class="close" onclick="closeModal()">×</button></div>
      <div class="modal-body">${bodyHTML}</div>
      <div class="modal-foot">${footHTML || '<button onclick="closeModal()">关闭</button>'}</div>
    </div>
  </div>`;
  root.style.pointerEvents = "auto";
}

function closeModal() {
  $("#modalRoot").innerHTML = "";
  $("#modalRoot").style.pointerEvents = "none";
}
window.closeModal = closeModal;

/* 折叠开关：只影响当前渲染，不写库 —— 这是看的方式，不是数据 */
function toggleRounds() {
  S.roundsOpen = !S.roundsOpen;
  if (S.analysis) renderRounds(S.analysis.rounds || [], S.analysis.stop_reason);
  else if (S.sample) renderRounds(S.sample.rounds || [], "");
}
window.toggleRounds = toggleRounds;

function toggleMetrics() {
  S.metricsOpen = !S.metricsOpen;
  renderMetrics();
}
window.toggleMetrics = toggleMetrics;

function toggleObs() {
  const long = S.analysis && S.analysis.diagnosis &&
    (S.analysis.diagnosis.observation || "").length > 40;
  if (!long && !(S.sample && S.sample.diagnosis &&
      (S.sample.diagnosis.observation || "").length > 40)) return;
  S.obsOpen = !S.obsOpen;
  const diag = (S.analysis && S.analysis.diagnosis) || (S.sample && S.sample.diagnosis);
  renderDiagnosis(diag, S.analysis);
}
window.toggleObs = toggleObs;

function toggleAudit() {
  S.auditOpen = !S.auditOpen;
  renderAudit();
}
window.toggleAudit = toggleAudit;

function openOpPicker() {
  const groups = S.schema.groups;
  const byGroup = groups.map((g) => {
    const items = S.schema.ops.filter((o) => o.group === g).map((o) =>
      `<button class="op-choice" data-op="${esc(o.op)}"><b>${esc(o.label)}</b><span>${esc(o.op)} · ${esc(o.hint)}</span></button>`
    ).join("");
    return `<div class="group-label">${esc(g)}</div>${items}`;
  }).join("");

  openModal("添加算子", `<div class="op-picker" id="opPicker">${groups ? byGroup : "无算子"}</div>`,
    `<button onclick="closeModal()">取消</button>`);

  $("#opPicker").addEventListener("click", async (e) => {
    const btn = e.target.closest(".op-choice");
    if (!btn) return;
    const op = btn.dataset.op;
    closeModal();
    await addStage(op);
  });
}

function openSettings() {
  api("/api/config").then((c) => {
    const presets = c.presets || {};
    const cur = c.preset || "deepseek";
    const opts = Object.entries(presets).map(([k, v]) =>
      `<option value="${k}" ${k === cur ? "selected" : ""}>${esc(v.label)}</option>`).join("");
    const note = presets[cur] ? presets[cur].note : "";

    const isRules = c.provider === "rules";
    openModal("视觉模型配置", `
      ${isRules ? `<div class="reason warn" id="cfgBanner" style="margin-bottom:12px">
        当前「诊断来源」是<b>本地规则</b>，不调用任何模型，所以「测试连通」不适用。
        <button class="tiny-btn" style="margin-left:8px" onclick="switchToCloud()">改为线上模型</button>
      </div>` : ""}

      <div class="field">
        <label>服务商预设</label>
        <select id="cfgPreset">${opts}</select>
        <div class="muted small" id="presetNote" style="margin-top:5px">${esc(note)}</div>
      </div>

      <div class="field">
        <label>诊断来源</label>
        <select id="cfgProvider">
          <option value="rules" ${c.provider === "rules" ? "selected" : ""}>本地规则</option>
          <option value="cloud" ${c.provider === "cloud" ? "selected" : ""}>线上模型</option>
          <option value="local" ${c.provider === "local" ? "selected" : ""}>本地模型</option>
        </select>
      </div>

      <div class="field">
        <label>API Key ${c.api_key_masked ? "（当前：" + esc(c.api_key_masked) + "）" : "（未设置）"}</label>
        <input type="password" id="cfgKey" placeholder="留空则不修改">
      </div>

      <div class="field"><label>Base URL</label>
        <input type="text" id="cfgBase" value="${esc(c.base_url || "")}"></div>

      <div class="field"><label>模型名</label>
        <input type="text" id="cfgModel" value="${esc(c.model || "")}"></div>

      <div class="field">
        <label>图片精度</label>
        <select id="cfgDetail">
          <option value="low" ${c.detail === "low" ? "selected" : ""}>low · 缩到 512×512（省 token，诊断成像条件够用）</option>
          <option value="original" ${c.detail === "original" ? "selected" : ""}>original · 原尺寸（更贵，需要看清细节时用）</option>
          <option value="" ${!c.detail ? "selected" : ""}>不传（部分厂商不支持该字段）</option>
        </select>
      </div>

      <div class="field">
        <label>图片传参格式</label>
        <select id="cfgStyle">
          <option value="data_url" ${c.image_style === "data_url" ? "selected" : ""}>data URL · data:image/jpeg;base64,…（OpenAI / DeepSeek / 百炼 / vLLM）</option>
          <option value="raw_base64" ${c.image_style === "raw_base64" ? "selected" : ""}>裸 base64 · 不带前缀（智谱 GLM）</option>
        </select>
        <div class="muted small" style="margin-top:5px">跟预设自动切换；只有接非标准服务时才需要手动改</div>
      </div>

      <div class="field"><label>超时（秒）</label>
        <input type="number" id="cfgTimeout" value="${c.timeout || 60}"></div>

      <div class="reason" style="margin-top:6px">
        模型<b>只做看图诊断</b>，不产生任何处理参数。参数永远来自配方库，保证可复现。
      </div>`,
      `<button onclick="testVision()">测试连通</button>
       <button class="primary" onclick="saveSettings()">保存</button>`);

    // 切预设时自动填地址和模型，省得人记 base_url
    $("#cfgPreset").addEventListener("change", (e) => {
      const p = presets[e.target.value];
      if (!p) return;
      $("#cfgBase").value = p.base_url;
      $("#cfgModel").value = p.model;
      $("#cfgStyle").value = p.image_style;
      $("#cfgDetail").value = p.detail;
      $("#presetNote").textContent = p.note || "";
    });

    // 切「诊断来源」时也要管地址：选了本地模型却留着公网 base_url，
    // 请求照样打到线上，界面却写着"本地模型"——演示时最容易被问穿。
    // 只在"地址还是空 / 仍是某个公网预设"时替换，不动用户手填的内网地址。
    $("#cfgProvider").addEventListener("change", (e) => {
      const v = e.target.value;
      const base = $("#cfgBase").value.trim();
      const cloudBases = Object.values(presets)
        .map((p) => p.base_url || "")
        .filter((u) => u && !/127\.0\.0\.1|localhost/.test(u));
      const isCloudBase = !base || cloudBases.some((u) => base.startsWith(u));
      if (v === "local" && isCloudBase && presets.local) {
        $("#cfgBase").value = presets.local.base_url;
        $("#cfgModel").value = presets.local.model;
        $("#cfgStyle").value = presets.local.image_style;
        $("#presetNote").textContent = presets.local.note || "";
        toast("已把地址换成本地模型默认值；内网 vLLM 请改成实际地址");
      } else if (v === "cloud" && /127\.0\.0\.1|localhost/.test(base) && presets.deepseek) {
        const p = presets[$("#cfgPreset").value] || presets.deepseek;
        $("#cfgBase").value = p.base_url;
        $("#cfgModel").value = p.model;
      }
    });
  });
}
window.openSettings = openSettings;

async function saveSettings() {
  const body = {
    preset: $("#cfgPreset").value,
    provider: $("#cfgProvider").value,
    base_url: $("#cfgBase").value.trim(),
    model: $("#cfgModel").value.trim(),
    detail: $("#cfgDetail").value,
    image_style: $("#cfgStyle").value,
    timeout: Number($("#cfgTimeout").value) || 60,
  };
  const k = $("#cfgKey").value.trim();
  if (k) body.api_key = k;
  try {
    await postJson("/api/config", body);
    toast("配置已保存");
    closeModal();
    await loadHealth();
  } catch (e) {
    toast("保存失败：" + e.message, true);
  }
}
window.saveSettings = saveSettings;

function switchToCloud() {
  const p = $("#cfgProvider");
  if (p) p.value = "cloud";
  const banner = $("#cfgBanner");
  if (banner) {
    banner.className = "reason";
    banner.innerHTML = "已改为「线上模型」。填好 API Key 后点「测试连通」，测试通过再点保存。";
  }
}
window.switchToCloud = switchToCloud;

async function testVision() {
  // 用表单当前值测，不用已保存的配置 —— 否则改了没保存就测，测的是旧的
  const body = {
    preset: $("#cfgPreset").value,
    provider: $("#cfgProvider").value,
    base_url: $("#cfgBase").value.trim(),
    model: $("#cfgModel").value.trim(),
    detail: $("#cfgDetail").value,
    image_style: $("#cfgStyle").value,
    timeout: Number($("#cfgTimeout").value) || 60,
  };
  const k = $("#cfgKey").value.trim();
  if (k) body.api_key = k;

  const bd = $("#modalRoot .modal-body");
  if (body.provider === "cloud" && !k) {
    const old = $("#testResult");
    if (old) old.remove();
    if (bd) bd.insertAdjacentHTML("afterbegin", `<div class="reason reject" id="testResult" style="margin-bottom:12px">
      还没填 API Key。粘贴 sk- 开头的 key 再点测试。</div>`);
    toast("先填 API Key", true);
    return;
  }

  toast("测试中…");

  try {
    const d = await postJson("/api/test_vision", body);
    if (!bd) return;

    const old = $("#testResult");
    if (old) old.remove();

    if (d.ok) {
      const tags = (d.diagnoses || []).join("、") || "—";
      const warn = d.warning
        ? `<br><b style="color:var(--warn)">注意：</b>${esc(d.warning)}`
        : "";
      bd.insertAdjacentHTML("afterbegin", `<div class="reason" id="testResult"
        style="border-left-color:var(--ok);background:rgba(53,199,160,.07);margin-bottom:12px">
        连通成功：<b>${esc(d.model)}</b> 于 ${esc(d.base_url)} 返回「${esc(tags)}」${warn}<br>
        <span class="muted small">${esc(d.observation || "")}　耗时 ${d.elapsed_ms} ms　
        别忘了点右下角「保存」</span></div>`);
      if (d.warning) toast("连通了，但地址和诊断来源不一致", true);
      else toast(`连通正常 · ${d.model} · ${d.elapsed_ms} ms`);
    } else {
      const hint = d.hint === "provider=rules"
        ? `<button class="tiny-btn" style="margin-top:8px" onclick="switchToCloud()">改为线上模型</button>`
        : "";
      bd.insertAdjacentHTML("afterbegin", `<div class="reason reject" id="testResult" style="margin-bottom:12px">
        ${esc(d.reason || "未知原因")}${hint}</div>`);
      toast("测试未通过", true);
    }
  } catch (e) {
    toast("失败：" + e.message, true);
    if (bd) bd.insertAdjacentHTML("afterbegin",
      `<div class="reason reject" id="testResult" style="margin-bottom:12px">${esc(e.message)}</div>`);
  }
}
window.testVision = testVision;

async function openPreflight() {
  if (!S.batchId) return toast("先选一个批次", true);
  openModal("算子准入表", `<div class="empty-hint"><span class="spin"></span> 正在跑 11 条算子链 × 4 张样本，约 15–25 秒…<br>
    <span class="small muted">这一步不调模型，纯算子计算，结果可重复</span></div>`,
    `<button onclick="closeModal()">关闭</button>`);
  try {
    const d = await postJson("/api/preflight", { batch_id: S.batchId, limit: 4, max_side: 900 });
    const body = $("#modalRoot .modal-body");
    if (!body) return;   // 用户中途关掉了弹窗，别炸
    const rows = d.rows.map((r) => `<tr>
      <td>${esc(r.recipe_id)}</td>
      <td class="txt">${esc(r.desc)}</td>
      <td>${(r.reject_rate * 100).toFixed(0)}%</td>
      <td>${r.mean_clip * 100 < 0.01 ? "0.0" : (r.mean_clip * 100).toFixed(1)}%</td>
      <td>σ${(r.mean_noise_abs * 255).toFixed(1)}</td>
      <td>${r.mean_score > 0 ? "+" : ""}${r.mean_score.toFixed(3)}</td>
      <td class="txt">${r.admitted ? '<span class="badge ok">准入</span>' : '<span class="badge danger">禁用</span>'}</td>
    </tr>`).join("");
    body.innerHTML = `
      <p style="margin-top:0;color:var(--text-2)">
        这张表回答一个问题：<b>每条算子链处理出来的图，还能不能用来标注、能不能照着配产线。</b>
        它把配方库里的链在几张样本上各跑一遍，统计被伪影闸门拒绝的比例——拒绝率高的链会把噪声、
        量化台阶放大成假缺陷，标进去就是错标签，所以宁可先禁用。
        全程不调模型、不需要人工，是这套工具里唯一能离线批量产出的结论表；选链之前先看这里。
      </p>
      <p style="color:var(--text-2)">
        ${d.total} 条算子链在样本上逐一跑伪影审计，<b style="color:var(--ok)">${d.admitted_count} 条准入</b>。
        判据：截断率 &gt;10%，或噪声 σ 超 7.6/255 且放大 &gt;2.5× 即禁用。
      </p>
      <table class="grid">
        <thead><tr><th>算子链</th><th>说明</th><th>拒绝率</th><th>截断</th><th>噪声σ</th><th>均分</th><th>结论</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  } catch (e) {
    const bd = $("#modalRoot .modal-body");
    if (bd) bd.innerHTML = `<div class="reason reject">失败：${esc(e.message)}</div>`;
  }
}

async function openEdits() {
  const bid = S.batchId;
  const d = await api(`/api/edits?limit=120${bid ? "&batch_id=" + bid : ""}`);
  const list = d.edits || [];
  const kindCn = { add: "加算子", remove: "删算子", move: "换序", param: "改参数", accept: "通过", reject: "拒绝", reset: "套用配方" };
  const rows = list.slice().reverse().map((e) => `<tr>
    <td>${new Date((e.ts || 0) * 1000).toLocaleTimeString()}</td>
    <td class="txt">${esc(e.filename || "")}</td>
    <td class="txt">${esc(kindCn[e.kind] || e.kind || "")}</td>
    <td>${esc(e.op || "")}</td>
  </tr>`).join("");
  openModal("人工改动流水", `
    <p style="margin-top:0;color:var(--text-2)">
      共 ${list.length} 条。这份流水是 AI 下次首推更准的语料来源——人都改了什么、改成什么，全部留痕。
    </p>
    ${list.length ? `<table class="grid"><thead><tr><th>时间</th><th>样本</th><th>动作</th><th>算子</th></tr></thead><tbody>${rows}</tbody></table>`
      : '<div class="empty-hint">还没有改动记录</div>'}`,
    `<button onclick="closeModal()">关闭</button>`);
}

/* ---------------------------------------------------------------- 事件绑定 */

function bindEvents() {
  $("#batchSelect").addEventListener("change", (e) => selectBatch(e.target.value));

  $("#newBatchBtn").addEventListener("click", () => {
    openModal("新建批次", `
      <div class="field"><label>批次名称</label><input type="text" id="nbName" placeholder="例如：铜线-A 低照度批次"></div>
      <div class="field"><label>说明（可选）</label><textarea id="nbNote" placeholder="这批图的成像条件、来源工位…"></textarea></div>`,
      `<button onclick="closeModal()">取消</button>
       <button class="primary" onclick="createBatch()">创建</button>`);
    setTimeout(() => $("#nbName") && $("#nbName").focus(), 50);
  });

  $("#delBatchBtn").addEventListener("click", deleteBatchDialog);

  $("#sampleList").addEventListener("click", (e) => {
    const item = e.target.closest(".sample-item");
    if (item) selectSample(item.dataset.sid);
  });

  $("#dropZone").addEventListener("click", () => $("#fileInput").click());
  $("#fileInput").addEventListener("change", (e) => uploadFiles(e.target.files));

  const dz = $("#dropZone");
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
  });

  // 选文件夹走系统原生目录选择器，和"选图片"是同一套操作习惯。
  // 选完由浏览器把文件读上来，不再让人手打磁盘绝对路径（路径还常被转义坑）。
  $("#importDirBtn").addEventListener("click", () => $("#dirInput").click());
  $("#dirInput").addEventListener("change", (e) => {
    const { imgs, skipped } = splitImages(e.target.files);
    e.target.value = "";   // 清掉 value，同一个文件夹连选两次也能再次触发
    if (!imgs.length) return toast("这个文件夹里没有图片（png / jpg / bmp / tif）", true);
    uploadFiles(imgs, skipped);
  });

  $("#runAllBtn").addEventListener("click", runAll);
  $("#exportBtn").addEventListener("click", exportBatch);
  $("#preflightBtn").addEventListener("click", openPreflight);
  $("#editsBtn").addEventListener("click", openEdits);
  $("#settingsBtn").addEventListener("click", openSettings);

  $("#aiAgainBtn").addEventListener("click", analyzeCurrent);
  $("#addOpBtn").addEventListener("click", openOpPicker);
  $("#chain").addEventListener("click", (e) => {
    if (e.target.closest('[data-act="add-op"]')) openOpPicker();
  });
  $("#acceptBtn").addEventListener("click", acceptCurrent);
  $("#viewer").addEventListener("click", (e) => {
    const img = e.target.closest("img[data-zoom]");
    if (img) {
      // 一律先开单图，要对比再点右上角切 —— 大图首屏看的是"这一张的细节"，
      // 直接进并排会把手里的图压掉一半
      openLightbox(img.src, img.dataset.zoom, Number(img.dataset.step), false);
    }
  });

  // 只点遮罩关闭。点图本身不关——想看清细节时手会点在图上，一碰就关很难受
  $("#lightbox").addEventListener("click", (e) => {
    if (e.target.closest(".lightbox-stage img")) return;
    closeLightbox();
  });

  $("#lightboxPairBtn").addEventListener("click", (e) => {
    e.stopPropagation();
    toggleLightboxPair();
  });

  $("#lightboxClose").addEventListener("click", (e) => {
    e.stopPropagation();
    closeLightbox();
  });

  $("#candSwitcher").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-rid]");
    if (btn) switchCandidate(btn.dataset.rid);
  });

  $("#applyRecipeBtn").addEventListener("click", () => {
    const rid = $("#recipeSelect").value;
    if (rid) applyRecipeById(rid);
  });

  $$(".mode-btn").forEach((b) => b.addEventListener("click", () => {
    $$(".mode-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    S.viewMode = b.dataset.mode;
    renderViewer();
  }));

  // 算子链：input（拖滑杆）/ change（松手）/ click（按钮与缩略图）
  const chain = $("#chain");

  chain.addEventListener("input", (e) => {
    const el = e.target;
    if (el.tagName !== "INPUT" || el.type !== "range") return;
    const i = Number(el.dataset.i), k = el.dataset.k;
    const isInt = el.dataset.t === "int";
    const v = isInt ? parseInt(el.value, 10) : parseFloat(el.value);
    S.stages[i].p[k] = v;
    const lab = chain.querySelector(`[data-val-for="${i}-${k}"]`);
    if (lab) lab.textContent = isInt ? v : Number(v).toFixed(2);
    debouncePreview();
  });

  chain.addEventListener("change", (e) => {
    const el = e.target;
    if (el.tagName === "SELECT" && el.dataset.k) {
      const i = Number(el.dataset.i), k = el.dataset.k;
      S.stages[i].p[k] = el.value;
      markEdited("param", S.stages[i].op);
      previewChain(true);
      return;
    }
    if (el.tagName === "INPUT" && el.type === "text" && el.dataset.t === "floats") {
      const i = Number(el.dataset.i), k = el.dataset.k;
      const arr = el.value.split(",").map((x) => parseFloat(x.trim())).filter((x) => !Number.isNaN(x));
      if (!arr.length) { toast("尺度组不能为空", true); return; }
      S.stages[i].p[k] = arr;
      markEdited("param", S.stages[i].op);
      previewChain(true);
      return;
    }
    if (el.tagName === "INPUT" && el.type === "range") {
      const i = Number(el.dataset.i);
      markEdited("param", S.stages[i] ? S.stages[i].op : null);
    }
  });

  chain.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-act]");
    if (btn) {
      const act = btn.dataset.act;
      if (act === "up") moveStage(Number(btn.dataset.i), -1);
      else if (act === "down") moveStage(Number(btn.dataset.i), 1);
      else if (act === "del") deleteStage(Number(btn.dataset.i));
      else if (act === "thumb") {
        S.activeStep = Number(btn.dataset.step);
        renderViewer(); renderStepStrip(); renderMetrics(); renderAudit();
        highlightCard();
      }
      return;
    }
    const card = e.target.closest(".op-card");
    if (card) {
      S.activeStep = Number(card.dataset.card) + 1;
      renderViewer(); renderStepStrip(); renderMetrics(); renderAudit();
      highlightCard();
    }
  });

  $("#stepStrip").addEventListener("click", (e) => {
    const cell = e.target.closest("[data-step-idx]");
    if (!cell) return;
    S.activeStep = Number(cell.dataset.stepIdx);
    renderViewer(); renderStepStrip(); renderMetrics(); renderAudit();
    highlightCard();
  });

  $("#roundsBtn").addEventListener("click", openRoundsModal);
  $("#metricsBtn").addEventListener("click", openMetricsModal);
  $("#auditBtn").addEventListener("click", openAuditModal);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!$("#lightbox").hidden) closeLightbox();
      else closeModal();
    }
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") acceptCurrent();
  });
}

function highlightCard() {
  $$(".op-card").forEach((c) => c.classList.remove("selected"));
  const idx = S.activeStep - 1;
  const card = document.querySelector(`[data-card="${idx}"]`);
  if (card) card.classList.add("selected");
}

async function createBatch() {
  const name = $("#nbName").value.trim();
  const note = $("#nbNote").value.trim();
  try {
    const d = await postJson("/api/batches", { name, note });
    closeModal();
    await loadBatches(d.batch.id);
    toast("批次已创建");
  } catch (e) {
    toast("创建失败：" + e.message, true);
  }
}
window.createBatch = createBatch;

/* ---------------------------------------------------------------- 上传 */

// 文件夹里常混着 .txt / .ini / 缩略图缓存，先在前端挑出来，
// 少传一遍再让后端 skip，状态栏的"上传 N 张"也才对得上
const IMG_EXT = /\.(png|jpe?g|bmp|tiff?)$/i;

function splitImages(files) {
  const all = Array.from(files || []);
  const imgs = all.filter((f) => IMG_EXT.test(f.name) || (f.type || "").startsWith("image/"));
  return { imgs, skipped: all.length - imgs.length };
}

async function uploadFiles(files, skippedLocal = 0) {
  if (!S.batchId) {
    const d = await postJson("/api/batches", { name: "批次 " + new Date().toLocaleTimeString() });
    await loadBatches(d.batch.id);
  }
  if (!files || !files.length) return;
  const fd = new FormData();
  // 目录上传时浏览器会把相对路径（"铜线/xxx.png"）当文件名塞进 multipart，
  // 落库后文件名带斜杠，导出拼路径就会指到不存在的子目录。这里只取最后一段。
  const baseName = (f) => (f.name || "image").split(/[\\/]/).pop();
  Array.from(files).forEach((f) => fd.append("files", f, baseName(f)));
  status(`上传 ${files.length} 张…`);
  try {
    const d = await api(`/api/batches/${S.batchId}/upload`, { method: "POST", body: fd });
    const skipped = skippedLocal + ((d.skipped || []).length);
    toast(`上传 ${d.count} 张${skipped ? `，跳过 ${skipped} 个非图片文件` : ""}`);
    await loadBatchRefresh();
    status("上传完成", `${d.count} 张${skipped ? ` · 跳过 ${skipped}` : ""}`);
  } catch (e) {
    toast("上传失败：" + e.message, true);
  }
}

/* ---------------------------------------------------------------- 启动 */

async function boot() {
  bindEvents();
  try {
    S.schema = await api("/api/schema");
    const sel = $("#recipeSelect");
    sel.innerHTML = S.schema.recipes
      .filter((r) => r.id !== "r00_baseline")
      .map((r) => `<option value="${r.id}">${esc(r.id)} · ${esc(r.desc)}</option>`)
      .join("");
  } catch (e) {
    toast("加载算子表失败，后端未启动？", true);
    return;
  }
  await loadHealth();
  await loadBatches();
  status("就绪", "先建批次 → 导入图片 → AI 分析 → 核验 → 导出");
}

boot();
