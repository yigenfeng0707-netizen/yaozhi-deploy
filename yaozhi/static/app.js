/* 窑智 · 前端逻辑（原生 JS，无外部依赖） */
(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const state = {
    ready: false,
    sampleUrl: null,
    vis: "overlay",
    result: null,
    defect: "",
    zone: null,
    kilnSeries: null,
    busy: false,
  };

  /* ---------------- 通用 ---------------- */
  async function api(url, opt) {
    const r = await fetch(url, opt);
    if (!r.ok) {
      let msg = r.statusText;
      try { msg = (await r.json()).detail || msg; } catch (_) {}
      throw new Error(msg);
    }
    const ct = r.headers.get("content-type") || "";
    return ct.includes("application/json") ? r.json() : r.blob();
  }
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function setBusy(btn, on, txt) {
    if (!btn) return;
    btn.disabled = on;
    if (on) { btn.dataset.t = btn.textContent; btn.textContent = txt || "处理中…"; }
    else if (btn.dataset.t) btn.textContent = btn.dataset.t;
  }

  /* ---------------- 初始化 ---------------- */
  async function initStatus() {
    const badge = $("#statusBadge");
    const txt = $("#statusText");
    try {
      const s = await api("/api/status");
      state.ready = s.ready;
      badge.classList.toggle("on", s.ready);
      txt.textContent = s.ready ? "内核就绪" : "未建库";
      const st = s.stats || {};
      $("#modelNote").textContent =
        `记忆库 ${st.bank || 0} 向量 · 阈值 ${st.tex_thr ?? "—"} / ${st.chroma_thr ?? "—"} · ${s.model_size_kb} KB`;
    } catch (e) {
      badge.classList.remove("on");
      txt.textContent = "服务异常";
      $("#modelNote").textContent = e.message;
    }
  }

  /* ---------------- 屏幕切换 ---------------- */
  $$(".tab").forEach((t) => t.addEventListener("click", () => {
    $$(".tab").forEach((x) => x.classList.remove("active"));
    $$(".screen").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("#" + t.dataset.screen).classList.add("active");
    if (t.dataset.screen === "s2" && !state.kilnSeries) runSim();
  }));

  /* ================= 屏 1：釉面质检 ================= */
  const drop = $("#drop");
  const fileInput = $("#fileInput");
  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("over");
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) detectBlob(f, f.name);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) detectBlob(fileInput.files[0], fileInput.files[0].name);
  });

  $$("#sampleChips .chip").forEach((c) => c.addEventListener("click", async () => {
    $$("#sampleChips .chip").forEach((x) => x.classList.remove("on"));
    c.classList.add("on");
    const kind = c.dataset.kind;
    const seed = 5;
    try {
      const blob = await api(`/api/sample?kind=${encodeURIComponent(kind)}&seed=${seed}`);
      await detectBlob(blob, `合成样本 · ${kind}`, kind);
    } catch (e) { alert("样本加载失败：" + e.message); }
  }));

  function setPreview(blob) {
    if (state.sampleUrl) URL.revokeObjectURL(state.sampleUrl);
    state.sampleUrl = URL.createObjectURL(blob);
    $("#imgSrc").src = state.sampleUrl;
  }

  async function detectBlob(blob, tag, kind) {
    if (state.busy) return;
    state.busy = true;
    setPreview(blob);
    $("#imgTag").textContent = tag || "样本";
    $("#verdict").textContent = "…";
    $("#verdict").className = "verdict";
    $("#latency").textContent = "推理中…";
    try {
      const fd = new FormData();
      fd.append("file", blob, "sample.png");
      const r = await api("/api/detect", { method: "POST", body: fd });
      state.result = r;
      state.defect = r.verdict === "NG" ? r.defect_type : "";
      state.zone = r.loop ? r.loop.zone : null;
      renderResult(r);
      if (r.verdict === "NG" && r.defect_type !== "无") {
        $$("#kilnChips .chip").forEach((x) => x.classList.toggle("on", x.dataset.dt === r.defect_type));
        state.defect = r.defect_type;
      }
      syncKilnZone();
    } catch (e) {
      $("#verdict").textContent = "ERR";
      $("#verdict").className = "verdict ng";
      $("#latency").textContent = e.message;
    } finally {
      state.busy = false;
    }
  }

  function renderResult(r) {
    const v = $("#verdict");
    v.textContent = r.verdict;
    v.className = "verdict " + (r.verdict === "NG" ? "ng" : "ok");
    const sc = Math.max(0, Math.min(100, r.score));
    $("#scoreFill").style.width = (r.verdict === "NG" ? Math.max(12, sc) : Math.max(2, sc * 0.5)) + "%";
    $("#scoreVal").textContent = sc.toFixed(1);
    $("#rType").textContent = r.verdict === "NG" ? r.defect_type : "正常";
    $("#rRatio").textContent = r.raw_score.toFixed(2) + "×";
    $("#rCount").textContent = r.regions.length;
    $("#rSize").textContent = r.size.join(" × ");
    $("#rCause").textContent = r.cause || "未见显著异常";
    $("#rAction").textContent = r.action || "维持当前工艺参数";
    $("#latency").textContent = `推理 ${r.latency_ms ?? "—"} ms · 阈值 1.00×`;
    $("#imgVis").src = state.vis === "heatmap" ? r.heatmap : r.overlay;
    state.vis = "overlay";
    $$(".vt").forEach((b) => b.classList.toggle("active", b.dataset.vis === "overlay"));
  }

  $$(".vt").forEach((b) => b.addEventListener("click", () => {
    if (!state.result) return;
    $$(".vt").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.vis = b.dataset.vis;
    $("#imgVis").src = state.vis === "heatmap" ? state.result.heatmap : state.result.overlay;
  }));

  /* ================= 屏 2：窑温孪生 ================= */
  const SLIDERS = [
    { k: "rate1", name: "升温速率", min: 50, max: 160, step: 5, unit: "℃/h" },
    { k: "rate2", name: "高温段速率", min: 40, max: 140, step: 5, unit: "℃/h" },
    { k: "soak_temp", name: "保温温度", min: 1140, max: 1210, step: 5, unit: "℃" },
    { k: "soak_min", name: "高温保温", min: 30, max: 120, step: 5, unit: "min" },
    { k: "cool_rate", name: "冷却速率", min: 40, max: 180, step: 5, unit: "℃/h" },
    { k: "hold573_min", name: "573℃ 平台", min: 0, max: 90, step: 5, unit: "min" },
  ];
  const params = { rate1: 120, rate2: 100, soak_temp: 1180, soak_min: 45, cool_rate: 120, hold573_min: 0 };

  function buildSliders() {
    const wrap = $("#sliders");
    wrap.innerHTML = SLIDERS.map((s) => `
      <div class="sl">
        <div class="sl-top"><label>${s.name}</label><b id="v-${s.k}">${params[s.k]}<small> ${s.unit}</small></b></div>
        <input type="range" id="r-${s.k}" min="${s.min}" max="${s.max}" step="${s.step}" value="${params[s.k]}">
      </div>`).join("");
    SLIDERS.forEach((s) => {
      const inp = $("#r-" + s.k);
      inp.addEventListener("input", () => {
        params[s.k] = Number(inp.value);
        $("#v-" + s.k).innerHTML = `${params[s.k]}<small> ${s.unit}</small>`;
        scheduleSim();
      });
    });
  }
  let simTimer = null;
  function scheduleSim() {
    clearTimeout(simTimer);
    simTimer = setTimeout(runSim, 220);
  }

  async function runSim() {
    try {
      const q = new URLSearchParams(params).toString();
      const r = await api("/api/kiln/simulate?" + q);
      state.kilnSeries = r.series;
      drawChart(r.series);
      renderKpis(r);
      $("#kilnHint").textContent = "仿真曲线 · " + r.cycle_min + " min";
    } catch (e) { $("#kilnHint").textContent = "仿真失败：" + e.message; }
  }

  function renderKpis(r) {
    const riskCls = r.risk >= 30 ? "bad" : r.risk >= 18 ? "warn" : "good";
    $("#kpis").innerHTML = `
      <div class="kpi"><label>周期</label><b>${r.cycle_min}<small>min</small></b></div>
      <div class="kpi"><label>能耗</label><b>${r.energy_kwh_m2}<small>kWh/m²</small></b></div>
      <div class="kpi ${riskCls}"><label>工艺风险</label><b>${r.risk}<small>%</small></b></div>
      <div class="kpi"><label>综合成本</label><b>${r.total_cost}<small>元/m²</small></b></div>
      <div class="kpi"><label>最大温差</label><b>${r.max_deltaT}<small>℃</small></b></div>
      <div class="kpi"><label>573℃ 温差</label><b>${r.quartz_deltaT}<small>℃</small></b></div>
      <div class="kpi"><label>冷却速率</label><b>${r.cool_rate}<small>℃/h</small></b></div>
      <div class="kpi"><label>废品率</label><b>${r.scrap_rate}<small>%</small></b></div>`;
  }

  /* ---------- Canvas 绘图 ---------- */
  function drawChart(series) {
    const cv = $("#chart");
    if (!cv || !series) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = cv.getBoundingClientRect();
    const W = Math.max(360, rect.width || 760);
    const H = 360;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.height = H + "px";
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);

    const pad = { l: 54, r: 16, t: 18, b: 34 };
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;
    const tmax = Math.ceil((series.t[series.t.length - 1] || 1200) / 200) * 200;
    const ymax = 1300;
    const X = (t) => pad.l + (t / tmax) * pw;
    const Y = (v) => pad.t + ph - (v / ymax) * ph;

    // 背景网格
    g.strokeStyle = "rgba(255,255,255,.055)";
    g.lineWidth = 1;
    g.font = "10px Consolas, monospace";
    g.fillStyle = "#5d7280";
    for (let v = 0; v <= ymax; v += 200) {
      const y = Y(v);
      g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
      g.textAlign = "right"; g.fillText(v + "℃", pad.l - 8, y + 3);
    }
    for (let t = 0; t <= tmax; t += 200) {
      const x = X(t);
      g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, pad.t + ph); g.stroke();
      g.textAlign = "center"; g.fillText(t + "", x, H - 14);
    }

    // 573℃ 石英转变参考线
    g.save();
    g.setLineDash([5, 5]);
    g.strokeStyle = "rgba(227,93,106,.55)";
    g.beginPath(); g.moveTo(pad.l, Y(573)); g.lineTo(W - pad.r, Y(573)); g.stroke();
    g.restore();
    g.fillStyle = "rgba(227,93,106,.8)";
    g.textAlign = "left";
    g.fillText("573℃ 石英晶型转变", pad.l + 8, Y(573) - 6);

    const line = (arr, color, w) => {
      g.beginPath();
      for (let i = 0; i < series.t.length; i++) {
        const x = X(series.t[i]), y = Y(arr[i]);
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      g.strokeStyle = color; g.lineWidth = w; g.lineJoin = "round";
      g.stroke();
    };
    // 填充气氛曲线下方
    g.beginPath();
    for (let i = 0; i < series.t.length; i++) {
      const x = X(series.t[i]), y = Y(series.gas[i]);
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    g.lineTo(X(series.t[series.t.length - 1]), Y(0));
    g.lineTo(X(series.t[0]), Y(0));
    g.closePath();
    const grad = g.createLinearGradient(0, pad.t, 0, pad.t + ph);
    grad.addColorStop(0, "rgba(240,169,59,.16)");
    grad.addColorStop(1, "rgba(240,169,59,0)");
    g.fillStyle = grad; g.fill();

    line(series.gas, "#f0a93b", 2);
    line(series.surface, "#35d0d8", 2);
    line(series.core, "#8f7bf0", 1.6);

    // 坐标轴
    g.strokeStyle = "rgba(255,255,255,.16)";
    g.beginPath();
    g.moveTo(pad.l, pad.t); g.lineTo(pad.l, pad.t + ph); g.lineTo(W - pad.r, pad.t + ph);
    g.stroke();
    g.fillStyle = "#5d7280"; g.textAlign = "right";
    g.fillText("min", W - pad.r, H - 2);
  }
  window.addEventListener("resize", () => {
    if (state.kilnSeries) drawChart(state.kilnSeries);
  });

  /* ---------- 缺陷联动 ---------- */
  $$("#kilnChips .chip").forEach((c) => c.addEventListener("click", () => {
    $$("#kilnChips .chip").forEach((x) => x.classList.remove("on"));
    c.classList.add("on");
    state.defect = c.dataset.dt || "";
    if (!state.defect) state.zone = null;
    syncKilnZone();
  }));

  function syncKilnZone() {
    const el = $("#kilnZone");
    if (!el) return;
    if (state.zone) el.textContent = `窑位：${state.zone.zone} · 温区偏差 +${state.zone.bias}℃`;
    else el.textContent = state.defect ? `缺陷：${state.defect} · 未指定窑位` : "窑位：未指定";
  }

  $("#btnSim").addEventListener("click", runSim);

  $("#btnOpt").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    setBusy(btn, true, "寻优中…");
    try {
      const body = { defect_type: state.defect || null, params, limit_h: 24 };
      if (state.zone) { body.u = state.zone.u; body.v = state.zone.v; body.zone_bias = state.zone.bias; }
      else if (state.defect) body.zone_bias = 8;
      const r = await api("/api/kiln/optimize", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      const rec = r.recommended, d = r.delta;
      $("#optBody").innerHTML = `
        <div class="delta-row">
          <div class="dc"><label>风险变化</label><b class="${d.risk <= 0 ? "down" : "up"}">${d.risk > 0 ? "+" : ""}${d.risk} pt</b></div>
          <div class="dc"><label>能耗变化</label><b class="${d.energy <= 0 ? "down" : "up"}">${d.energy > 0 ? "+" : ""}${d.energy} kWh</b></div>
          <div class="dc"><label>成本变化</label><b class="${d.cost <= 0 ? "down" : "up"}">${d.cost > 0 ? "+" : ""}${d.cost} 元</b></div>
          <div class="dc"><label>周期变化</label><b class="${d.cycle <= 0 ? "down" : "up"}">${d.cycle > 0 ? "+" : ""}${d.cycle} min</b></div>
        </div>
        <p class="micro">参数网格 ${r.search_total} 组（窑期≤${body.limit_h}h 约束内评估 ${r.search_space} 组）；推荐曲线风险 ${rec.risk}%、综合成本 ${rec.total_cost} 元/m²。</p>`;
      Object.assign(params, rec.params);
      SLIDERS.forEach((s) => {
        $("#r-" + s.k).value = params[s.k];
        $("#v-" + s.k).innerHTML = `${params[s.k]}<small> ${s.unit}</small>`;
      });
      state.kilnSeries = rec.series;
      drawChart(rec.series);
      renderKpis(rec);
      $("#kilnHint").textContent = "推荐曲线 · " + rec.cycle_min + " min";
    } catch (err) {
      $("#optBody").innerHTML = `<p class="micro" style="color:#e35d6a">寻优失败：${esc(err.message)}</p>`;
    } finally {
      setBusy(btn, false);
    }
  });

  /* ================= 屏 3：纹样 ================= */
  async function initPattern() {
    try {
      const o = await api("/api/pattern/options");
      $("#pStyle").innerHTML = o.styles.map((s) => `<option>${s}</option>`).join("");
      $("#pVessel").innerHTML = o.vessels.map((s) => `<option>${s}</option>`).join("");
      genPattern();
    } catch (e) { $("#patGrid").innerHTML = `<p class="micro" style="color:#e35d6a">${esc(e.message)}</p>`; }
  }

  async function genPattern() {
    const btn = $("#btnPattern");
    setBusy(btn, true, "生成中…");
    const q = new URLSearchParams({
      style: $("#pStyle").value, vessel: $("#pVessel").value,
      seed: $("#pSeed").value || "1", count: "4",
    });
    try {
      const r = await api("/api/pattern/generate?" + q.toString());
      $("#patGrid").innerHTML = r.variants.map((v, i) => {
        const m = v.metrics;
        const best = i === r.best_index;
        const cls = best ? "pat best" : "pat";
        const issues = m.issues && m.issues.length
          ? `<div class="pat-issues">! ${m.issues.map(esc).join("；")}</div>`
          : `<div class="pat-ok">✓ 满足丝网/贴花工艺下限</div>`;
        return `<div class="${cls}">
          <div class="pat-img">
            <img src="${v.image}" alt="">
            <div class="pat-badge">${m.score} 分${best ? " · 推荐" : ""}</div>
          </div>
          <div class="pat-meta">
            <div class="pn">${esc(v.palette)} · ${esc(r.summary.style)}</div>
            <div class="pm"><span>线宽 ${m.min_width_mm}mm</span><span>覆盖 ${(m.coverage * 100).toFixed(1)}%</span></div>
            <div class="pm" style="margin-top:4px"><span>对称 ${(m.symmetry * 100).toFixed(0)}%</span><span>留白 ${m.edge_clearance_mm}mm</span></div>
            ${issues}
          </div>
        </div>`;
      }).join("");
    } catch (e) {
      $("#patGrid").innerHTML = `<p class="micro" style="color:#e35d6a">生成失败：${esc(e.message)}</p>`;
    } finally {
      setBusy(btn, false);
    }
  }
  $("#btnPattern").addEventListener("click", genPattern);

  /* ================= 闭环演示 ================= */
  const ov = $("#loopOverlay");
  $("#btnDemo").addEventListener("click", runLoop);
  $("#btnCloseLoop").addEventListener("click", () => ov.classList.remove("on"));
  ov.addEventListener("click", (e) => { if (e.target === ov) ov.classList.remove("on"); });

  function stage(i) {
    $$(".step").forEach((s) => {
      const n = Number(s.dataset.i);
      s.classList.toggle("done", n < i);
      s.classList.toggle("active", n === i);
    });
  }

  async function runLoop() {
    const kind = "釉泡";
    const seed = 5;
    ov.classList.add("on");
    stage(0);
    $("#loopBody").innerHTML = `<div class="loading"><div class="spinner"></div>生成「${kind}」缺陷样本并执行检测…</div>`;
    try {
      const r = await api("/api/demo", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind, seed }),
      });
      stage(1);
      await sleep(420);
      stage(2);
      await sleep(420);
      stage(3);
      const b = r.before, a = r.after, o = r.optimize;
      const d = o.delta;
      $("#loopBody").innerHTML = `
        <div class="lb-grid">
          <div class="lb-card">
            <div class="hd">① 缺陷检测 · ${esc(b.defect_type)} · 显著度 ${b.score}</div>
            <div class="bd"><img src="${b.overlay}" alt=""></div>
          </div>
          <div class="lb-card">
            <div class="hd">④ 下一窑首件复检 · ${a.verdict} · 显著度 ${a.score}</div>
            <div class="bd"><img src="${a.overlay}" alt=""></div>
          </div>
        </div>
        <div class="lb-flow">
          <div class="flow-item"><label>② 窑位归因</label><b>${esc(r.zone.zone)}</b></div>
          <div class="flow-item"><label>③ 推荐保温温度</label><b>${o.recommended.params.soak_temp} ℃</b></div>
          <div class="flow-item"><label>风险变化</label><b class="hl">${d.risk > 0 ? "+" : ""}${d.risk} pt → ${o.recommended.risk}%</b></div>
          <div class="flow-item"><label>综合成本</label><b class="hl">${o.recommended.total_cost} 元/m²</b></div>
          <div class="flow-item"><label>周期</label><b>${o.recommended.cycle_min} min</b></div>
          <div class="flow-item"><label>全链路耗时</label><b>${r.elapsed_ms} ms</b></div>
        </div>
        <div class="loop-note">
          数据口径：缺陷样本与复检良品均为参数化合成（釉色族「${esc(r.family)}」），用于离线复现与演示；
          接入产线后同一接口可直接用实拍良品图重建特征库，热工模型可用热电偶实测曲线反演标定。
        </div>`;
      state.defect = b.defect_type;
      state.zone = r.zone;
      syncKilnZone();
      $$("#kilnChips .chip").forEach((x) => x.classList.toggle("on", x.dataset.dt === state.defect));
    } catch (e) {
      $("#loopBody").innerHTML = `<p class="micro" style="color:#e35d6a">演示失败：${esc(e.message)}</p>`;
    }
  }
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  /* ---------------- 启动 ---------------- */
  buildSliders();
  initStatus();
  initPattern();
  runSim();
})();
