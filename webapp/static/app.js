/* ===================================================================
   TrafficAI — app.js (Command Center UI)
   =================================================================== */

(() => {
  "use strict";

  // ── Elements ──────────────────────────────────────────────────────
  const tabBtns       = document.querySelectorAll(".tab-btn");
  const panelSingle   = document.getElementById("panel-single");
  const panelBatch    = document.getElementById("panel-batch");

  const singleForm    = document.getElementById("singleForm");
  const resultCard    = document.getElementById("resultCard");
  const resultIdle    = document.getElementById("resultIdle");
  const resultData    = document.getElementById("resultData");

  const demandNumber  = document.getElementById("demandNumber");
  const demandPct     = document.getElementById("demandPct");
  const demandBadge   = document.getElementById("demandBadge");
  const demandBarFill = document.getElementById("demandBarFill");
  const demandBarPin  = document.getElementById("demandBarPin");
  const demandBarTip  = document.getElementById("demandBarTip");

  const statusDot     = document.getElementById("statusDot");
  const statusText    = document.getElementById("statusText");

  const dropzone        = document.getElementById("dropzone");
  const fileInput       = document.getElementById("fileInput");
  const fileInfo        = document.getElementById("fileInfo");
  const fileName        = document.getElementById("fileName");
  const fileMeta        = document.getElementById("fileMeta");
  const batchPredictBtn = document.getElementById("batchPredictBtn");
  const downloadBtn     = document.getElementById("downloadBtn");
  const previewWrap     = document.getElementById("previewWrap");
  const previewTable    = document.getElementById("previewTable");
  const loadingOverlay  = document.getElementById("loadingOverlay");

  let currentFile  = null;
  let lastBlobUrl  = null;

  // ── Health Probe ──────────────────────────────────────────────────
  (async () => {
    try {
      const r = await fetch("/health");
      const d = await r.json();
      if (r.ok && d.status === "ok") {
        statusDot.classList.add("ok");
        statusText.textContent = "Model Ready";
      } else {
        statusDot.classList.add("err");
        statusText.textContent = "Model Offline";
      }
    } catch {
      statusDot.classList.add("err");
      statusText.textContent = "Offline";
    }
  })();

  // ── Tab switching ─────────────────────────────────────────────────
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      tabBtns.forEach(b => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });
      const isSingle = target === "single";
      panelSingle.classList.toggle("active", isSingle);
      panelSingle.hidden = !isSingle;
      panelBatch.classList.toggle("active", !isSingle);
      panelBatch.hidden = isSingle;
    });
  });

  // ── Custom Selects ────────────────────────────────────────────────
  document.querySelectorAll(".csel").forEach(csel => {
    const btn    = csel.querySelector(".csel-btn");
    const val    = csel.querySelector(".csel-val");
    const list   = csel.querySelector(".csel-list");
    const hidden = csel.querySelector("input[type=hidden]");
    const items  = csel.querySelectorAll(".csel-item");

    const openMenu  = () => { csel.classList.add("open");  btn.setAttribute("aria-expanded", "true"); };
    const closeMenu = () => { csel.classList.remove("open"); btn.setAttribute("aria-expanded", "false"); };
    const toggle    = () => csel.classList.contains("open") ? closeMenu() : openMenu();

    btn.addEventListener("click", e => { e.stopPropagation(); toggle(); });
    btn.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
      if (e.key === "Escape") closeMenu();
      if (e.key === "ArrowDown") { e.preventDefault(); openMenu(); list.querySelector(".csel-item")?.focus(); }
    });

    items.forEach(item => {
      item.addEventListener("click", () => {
        items.forEach(i => i.classList.remove("sel"));
        item.classList.add("sel");
        hidden.value = item.dataset.v;
        val.textContent = item.textContent;
        closeMenu();
        btn.focus();
      });
      item.addEventListener("keydown", e => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); item.click(); }
        if (e.key === "Escape") { closeMenu(); btn.focus(); }
      });
    });

    // Close on outside click
    document.addEventListener("click", e => {
      if (!csel.contains(e.target)) closeMenu();
    });
  });

  // ── Single Predict ────────────────────────────────────────────────
  singleForm.addEventListener("submit", async e => {
    e.preventDefault();
    const fd = new FormData(singleForm);
    const payload = {};
    fd.forEach((v, k) => {
      payload[k] = ["day", "Temperature", "NumberofLanes"].includes(k) ? Number(v) : v;
    });

    showLoading(true);
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        alert(`Prediction failed: ${err.error || res.statusText}`);
        return;
      }
      const data = await res.json();
      renderResult(data);
    } catch (err) {
      alert(`Network error: ${err.message}`);
    } finally {
      showLoading(false);
    }
  });

  function renderResult({ demand }) {
    const value = Math.max(0, Math.min(1, Number(demand) || 0));
    const pct   = value * 100;

    // Reveal result panel
    resultIdle.hidden = true;
    resultData.hidden = false;
    resultCard.classList.add("has-result");

    // Badge
    let cls, label;
    if (value > 0.7)       { cls = "high";     label = "High Demand";  }
    else if (value >= 0.4) { cls = "moderate"; label = "Moderate";     }
    else                   { cls = "low";       label = "Low Demand";   }

    demandBadge.className = `rd-badge ${cls}`;
    demandBadge.textContent = label;

    // Color the big number
    const numColors = { high: "#fb7185", moderate: "#f59e0b", low: "#34d399" };
    demandNumber.style.color = numColors[cls];

    // Animate count-up
    animateNum(demandNumber, 0, value, 950, n => n.toFixed(4));
    animateNum(demandPct,    0, pct,   950, n => `${n.toFixed(1)}%`);

    // Animate bar (double rAF so CSS transition fires)
    requestAnimationFrame(() => requestAnimationFrame(() => {
      demandBarFill.style.width = `${pct}%`;
      demandBarPin.style.left   = `${Math.min(pct, 98)}%`;
      demandBarTip.textContent  = `${pct.toFixed(1)}%`;
    }));
  }

  function animateNum(el, from, to, ms, fmt) {
    const t0 = performance.now();
    const tick = now => {
      const p = Math.min(1, (now - t0) / ms);
      const e = 1 - Math.pow(1 - p, 3); // easeOutCubic
      el.textContent = fmt(from + (to - from) * e);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // ── Drag & Drop ───────────────────────────────────────────────────
  dropzone.addEventListener("click",  () => fileInput.click());
  dropzone.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });

  ["dragenter", "dragover"].forEach(ev =>
    dropzone.addEventListener(ev, e => {
      e.preventDefault(); e.stopPropagation();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach(ev =>
    dropzone.addEventListener(ev, e => {
      e.preventDefault(); e.stopPropagation();
      if (ev === "dragleave" && dropzone.contains(e.relatedTarget)) return;
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", e => {
    const f = e.dataTransfer?.files?.[0];
    if (f) handleFile(f);
  });
  fileInput.addEventListener("change", e => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  });

  function handleFile(file) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      alert("Please select a .csv file");
      return;
    }
    currentFile = file;
    fileName.textContent = file.name;
    fileMeta.textContent = `${(file.size / 1024).toFixed(1)} KB · counting rows…`;
    fileInfo.hidden = false;
    batchPredictBtn.disabled = false;
    downloadBtn.disabled = true;
    previewWrap.hidden = true;
    previewTable.innerHTML = "";
    if (lastBlobUrl) { URL.revokeObjectURL(lastBlobUrl); lastBlobUrl = null; }

    // Quick row-count estimate from first 64 KB
    const reader = new FileReader();
    reader.onload = () => {
      const lines = String(reader.result || "").split(/\r?\n/).filter(l => l.trim());
      const est = Math.max(0, lines.length - 1);
      fileMeta.textContent = `${(file.size / 1024).toFixed(1)} KB · ~${est.toLocaleString()} rows`;
    };
    reader.readAsText(file.slice(0, 64 * 1024));
  }

  // ── Batch Predict ─────────────────────────────────────────────────
  batchPredictBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    const formData = new FormData();
    formData.append("file", currentFile);

    showLoading(true);
    try {
      const res = await fetch("/predict_batch", { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        alert(`Batch failed: ${err.error || res.statusText}`);
        return;
      }
      const blob = await res.blob();
      if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
      lastBlobUrl = URL.createObjectURL(blob);
      downloadBtn.disabled = false;
      renderPreview(await blob.text());
    } catch (err) {
      alert(`Network error: ${err.message}`);
    } finally {
      showLoading(false);
    }
  });

  function renderPreview(csvText) {
    const rows = parseCSV(csvText);
    if (!rows.length) { previewWrap.hidden = true; return; }
    const headers = rows[0];
    const body    = rows.slice(1, 26); // first 25 data rows
    const thead = `<thead><tr>${headers.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead>`;
    const tbody = `<tbody>${body.map(r =>
      `<tr>${r.map((cell, i) =>
        `<td class="${headers[i] === "demand" ? "demand-cell" : ""}">${esc(cell)}</td>`
      ).join("")}</tr>`
    ).join("")}</tbody>`;
    previewTable.innerHTML = thead + tbody;
    previewWrap.hidden = false;
  }

  // ── Download ──────────────────────────────────────────────────────
  downloadBtn.addEventListener("click", () => {
    if (!lastBlobUrl) return;
    const a = document.createElement("a");
    a.href = lastBlobUrl;
    a.download = "predictions.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  // ── Loading overlay ───────────────────────────────────────────────
  function showLoading(on) { loadingOverlay.hidden = !on; }

  // ── RFC-4180 CSV parser ───────────────────────────────────────────
  function parseCSV(text) {
    const rows = []; let row = [], cur = "", inQ = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (inQ) {
        if (c === '"') { if (text[i + 1] === '"') { cur += '"'; i++; } else inQ = false; }
        else cur += c;
      } else {
        if (c === '"') inQ = true;
        else if (c === ',') { row.push(cur); cur = ""; }
        else if (c === '\n') { row.push(cur); rows.push(row); row = []; cur = ""; }
        else if (c !== '\r') cur += c;
      }
    }
    if (cur || row.length) { row.push(cur); rows.push(row); }
    return rows.filter(r => r.length > 1 || (r.length === 1 && r[0] !== ""));
  }

  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

})();
