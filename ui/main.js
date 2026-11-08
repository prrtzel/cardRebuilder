const state = {
  port: null,
  ocrResults: [],
  currentImage: null,
  images: {},
  activeImage: null,
  modelConnected: false,
  busy: false,
  manualMask: [],
  symbols: [],
  symbolLibrary: [],
  symbolById: {},
  symbolWarnings: [],
  symbolMode: "ai",
  drawMode: null, // null | "mask" | "symbol"
  hoverRow: null,
  hoverSymbol: null,
  upscale: 4,
};

const canvasImg = document.getElementById("canvas-img");
const viewport = document.getElementById("canvas-viewport");
const overlay = document.getElementById("canvas-overlay");
const overlayCtx = overlay.getContext("2d");
const modelLabel = document.getElementById("model-label");

const btnLoad = document.getElementById("btn-load");
const btnOcr = document.getElementById("btn-ocr");
const btnAiFix = document.getElementById("btn-ai-fix");
const btnRender = document.getElementById("btn-render");
const btnInpaint = document.getElementById("btn-inpaint");
const btnDrawMask = document.getElementById("btn-draw-mask");
const btnDrawSymbol = document.getElementById("btn-draw-symbol");
const btnClearMask = document.getElementById("btn-clear-mask");
const btnDetectSymbols = document.getElementById("btn-detect-symbols");
const symbolMatcher = document.getElementById("symbol-matcher");
const symbolStatus = document.getElementById("symbol-status");
const symbolTableBody = document.getElementById("symbol-table-body");
const tabSymbols = document.querySelector('#tabs .tab[data-tab="symbols"]');
const drawHint = document.getElementById("draw-hint");
const upscaleSelect = document.getElementById("upscale-factor");
const finalSize = document.getElementById("final-size");

let scale = 1;
let tx = 0;
let ty = 0;
let editing = null; // { idx, td, original }
let maskDrag = null; // { startX, startY, curX, curY } in image coordinates
let symbolDrag = null; // same shape, created by the symbol draw mode

// ---------------------------------------------------------------------------
// Canvas: pan (drag) + zoom (wheel) + OCR box overlay
// ---------------------------------------------------------------------------
function applyTransform() {
  canvasImg.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
  syncOverlayGeometry();
}

function syncOverlayGeometry() {
  if (!canvasImg.complete || !canvasImg.naturalWidth || !canvasImg.naturalHeight) {
    overlay.style.display = "none";
    return;
  }
  const vpRect = viewport.getBoundingClientRect();
  const imgRect = canvasImg.getBoundingClientRect();
  if (imgRect.width < 1 || imgRect.height < 1) {
    overlay.style.display = "none";
    return;
  }
  overlay.style.display = "block";
  overlay.style.left = `${imgRect.left - vpRect.left}px`;
  overlay.style.top = `${imgRect.top - vpRect.top}px`;
  overlay.style.width = `${imgRect.width}px`;
  overlay.style.height = `${imgRect.height}px`;

  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(imgRect.width * dpr));
  const h = Math.max(1, Math.round(imgRect.height * dpr));
  if (overlay.width !== w || overlay.height !== h) {
    overlay.width = w;
    overlay.height = h;
  }
  drawBoxes();
}

function drawBoxes() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  // Mask regions map to image coordinates, so they can be shown on both the
  // original and the clean background (useful for spotting leftovers).
  const visible = state.activeImage === "original" || state.activeImage === "clean";
  if (!visible) return;
  const nw = canvasImg.naturalWidth;
  const nh = canvasImg.naturalHeight;
  if (!nw || !nh) return;

  const sx = overlay.width / nw;
  const sy = overlay.height / nh;
  const dpr = window.devicePixelRatio || 1;

  const drawRect = (b, stroke, fill, lineWidth, dash = []) => {
    const x = b.x_min * sx;
    const y = b.y_min * sy;
    const w = Math.max(1, (b.x_max - b.x_min) * sx);
    const h = Math.max(1, (b.y_max - b.y_min) * sy);
    overlayCtx.save();
    overlayCtx.lineWidth = Math.max(1, Math.round(lineWidth * dpr));
    overlayCtx.setLineDash(dash.map((v) => v * dpr));
    overlayCtx.strokeStyle = stroke;
    overlayCtx.fillStyle = fill;
    overlayCtx.fillRect(x, y, w, h);
    overlayCtx.strokeRect(x, y, w, h);
    overlayCtx.restore();
  };

  state.ocrResults.forEach((row, i) => {
    const b = row.bbox;
    if (!b || b.x_min === undefined) return;
    const enabled = row.inpaint !== false;
    const hovered = state.hoverRow === i;
    if (enabled) {
      drawRect(
        b,
        "#00e5ff",
        hovered ? "rgba(0, 229, 255, 0.32)" : "rgba(0, 229, 255, 0.15)",
        hovered ? 3 : 2
      );
    } else {
      // Excluded from the mask: shown dashed so it is obvious what survives.
      drawRect(
        b,
        "#8a8a8a",
        "rgba(136, 136, 136, 0.08)",
        hovered ? 3 : 1.5,
        [6, 4]
      );
    }
  });

  for (const b of state.manualMask) {
    drawRect(b, "#ff9f43", "rgba(255, 159, 67, 0.18)", 2);
  }

  for (const row of state.symbols) {
    const b = row.bbox;
    if (!b || b.x_min === undefined) continue;
    const hovered = row.row_id === state.hoverSymbol;
    const confirmed = !!row.symbol_id;
    const stroke = confirmed ? "#ff4fd8" : "#ffb347";
    const fill = hovered
      ? "rgba(255, 79, 216, 0.34)"
      : confirmed
        ? "rgba(255, 79, 216, 0.13)"
        : "rgba(255, 179, 71, 0.16)";
    drawRect(b, stroke, fill, hovered ? 3 : 2, confirmed ? [] : [6, 4]);
  }

  if (symbolDrag) {
    drawRect(
      {
        x_min: Math.min(symbolDrag.startX, symbolDrag.curX),
        y_min: Math.min(symbolDrag.startY, symbolDrag.curY),
        x_max: Math.max(symbolDrag.startX, symbolDrag.curX),
        y_max: Math.max(symbolDrag.startY, symbolDrag.curY),
      },
      "#ff4fd8",
      "rgba(255, 79, 216, 0.14)",
      1.5,
      [5, 3]
    );
  }

  if (maskDrag) {
    drawRect(
      {
        x_min: Math.min(maskDrag.startX, maskDrag.curX),
        y_min: Math.min(maskDrag.startY, maskDrag.curY),
        x_max: Math.max(maskDrag.startX, maskDrag.curX),
        y_max: Math.max(maskDrag.startY, maskDrag.curY),
      },
      "#ffffff",
      "rgba(255, 255, 255, 0.12)",
      1.5,
      [5, 3]
    );
  }
}

function clientToImage(e) {
  const nw = canvasImg.naturalWidth;
  const nh = canvasImg.naturalHeight;
  const rect = canvasImg.getBoundingClientRect();
  if (!nw || !nh || rect.width < 1 || rect.height < 1) return null;
  return {
    x: Math.min(nw, Math.max(0, ((e.clientX - rect.left) / rect.width) * nw)),
    y: Math.min(nh, Math.max(0, ((e.clientY - rect.top) / rect.height) * nh)),
  };
}

function dragToBox(drag) {
  const x1 = Math.round(Math.min(drag.startX, drag.curX));
  const y1 = Math.round(Math.min(drag.startY, drag.curY));
  const x2 = Math.round(Math.max(drag.startX, drag.curX));
  const y2 = Math.round(Math.max(drag.startY, drag.curY));
  if (x2 - x1 < 4 || y2 - y1 < 4) return null;
  return { x_min: x1, y_min: y1, x_max: x2, y_max: y2 };
}

viewport.addEventListener("wheel", (e) => {
  e.preventDefault();
  const factor = e.deltaY > 0 ? 0.9 : 1.1;
  scale = Math.min(8, Math.max(0.25, scale * factor));
  applyTransform();
}, { passive: false });

let panning = false;
let panStart = { x: 0, y: 0, tx: 0, ty: 0 };

viewport.addEventListener("pointerdown", (e) => {
  // Stop the webview from starting a native image drag; otherwise pointer
  // events stop firing mid-drag and neither panning nor box drawing works.
  e.preventDefault();
  if (state.drawMode) {
    if (e.button !== 0) return;
    const p = clientToImage(e);
    if (!p) return;
    const drag = { startX: p.x, startY: p.y, curX: p.x, curY: p.y };
    if (state.drawMode === "mask") {
      maskDrag = drag;
    } else {
      symbolDrag = drag;
    }
    viewport.setPointerCapture(e.pointerId);
    drawBoxes();
    return;
  }
  panning = true;
  panStart = { x: e.clientX, y: e.clientY, tx: tx, ty: ty };
  viewport.setPointerCapture(e.pointerId);
});
viewport.addEventListener("pointermove", (e) => {
  if (maskDrag || symbolDrag) {
    const p = clientToImage(e);
    if (p) {
      const drag = maskDrag || symbolDrag;
      drag.curX = p.x;
      drag.curY = p.y;
      drawBoxes();
    }
    return;
  }
  if (!panning) return;
  tx = panStart.tx + (e.clientX - panStart.x) / scale;
  ty = panStart.ty + (e.clientY - panStart.y) / scale;
  applyTransform();
});
viewport.addEventListener("pointerup", (e) => {
  if (symbolDrag) {
    const drag = symbolDrag;
    symbolDrag = null;
    const box = dragToBox(drag);
    if (box) addManualSymbolBox(box);
    try { viewport.releasePointerCapture(e.pointerId); } catch (_) { /* noop */ }
    drawBoxes();
    return;
  }
  if (maskDrag) {
    const drag = maskDrag;
    maskDrag = null;
    const box = dragToBox(drag);
    if (box) {
      state.manualMask.push(box);
      updatePipelineState();
    }
    try { viewport.releasePointerCapture(e.pointerId); } catch (_) { /* noop */ }
    drawBoxes();
    return;
  }
  panning = false;
  try { viewport.releasePointerCapture(e.pointerId); } catch (_) { /* noop */ }
});
viewport.addEventListener("pointercancel", () => {
  panning = false;
  if (maskDrag || symbolDrag) {
    maskDrag = null;
    symbolDrag = null;
    drawBoxes();
  }
});

viewport.addEventListener("contextmenu", (e) => {
  if (!state.drawMode) return;
  const p = clientToImage(e);
  if (!p) return;
  if (state.drawMode === "symbol") {
    const row = findSymbolAt(p, true);
    if (row) {
      e.preventDefault();
      removeSymbolRow(row.row_id);
    }
    return;
  }
  for (let i = state.manualMask.length - 1; i >= 0; i--) {
    const b = state.manualMask[i];
    if (p.x >= b.x_min && p.x <= b.x_max && p.y >= b.y_min && p.y <= b.y_max) {
      e.preventDefault();
      state.manualMask.splice(i, 1);
      updatePipelineState();
      drawBoxes();
      return;
    }
  }
});

canvasImg.addEventListener("dragstart", (e) => e.preventDefault());
canvasImg.addEventListener("load", () => {
  // Keep the current zoom/pan when swapping views (Original/Clean/Final) or
  // refreshing the same card after inpaint/render. The view is only reset
  // explicitly when a different card is loaded.
  applyTransform();
});
canvasImg.addEventListener("error", () => {
  overlay.style.display = "none";
});
window.addEventListener("resize", syncOverlayGeometry);

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
async function api(path, options = {}) {
  const res = await fetch(`http://127.0.0.1:${state.port}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  if (!res.ok) {
    let detail = "";
    try {
      const data = await res.json();
      detail = data && data.detail ? String(data.detail) : "";
    } catch (_) {
      // Non-JSON error body; fall back to the generic message.
    }
    throw new Error(detail || `API ${res.status}: ${path}`);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return await res.json();
  }
  return null;
}

async function waitForBackend(timeoutMs = 60000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(`http://127.0.0.1:${state.port}/api/health`, { cache: "no-store" });
      if (res.ok) return true;
    } catch (_) {
      // Sidecar process is still starting up; retry.
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

async function refreshModel() {
  try {
    const data = await api("/api/model");
    state.modelConnected = !!(data && data.model);
    modelLabel.textContent = state.modelConnected
      ? `🟢 Active Model: ${data.model}`
      : "🔴 Active Model: Not Connected";
  } catch (e) {
    state.modelConnected = false;
    modelLabel.textContent = "🔴 Active Model: Not Connected";
  }
  updatePipelineState();
}

async function setImage(name) {
  const url = state.images ? state.images[name] : null;
  if (!url) return;
  state.activeImage = name;
  // Cache-bust: the URL is identical for every card, so without this the
  // webview can keep showing the previous image after a reload.
  canvasImg.src = `http://127.0.0.1:${state.port}${url}?t=${Date.now()}`;
}

function resetView() {
  scale = 1;
  tx = 0;
  ty = 0;
  applyTransform();
}

async function selectImage(name) {
  const btn = document.querySelector(`#image-toggle button[data-name="${name}"]`);
  if (!btn || btn.disabled) return;
  document.querySelectorAll("#image-toggle button").forEach((x) => x.classList.remove("active"));
  btn.classList.add("active");
  if (name === "final" && state.drawMode) setDrawMode(null);
  await setImage(name);
}

function setDrawMode(mode) {
  // Clicking the active mode's button toggles back to pan mode.
  if (mode === state.drawMode) mode = null;
  state.drawMode = mode || null;
  const on = !!state.drawMode;
  btnDrawMask.classList.toggle("active", state.drawMode === "mask");
  btnDrawSymbol.classList.toggle("active", state.drawMode === "symbol");
  viewport.classList.toggle("draw-mode", on);
  drawHint.style.display = on ? "inline" : "none";
  if (state.drawMode === "symbol") {
    drawHint.textContent =
      "Drag a box around a symbol. Right-click a manual symbol box to delete it.";
  } else if (state.drawMode === "mask") {
    drawHint.textContent =
      "Drag on the image to mask a region. Right-click a drawn box to delete it.";
  }
  if (on && state.activeImage !== "original" && state.activeImage !== "clean") {
    selectImage("original");
  }
  drawBoxes();
}

function updateImageToggles() {
  document.querySelectorAll("#image-toggle button").forEach((b) => {
    b.disabled = !state.images[b.dataset.name];
  });
}

function ensureActiveImage() {
  const buttons = [...document.querySelectorAll("#image-toggle button")];
  let active = buttons.find((b) => b.classList.contains("active"));
  if (!active || active.disabled) {
    active = buttons.find((b) => !b.disabled) || null;
    buttons.forEach((b) => b.classList.toggle("active", b === active));
  }
  return active;
}

async function refreshImages() {
  try {
    const data = await api("/api/images");
    state.images = data || {};
  } catch (e) {
    state.images = {};
  }
  updateImageToggles();
  const active = ensureActiveImage();
  if (active) {
    await setImage(active.dataset.name);
  } else {
    state.activeImage = null;
    drawBoxes();
  }
  return state.images;
}

// ---------------------------------------------------------------------------
// Pipeline button state
// ---------------------------------------------------------------------------
function updatePipelineState() {
  const hasImage = !!state.currentImage;
  const hasResults = state.ocrResults.length > 0;
  const drawableSymbols = state.symbols.filter(
    (row) => row.symbol_id && row.redraw !== false
  );
  const hasMaskInput =
    hasResults || state.manualMask.length > 0 || drawableSymbols.length > 0;
  const busy = state.busy;

  btnLoad.disabled = busy;
  btnOcr.disabled = busy || !hasImage;
  btnAiFix.disabled = busy || !hasResults || !state.modelConnected;
  btnRender.disabled = busy || !hasImage || !hasMaskInput;
  btnInpaint.disabled = busy || !hasImage || !hasMaskInput;
  btnDrawMask.disabled = busy || !hasImage;
  btnDrawSymbol.disabled = busy || !hasImage;
  btnClearMask.disabled = busy || state.manualMask.length === 0;
  btnDetectSymbols.disabled = busy || !hasImage || !state.symbolLibrary.length;
  symbolMatcher.disabled = busy || !hasImage;
  upscaleSelect.disabled = busy;
}

async function withBusy(btn, label, fn) {
  if (state.busy) return;
  state.busy = true;
  const original = btn.textContent;
  btn.textContent = label;
  updatePipelineState();
  try {
    await fn();
  } finally {
    btn.textContent = original;
    state.busy = false;
    updatePipelineState();
  }
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
function switchTab(name) {
  document.querySelectorAll("#tabs .tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    p.classList.toggle("active", p.id === `tab-panel-${name}`);
  });
}

// ---------------------------------------------------------------------------
// Text table (contenteditable: no input boxes, cursor moves naturally)
// ---------------------------------------------------------------------------
function flagCell(row, i, key, title) {
  const td = document.createElement("td");
  td.className = "flag-cell";
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = row[key] !== false;
  cb.dataset.rowIndex = i;
  cb.dataset.flag = key;
  cb.title = title;
  td.appendChild(cb);
  return td;
}

function renderTable() {
  const tbody = document.getElementById("table-body");
  tbody.innerHTML = "";
  state.hoverRow = null;
  state.ocrResults.forEach((row, i) => {
    const tr = document.createElement("tr");
    tr.dataset.rowIndex = i;
    const tdMask = flagCell(row, i, "inpaint", "Include this region in the inpaint mask");
    const tdRedraw = flagCell(row, i, "redraw", "Redraw this text on the rebuilt card");
    const tdText = document.createElement("td");
    tdText.className = "text-cell" + (row.ai_modified ? " ai-modified" : "");
    tdText.dataset.rowIndex = i;
    tdText.spellcheck = false;
    tdText.textContent = row.text || "";
    const tdY = document.createElement("td");
    tdY.className = "y-cell";
    tdY.textContent = (row.bbox && row.bbox.y_min !== undefined) ? String(row.bbox.y_min) : "";
    tr.appendChild(tdMask);
    tr.appendChild(tdRedraw);
    tr.appendChild(tdText);
    tr.appendChild(tdY);
    tbody.appendChild(tr);
  });
}

function startEdit(idx, td) {
  if (editing) {
    if (editing.td === td) return;
    commitEdit();
  }
  editing = { idx, td, original: state.ocrResults[idx].text || "" };
  td.contentEditable = "true";
  td.classList.add("editing");
  td.focus();

  const range = document.createRange();
  range.selectNodeContents(td);
  range.collapse(false);
  const sel = window.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
}

function commitEdit() {
  if (!editing) return;
  const { idx, td } = editing;
  editing = null;
  td.contentEditable = "false";
  td.classList.remove("editing");
  const value = td.textContent.replace(/\s+/g, " ").trim();
  td.textContent = value;
  if (state.ocrResults[idx]) state.ocrResults[idx].text = value;
}

function cancelEdit() {
  if (!editing) return;
  const { td, original } = editing;
  editing = null;
  td.contentEditable = "false";
  td.classList.remove("editing");
  td.textContent = original;
}

const tableBody = document.getElementById("table-body");

tableBody.addEventListener("click", (e) => {
  const td = e.target.closest("td.text-cell");
  if (!td) return;
  if (editing && editing.td === td) return;
  const idx = Number(td.dataset.rowIndex);
  if (Number.isNaN(idx)) return;
  startEdit(idx, td);
});

tableBody.addEventListener("keydown", (e) => {
  if (!editing) return;
  if (e.key === "Enter") {
    e.preventDefault();
    commitEdit();
  } else if (e.key === "Escape") {
    e.preventDefault();
    cancelEdit();
  }
});

tableBody.addEventListener("focusout", (e) => {
  if (editing && e.target === editing.td) commitEdit();
});

tableBody.addEventListener("paste", (e) => {
  if (!editing) return;
  e.preventDefault();
  const raw = (e.clipboardData || window.clipboardData).getData("text");
  const text = raw.replace(/\s+/g, " ");
  const sel = window.getSelection();
  if (!sel || !sel.rangeCount) return;
  const range = sel.getRangeAt(0);
  range.deleteContents();
  const node = document.createTextNode(text);
  range.insertNode(node);
  range.setStartAfter(node);
  range.collapse(true);
  sel.removeAllRanges();
  sel.addRange(range);
});

tableBody.addEventListener("change", (e) => {
  const cb = e.target.closest("input[data-flag]");
  if (!cb) return;
  const idx = Number(cb.dataset.rowIndex);
  const row = state.ocrResults[idx];
  if (!row) return;
  row[cb.dataset.flag] = cb.checked;
  if (cb.dataset.flag === "inpaint") drawBoxes();
});

tableBody.addEventListener("mouseover", (e) => {
  const tr = e.target.closest("tr[data-row-index]");
  const idx = tr ? Number(tr.dataset.rowIndex) : null;
  if (idx !== state.hoverRow) {
    state.hoverRow = idx;
    drawBoxes();
  }
});

tableBody.addEventListener("mouseleave", () => {
  if (state.hoverRow !== null) {
    state.hoverRow = null;
    drawBoxes();
  }
});

// ---------------------------------------------------------------------------
// Golden symbols
// ---------------------------------------------------------------------------
function setSymbolStatus(text, cls = "") {
  symbolStatus.textContent = text;
  symbolStatus.className = cls;
  if (cls === "warn") tabSymbols.classList.add("symbol-warn");
}

function symbolDisplayName(symbolId) {
  const entry = state.symbolById[symbolId];
  return entry ? entry.name : symbolId || "—";
}

function symbolAssetUrl(symbolId) {
  return `http://127.0.0.1:${state.port}/api/symbols/asset/${encodeURIComponent(symbolId)}`;
}

function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(2) : "";
}

function bboxIoU(a, b) {
  if (!a || !b) return 0;
  const ix1 = Math.max(a.x_min, b.x_min);
  const iy1 = Math.max(a.y_min, b.y_min);
  const ix2 = Math.min(a.x_max, b.x_max);
  const iy2 = Math.min(a.y_max, b.y_max);
  if (ix2 <= ix1 || iy2 <= iy1) return 0;
  const intersection = (ix2 - ix1) * (iy2 - iy1);
  const areaA = (a.x_max - a.x_min) * (a.y_max - a.y_min);
  const areaB = (b.x_max - b.x_min) * (b.y_max - b.y_min);
  const union = areaA + areaB - intersection;
  return union > 0 ? intersection / union : 0;
}

async function loadSymbolLibrary() {
  try {
    const data = await api("/api/symbols/library");
    state.symbolLibrary = (data && data.symbols) || [];
  } catch (e) {
    state.symbolLibrary = [];
  }
  state.symbolById = Object.fromEntries(
    state.symbolLibrary.map((symbol) => [symbol.id, symbol])
  );
  updatePipelineState();
}

function updateSymbolStatus() {
  if (!state.currentImage) {
    setSymbolStatus("Load a card, then detect symbols.");
    tabSymbols.classList.remove("symbol-warn");
    return;
  }

  const total = state.symbols.length;
  const assigned = state.symbols.filter((row) => !!row.symbol_id).length;
  const parts = [];
  if (total) {
    parts.push(`${total} box${total === 1 ? "" : "es"} · ${assigned} assigned`);
  } else {
    parts.push("No symbols yet — click Detect Symbols or draw a box on the card.");
  }
  if (state.symbolWarnings.length) {
    parts.push(state.symbolWarnings.join(" "));
  }

  let cls = "";
  if (state.symbolWarnings.length) {
    cls = "warn";
  } else if (total && assigned < total) {
    cls = "warn";
  } else if (total) {
    cls = "ok";
  }
  setSymbolStatus(parts.join(" "), cls);

  const hasRowWarning = state.symbols.some((row) => !!row.warning);
  tabSymbols.classList.toggle(
    "symbol-warn",
    hasRowWarning || state.symbolWarnings.length > 0 || (total > 0 && assigned < total)
  );
}

function renderSymbolTable() {
  symbolTableBody.innerHTML = "";
  state.hoverSymbol = null;

  if (!state.symbols.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.className = "symbol-empty";
    td.textContent = state.currentImage
      ? "No symbols yet. Click Detect Symbols or draw a box on the card."
      : "Load a card to detect symbols.";
    tr.appendChild(td);
    symbolTableBody.appendChild(tr);
    return;
  }

  for (const row of state.symbols) {
    const tr = document.createElement("tr");
    tr.className = "symbol-row" + (row.symbol_id ? "" : " unassigned");
    tr.dataset.rowId = row.row_id;

    const tdThumb = document.createElement("td");
    if (row.symbol_id) {
      const img = document.createElement("img");
      img.className = "symbol-thumb";
      img.src = symbolAssetUrl(row.symbol_id);
      img.alt = symbolDisplayName(row.symbol_id);
      img.title = symbolDisplayName(row.symbol_id);
      img.draggable = false;
      tdThumb.appendChild(img);
    } else {
      const placeholder = document.createElement("span");
      placeholder.className = "symbol-thumb-placeholder";
      placeholder.textContent = "?";
      placeholder.title = "No golden symbol assigned";
      tdThumb.appendChild(placeholder);
    }
    tr.appendChild(tdThumb);

    const tdMatch = document.createElement("td");
    tdMatch.className = "symbol-match-cell";
    const select = document.createElement("select");
    select.className = "symbol-match-select";
    select.dataset.rowId = row.row_id;
    select.title = "Choose the golden symbol to draw here";
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "— none —";
    select.appendChild(none);
    for (const symbol of state.symbolLibrary) {
      const option = document.createElement("option");
      option.value = symbol.id;
      option.textContent = symbol.name;
      select.appendChild(option);
    }
    select.value = row.symbol_id || "";
    tdMatch.appendChild(select);

    const meta = document.createElement("div");
    meta.className = "symbol-meta";
    if (row.source === "manual") {
      const label = document.createElement("span");
      label.textContent = "Manual box";
      meta.appendChild(label);
    } else if (row.method === "ai") {
      const label = document.createElement("span");
      label.textContent = `AI match · ${formatScore(row.confidence)}`;
      meta.appendChild(label);
    } else if (row.method === "template") {
      const label = document.createElement("span");
      label.textContent = `Template match · ${formatScore(row.confidence)}`;
      meta.appendChild(label);
    }
    if (
      row.template_id &&
      row.template_id !== row.symbol_id &&
      row.template_score != null
    ) {
      const useTemplate = document.createElement("button");
      useTemplate.type = "button";
      useTemplate.className = "link symbol-use-template";
      useTemplate.dataset.rowId = row.row_id;
      useTemplate.textContent =
        `Use template: ${symbolDisplayName(row.template_id)} ` +
        `(${formatScore(row.template_score)})`;
      meta.appendChild(useTemplate);
    }
    if (row.warning) {
      const warning = document.createElement("div");
      warning.className = "warn";
      warning.textContent = row.warning;
      meta.appendChild(warning);
    }
    if (meta.childNodes.length) tdMatch.appendChild(meta);
    tr.appendChild(tdMatch);

    const tdRedraw = document.createElement("td");
    tdRedraw.className = "flag-cell";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = row.redraw !== false;
    checkbox.disabled = !row.symbol_id;
    checkbox.dataset.symbolFlag = "redraw";
    checkbox.dataset.rowId = row.row_id;
    checkbox.title = "Replace the original symbol on the rebuilt card";
    tdRedraw.appendChild(checkbox);
    tr.appendChild(tdRedraw);

    const tdDelete = document.createElement("td");
    const del = document.createElement("button");
    del.type = "button";
    del.className = "symbol-delete";
    del.dataset.rowId = row.row_id;
    del.title = "Remove this symbol box";
    del.textContent = "✕";
    tdDelete.appendChild(del);
    tr.appendChild(tdDelete);

    symbolTableBody.appendChild(tr);
  }
}

function findSymbolRow(rowId) {
  return state.symbols.find((row) => row.row_id === rowId) || null;
}

function findSymbolAt(point, manualOnly = false) {
  for (let i = state.symbols.length - 1; i >= 0; i--) {
    const row = state.symbols[i];
    if (manualOnly && row.source !== "manual") continue;
    const b = row.bbox;
    if (!b) continue;
    if (
      point.x >= b.x_min &&
      point.x <= b.x_max &&
      point.y >= b.y_min &&
      point.y <= b.y_max
    ) {
      return row;
    }
  }
  return null;
}

function removeSymbolRow(rowId) {
  const before = state.symbols.length;
  state.symbols = state.symbols.filter((row) => row.row_id !== rowId);
  if (state.symbols.length === before) return;
  if (state.hoverSymbol === rowId) state.hoverSymbol = null;
  renderSymbolTable();
  updatePipelineState();
  updateSymbolStatus();
  drawBoxes();
}

async function addManualSymbolBox(box) {
  // A manual box wins over any auto detection it overlaps, so the same
  // symbol is never drawn twice and the table has no duplicate rows.
  state.symbols = state.symbols.filter(
    (row) => row.source === "manual" || bboxIoU(row.bbox, box) <= 0.25
  );
  const row = {
    row_id: `manual_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
    symbol_id: null,
    bbox: box,
    source: "manual",
    method: "manual",
    confidence: null,
    template_id: null,
    template_score: null,
    alternates: [],
    warning: "",
    redraw: true,
  };
  state.symbols.push(row);
  renderSymbolTable();
  updatePipelineState();
  drawBoxes();
  setSymbolStatus("Matching manual box...");

  try {
    const data = await api("/api/symbols/match", {
      method: "POST",
      body: JSON.stringify({ bbox: box }),
    });
    const current = findSymbolRow(row.row_id);
    if (!current) return; // Removed while the match was running.
    const suggestions = (data && data.suggestions) || [];
    const best = suggestions[0] || null;
    if (best) {
      current.symbol_id = best.symbol_id;
      current.template_id = best.symbol_id;
      current.template_score = Number(best.score);
      current.alternates = suggestions.slice(1, 4);
      current.redraw = true;
      if (Number(best.score) >= 0.66 && best.bbox) {
        current.bbox = best.bbox;
      }
      if (Number(best.score) < 0.58) {
        current.warning = "Weak template match; verify or change the symbol.";
      }
    } else {
      current.warning = "No template match; choose a symbol from the Match list.";
      current.redraw = false;
    }
  } catch (e) {
    const current = findSymbolRow(row.row_id);
    if (current) {
      current.warning = `Template matching failed: ${e.message || e}`;
      current.redraw = false;
    }
  }

  renderSymbolTable();
  updatePipelineState();
  updateSymbolStatus();
  drawBoxes();
}

async function detectSymbols() {
  if (!state.currentImage) return;
  if (!state.symbolLibrary.length) {
    await loadSymbolLibrary();
  }
  if (!state.symbolLibrary.length) {
    setSymbolStatus(
      "Symbol library unavailable. Check that the backend is running, then retry.",
      "warn"
    );
    return;
  }
  state.symbolWarnings = [];
  setSymbolStatus(
    state.symbolMode === "ai"
      ? "Detecting symbols... AI matching can take up to ~30 seconds."
      : "Detecting symbols with template matching..."
  );

  await withBusy(btnDetectSymbols, "Detecting...", async () => {
    try {
      const data = await api("/api/symbols/detect", {
        method: "POST",
        body: JSON.stringify({ mode: state.symbolMode }),
      });
      const manual = state.symbols.filter((row) => row.source === "manual");
      const auto = (data.symbols || []).filter(
        (candidate) =>
          !manual.some((row) => bboxIoU(candidate.bbox, row.bbox) > 0.25)
      );
      state.symbols = [...manual, ...auto].sort(
        (a, b) =>
          (a.bbox ? a.bbox.y_min : 0) - (b.bbox ? b.bbox.y_min : 0) ||
          (a.bbox ? a.bbox.x_min : 0) - (b.bbox ? b.bbox.x_min : 0)
      );
      state.symbolWarnings = Array.isArray(data.warnings) ? data.warnings : [];
      renderSymbolTable();
      updatePipelineState();
      updateSymbolStatus();
      drawBoxes();
    } catch (e) {
      state.symbolWarnings = [];
      setSymbolStatus(`Symbol detection failed: ${e.message || e}`, "warn");
    }
  });
}

function applySymbolWarnings(data) {
  state.symbolWarnings =
    data && Array.isArray(data.warnings) ? data.warnings.slice() : [];
  updateSymbolStatus();
}

// ---------------------------------------------------------------------------
// Pipeline actions
// ---------------------------------------------------------------------------
async function loadImage() {
  let selected;
  try {
    selected = await window.__TAURI__.dialog.open({
      multiple: false,
      filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg"] }],
    });
  } catch (e) {
    alert(`Could not open the file dialog: ${e.message || e}`);
    return;
  }
  const filePath = Array.isArray(selected) ? selected[0] : selected;
  if (!filePath) return;

  await withBusy(btnLoad, "Waiting for backend...", async () => {
    if (!(await waitForBackend(60000))) {
      alert("The backend did not start in time. Wait a few seconds and try again.");
      return;
    }
    btnLoad.textContent = "Loading image...";
    try {
      await api("/api/load", {
        method: "POST",
        body: JSON.stringify({ path: filePath }),
      });
      state.currentImage = filePath;
      state.ocrResults = [];
      state.manualMask = [];
      state.symbols = [];
      state.symbolWarnings = [];
      state.hoverRow = null;
      state.hoverSymbol = null;
      finalSize.textContent = "";
      finalSize.classList.remove("warn");
      setDrawMode(null);
      resetView();
      renderTable();
      renderSymbolTable();
      updateSymbolStatus();
      await refreshImages();
    } catch (e) {
      alert(`Failed to load image: ${e.message || e}`);
    }
  });
}

async function runOcr() {
  await withBusy(btnOcr, "Running OCR...", async () => {
    try {
      const results = await api("/api/ocr", { method: "POST" });
      state.ocrResults = results || [];
      renderTable();
      syncOverlayGeometry();
    } catch (e) {
      alert(`OCR failed: ${e.message || e}`);
    }
  });
}

async function aiFix() {
  if (!state.ocrResults.length) return;
  await withBusy(btnAiFix, "Fixing text...", async () => {
    try {
      const data = await api("/api/ai-fix", {
        method: "POST",
        body: JSON.stringify({ results: state.ocrResults }),
      });
      document.getElementById("context-panel").textContent = data.context || "";
      switchTab("context");
      const corrected = data.corrected || [];
      for (let i = 0; i < state.ocrResults.length; i++) {
        if (i < corrected.length) {
          const newText = corrected[i];
          const oldText = state.ocrResults[i].text;
          state.ocrResults[i].ai_modified = newText !== oldText;
          state.ocrResults[i].text = newText;
        }
      }
      if (corrected.length > state.ocrResults.length) {
        const extra = corrected.slice(state.ocrResults.length).join(" ");
        const last = state.ocrResults[state.ocrResults.length - 1];
        if (last) {
          last.text += " " + extra;
          last.ai_modified = true;
        }
      }
      renderTable();
    } catch (e) {
      alert(`AI Fix failed: ${e.message || e}`);
    }
  });
}

async function renderCard() {
  const scaleLabel = state.upscale > 1 ? ` ${state.upscale}x` : "";
  await withBusy(btnRender, `Rendering${scaleLabel}...`, async () => {
    finalSize.textContent = state.upscale > 1 ? `Upscaling ${state.upscale}x...` : "";
    finalSize.classList.remove("warn");
    try {
      const data = await api("/api/render", {
        method: "POST",
        body: JSON.stringify({
          results: state.ocrResults,
          extra_mask: state.manualMask,
          symbols: state.symbols,
          upscale: state.upscale,
        }),
      });
      showFinalSize(data);
      applySymbolWarnings(data);
      await refreshImages();
      // Show the result immediately instead of silently staying on Original.
      await selectImage("final");
    } catch (e) {
      finalSize.textContent = "";
      alert(`Render failed: ${e.message || e}`);
    }
  });
}

function showFinalSize(data) {
  if (!data || !Array.isArray(data.size)) {
    finalSize.textContent = "";
    finalSize.classList.remove("warn");
    return;
  }
  const [w, h] = data.size;
  const factor = data.upscale > 1 ? `${data.upscale}x` : "1x";
  const fallback = data.engine === "lanczos-fallback";
  finalSize.textContent =
    `Final: ${w} × ${h} (${factor})` + (fallback ? " — Lanczos fallback" : "");
  finalSize.classList.toggle("warn", fallback);
}

async function inpaintOnly() {
  await withBusy(btnInpaint, "Updating clean BG...", async () => {
    try {
      const data = await api("/api/inpaint", {
        method: "POST",
        body: JSON.stringify({
          results: state.ocrResults,
          extra_mask: state.manualMask,
          symbols: state.symbols,
        }),
      });
      applySymbolWarnings(data);
      await refreshImages();
      await selectImage("clean");
    } catch (e) {
      alert(`Clean background update failed: ${e.message || e}`);
    }
  });
}

// ---------------------------------------------------------------------------
// Event binding
// ---------------------------------------------------------------------------
function bindEvents() {
  document.getElementById("btn-refresh-model").addEventListener("click", refreshModel);
  btnLoad.addEventListener("click", loadImage);
  btnOcr.addEventListener("click", runOcr);
  btnAiFix.addEventListener("click", aiFix);
  btnRender.addEventListener("click", renderCard);
  btnInpaint.addEventListener("click", inpaintOnly);

  upscaleSelect.addEventListener("change", () => {
    state.upscale = Number(upscaleSelect.value) || 1;
    // The readout describes the previous render; clear it until the next one.
    finalSize.textContent = "";
    finalSize.classList.remove("warn");
  });

  btnDrawMask.addEventListener("click", () => setDrawMode("mask"));
  btnDrawSymbol.addEventListener("click", () => setDrawMode("symbol"));
  btnClearMask.addEventListener("click", () => {
    if (!state.manualMask.length) return;
    state.manualMask = [];
    updatePipelineState();
    drawBoxes();
  });

  btnDetectSymbols.addEventListener("click", detectSymbols);
  symbolMatcher.addEventListener("change", () => {
    state.symbolMode = symbolMatcher.value === "template" ? "template" : "ai";
  });

  symbolTableBody.addEventListener("change", (e) => {
    const rowId = e.target.dataset.rowId;
    const row = rowId ? findSymbolRow(rowId) : null;
    if (!row) return;
    if (e.target.classList.contains("symbol-match-select")) {
      row.symbol_id = e.target.value || null;
      row.redraw = !!row.symbol_id;
      row.warning = "";
      renderSymbolTable();
      updatePipelineState();
      updateSymbolStatus();
      drawBoxes();
    } else if (e.target.dataset.symbolFlag === "redraw") {
      row.redraw = e.target.checked;
      updatePipelineState();
      updateSymbolStatus();
    }
  });

  symbolTableBody.addEventListener("click", (e) => {
    const useTemplate = e.target.closest(".symbol-use-template");
    if (useTemplate) {
      const row = findSymbolRow(useTemplate.dataset.rowId);
      if (!row || !row.template_id) return;
      row.symbol_id = row.template_id;
      row.redraw = true;
      row.warning = "";
      renderSymbolTable();
      updatePipelineState();
      updateSymbolStatus();
      drawBoxes();
      return;
    }
    const del = e.target.closest(".symbol-delete");
    if (del) removeSymbolRow(del.dataset.rowId);
  });

  symbolTableBody.addEventListener("mouseover", (e) => {
    const tr = e.target.closest("tr[data-row-id]");
    const rowId = tr ? tr.dataset.rowId : null;
    if (rowId !== state.hoverSymbol) {
      state.hoverSymbol = rowId;
      drawBoxes();
    }
  });

  symbolTableBody.addEventListener("mouseleave", () => {
    if (state.hoverSymbol !== null) {
      state.hoverSymbol = null;
      drawBoxes();
    }
  });

  document.querySelectorAll("#image-toggle button").forEach((b) => {
    b.addEventListener("click", () => selectImage(b.dataset.name));
  });

  document.querySelectorAll("#tabs .tab").forEach((b) => {
    b.addEventListener("click", () => switchTab(b.dataset.tab));
  });
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", async () => {
  try {
    state.port = await window.__TAURI__.core.invoke("get_backend_port");
  } catch (e) {
    state.port = 8008;
  }
  bindEvents();
  updatePipelineState();
  renderTable();
  renderSymbolTable();
  updateSymbolStatus();
  modelLabel.textContent = "⏳ Starting backend...";
  await waitForBackend(30000);
  await refreshModel();
  await Promise.all([loadSymbolLibrary(), refreshImages()]);
  renderSymbolTable();
  updateSymbolStatus();
});
