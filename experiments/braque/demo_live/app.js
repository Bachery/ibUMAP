"use strict";

const state = {
  config: null,
  result: null,
  running: false,
  runStarted: null,
  timer: null,
  positions: [],
  history: [],
};

const elements = {
  form: document.getElementById("run-form"),
  algorithm: document.getElementById("algorithm-select"),
  cellCount: document.getElementById("cell-count-select"),
  deterministic: document.getElementById("deterministic-toggle"),
  seed: document.getElementById("seed-input"),
  runButton: document.getElementById("run-button"),
  algorithmNote: document.getElementById("algorithm-note"),
  inputScope: document.getElementById("input-scope"),
  serviceState: document.getElementById("service-state"),
  runStatus: document.getElementById("run-status"),
  statusTitle: document.getElementById("status-title"),
  statusDetail: document.getElementById("status-detail"),
  liveElapsed: document.getElementById("live-elapsed"),
  metricReduction: document.getElementById("metric-reduction"),
  metricHdbscan: document.getElementById("metric-hdbscan"),
  metricClusters: document.getElementById("metric-clusters"),
  metricNoise: document.getElementById("metric-noise"),
  metricNoiseCount: document.getElementById("metric-noise-count"),
  metricFingerprint: document.getElementById("metric-fingerprint"),
  plotTitle: document.getElementById("plot-title"),
  plotSubtitle: document.getElementById("plot-subtitle"),
  canvas: document.getElementById("embedding-canvas"),
  clusterBars: document.getElementById("cluster-bars"),
  effectiveConfig: document.getElementById("effective-config"),
  history: document.getElementById("run-history"),
  tooltip: document.getElementById("tooltip"),
  loadError: document.getElementById("load-error"),
};

const PHASES = {
  queued: ["Queued", "Waiting for the local execution slot."],
  preparing_input: ["Preparing the fixed cell subset", "The subset is independent of the algorithm seed."],
  embedding: ["Computing the embedding", "The selected dimensionality-reduction algorithm is running now."],
  hdbscan: ["Running HDBSCAN", "Clustering the newly computed 2D coordinates."],
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function seconds(value) {
  if (!Number.isFinite(value)) return "—";
  return value < 10 ? `${value.toFixed(2)} s` : `${value.toFixed(1)} s`;
}

function percent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
}

function integer(value) {
  return Number.isFinite(value) ? Math.round(value).toLocaleString() : "—";
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  let payload = {};
  try {
    payload = await response.json();
  } catch (_error) {
    payload = {};
  }
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function setServiceState(connected, title, detail) {
  elements.serviceState.classList.toggle("connected", connected);
  elements.serviceState.querySelector("strong").textContent = title;
  elements.serviceState.querySelector("small").textContent = detail;
}

function updateAlgorithmNote() {
  if (!state.config) return;
  const algorithm = elements.algorithm.value;
  const deterministic = elements.deterministic.checked;
  const base = state.config.algorithms[algorithm].description;
  let detail;
  if (deterministic && algorithm === "umap_learn") {
    detail = "A fixed random_state is used and umap-learn runs with n_jobs=1.";
  } else if (deterministic) {
    detail = `A fixed root seed is used with ibUMAP's deterministic execution path (${state.config.threads} graph workers).`;
  } else {
    detail = `No algorithm seed is supplied; up to ${state.config.threads} CPU workers may be used.`;
  }
  elements.algorithmNote.innerHTML = `<strong>${escapeHtml(state.config.algorithms[algorithm].label)}</strong><span>${escapeHtml(base)} ${escapeHtml(detail)}</span>`;
  elements.runButton.querySelector("span").textContent = `Run ${state.config.algorithms[algorithm].label} + HDBSCAN`;
}

function syncDeterministicControl() {
  elements.seed.disabled = !elements.deterministic.checked || state.running;
  updateAlgorithmNote();
}

function setRunning(running) {
  state.running = running;
  elements.algorithm.disabled = running;
  elements.cellCount.disabled = running;
  elements.deterministic.disabled = running;
  elements.seed.disabled = running || !elements.deterministic.checked;
  elements.runButton.disabled = running || !state.config;
  elements.runButton.classList.toggle("running", running);
}

function resetMetrics() {
  for (const element of [elements.metricReduction, elements.metricHdbscan, elements.metricClusters, elements.metricNoise, elements.metricFingerprint]) {
    element.textContent = "—";
  }
  elements.metricNoiseCount.textContent = "—";
}

function startClock() {
  stopClock();
  state.runStarted = performance.now();
  elements.liveElapsed.textContent = "0.0 s";
  state.timer = window.setInterval(() => {
    const elapsed = (performance.now() - state.runStarted) / 1000;
    elements.liveElapsed.textContent = `${elapsed.toFixed(1)} s`;
  }, 100);
}

function stopClock(finalSeconds = null) {
  if (state.timer !== null) window.clearInterval(state.timer);
  state.timer = null;
  if (Number.isFinite(finalSeconds)) elements.liveElapsed.textContent = seconds(finalSeconds);
}

function renderPhase(phase) {
  const [title, detail] = PHASES[phase] || ["Computing", "The backend is processing this run."];
  elements.runStatus.className = "run-status running";
  elements.statusTitle.textContent = title;
  elements.statusDetail.textContent = detail;
}

function clusterColor(label, alpha = 0.78) {
  if (label === -1) return `rgba(130, 139, 143, ${alpha})`;
  const hue = (label * 47 + 183) % 360;
  return `hsla(${hue}, 55%, 43%, ${alpha})`;
}

function prepareCanvas() {
  const rect = elements.canvas.getBoundingClientRect();
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  elements.canvas.width = Math.round(width * ratio);
  elements.canvas.height = Math.round(height * ratio);
  const context = elements.canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width, height };
}

function drawEmbedding() {
  const { context, width, height } = prepareCanvas();
  context.fillStyle = "#fbfaf6";
  context.fillRect(0, 0, width, height);
  state.positions = [];
  if (!state.result) return;
  const coordinates = state.result.coordinates_uint16;
  const labels = state.result.labels;
  const pad = 18;
  const usableWidth = width - pad * 2;
  const usableHeight = height - pad * 2;
  const radius = labels.length <= 2500 ? 1.65 : labels.length <= 5000 ? 1.25 : .9;
  const order = labels.map((label, index) => [label === -1 ? 0 : 1, index]).sort((a, b) => a[0] - b[0]);
  for (const [, index] of order) {
    const x = pad + (coordinates[index * 2] / 65535) * usableWidth;
    const y = pad + (1 - coordinates[index * 2 + 1] / 65535) * usableHeight;
    state.positions[index] = [x, y];
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.fillStyle = clusterColor(labels[index], labels[index] === -1 ? .34 : .68);
    context.fill();
  }
}

function renderClusterBars(result) {
  const rows = result.cluster_sizes.slice(0, 9);
  const maximum = Math.max(...rows.map((row) => row.count), 1);
  elements.clusterBars.innerHTML = rows.map((row) => {
    const name = row.label === -1 ? "Noise (−1)" : `Cluster ${row.label}`;
    return `<div class="cluster-row">
      <span><i style="background:${clusterColor(row.label, .9)}"></i>${name}</span>
      <div><b style="width:${row.count / maximum * 100}%;background:${clusterColor(row.label, .8)}"></b></div>
      <strong>${integer(row.count)}</strong>
    </div>`;
  }).join("");
}

function renderEffective(result) {
  const entries = [
    ["Engine", result.effective.engine],
    ["Mode", result.effective.deterministic ? "deterministic" : "unseeded / fast"],
    ["Seed", result.effective.random_state ?? "not set"],
    ["CPU workers", result.effective.n_jobs],
    ["UMAP settings", `neighbors ${result.effective.n_neighbors} · epochs ${result.effective.n_epochs} · min_dist ${result.effective.min_dist}`],
    ["HDBSCAN", `min cluster ${result.effective.hdbscan_min_cluster_size} · ε ${result.effective.hdbscan_cluster_selection_epsilon}`],
  ];
  elements.effectiveConfig.innerHTML = entries.map(([term, value]) => `<div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
}

function renderHistory() {
  elements.history.innerHTML = state.history.map((item, index) => `
    <article>
      <span>${state.history.length - index}</span>
      <div><strong>${escapeHtml(item.algorithm)}</strong><small>${escapeHtml(item.mode)} · ${integer(item.cells)} cells</small></div>
      <div><strong>${seconds(item.runtime)}</strong><small>reduction</small></div>
      <code>${escapeHtml(item.fingerprint)}</code>
    </article>`).join("");
}

function renderResult(result) {
  state.result = result;
  elements.metricReduction.textContent = seconds(result.timing_seconds.reduction);
  elements.metricHdbscan.textContent = seconds(result.timing_seconds.hdbscan);
  elements.metricClusters.textContent = integer(result.summary.cluster_count_excluding_noise);
  elements.metricNoise.textContent = percent(result.summary.noise_fraction);
  elements.metricNoiseCount.textContent = `${integer(result.summary.noise_count)} of ${integer(result.input.cell_count)} cells`;
  elements.metricFingerprint.textContent = result.summary.result_fingerprint.slice(0, 10);
  const algorithm = state.config.algorithms[result.request.algorithm].label;
  const mode = result.request.deterministic ? `deterministic · seed ${result.request.seed}` : "unseeded / fast";
  elements.plotTitle.textContent = `${algorithm} → HDBSCAN`;
  elements.plotSubtitle.textContent = `${integer(result.input.cell_count)} cells · ${mode}`;
  renderClusterBars(result);
  renderEffective(result);
  drawEmbedding();
  state.history.unshift({
    algorithm,
    mode,
    cells: result.input.cell_count,
    runtime: result.timing_seconds.reduction,
    fingerprint: result.summary.result_fingerprint.slice(0, 10),
  });
  state.history = state.history.slice(0, 5);
  renderHistory();
}

async function pollJob(statusUrl) {
  while (true) {
    const job = await requestJson(statusUrl);
    if (job.status === "completed") return job;
    if (job.status === "failed") throw new Error(`${job.error.type}: ${job.error.message}`);
    renderPhase(job.phase);
    await new Promise((resolve) => window.setTimeout(resolve, 600));
  }
}

async function runPipeline(event) {
  event.preventDefault();
  if (state.running) return;
  const seed = Number.parseInt(elements.seed.value, 10);
  if (elements.deterministic.checked && (!Number.isInteger(seed) || seed < 0 || seed > 2147483647)) {
    elements.seed.focus();
    return;
  }
  const request = {
    algorithm: elements.algorithm.value,
    cell_count: Number.parseInt(elements.cellCount.value, 10),
    deterministic: elements.deterministic.checked,
    seed,
  };
  setRunning(true);
  resetMetrics();
  elements.runStatus.className = "run-status running";
  elements.statusTitle.textContent = "Submitting live computation";
  elements.statusDetail.textContent = "The input subset and registered parameters are being validated.";
  startClock();
  try {
    const submitted = await requestJson("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    const job = await pollJob(submitted.status_url);
    renderResult(job.result);
    stopClock(job.server_elapsed_seconds);
    elements.runStatus.className = "run-status completed";
    elements.statusTitle.textContent = "Run completed";
    elements.statusDetail.textContent = "The plot and every metric come from this newly computed result.";
  } catch (error) {
    stopClock();
    elements.runStatus.className = "run-status failed";
    elements.statusTitle.textContent = "Run failed";
    elements.statusDetail.textContent = error.message;
  } finally {
    setRunning(false);
  }
}

function nearestPoint(event) {
  const rect = elements.canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  let nearest = null;
  let bestDistance = 64;
  state.positions.forEach((position, index) => {
    if (!position) return;
    const distance = (x - position[0]) ** 2 + (y - position[1]) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      nearest = index;
    }
  });
  return nearest;
}

elements.canvas.addEventListener("pointermove", (event) => {
  if (!state.result) return;
  const index = nearestPoint(event);
  if (index === null) {
    elements.tooltip.hidden = true;
    return;
  }
  const label = state.result.labels[index];
  elements.tooltip.hidden = false;
  elements.tooltip.style.left = `${Math.min(window.innerWidth - 230, event.clientX + 12)}px`;
  elements.tooltip.style.top = `${Math.max(8, event.clientY - 40)}px`;
  elements.tooltip.textContent = `${state.result.cell_ids[index]} · ${label === -1 ? "noise" : `cluster ${label}`}`;
});
elements.canvas.addEventListener("pointerleave", () => { elements.tooltip.hidden = true; });

async function load() {
  try {
    state.config = await requestJson("/api/config");
    elements.algorithm.innerHTML = Object.entries(state.config.algorithms)
      .map(([value, item]) => `<option value="${escapeHtml(value)}">${escapeHtml(item.label)}</option>`).join("");
    elements.cellCount.innerHTML = state.config.cell_count_options
      .map((count) => `<option value="${count}">${integer(count)} cells</option>`).join("");
    elements.cellCount.value = String(state.config.default_cell_count);
    elements.seed.value = String(state.config.default_seed);
    elements.inputScope.textContent = `${integer(state.config.full_cell_count)} prepared cells · ${integer(state.config.feature_count)} features · fixed subset sampling`;
    setServiceState(true, "Backend ready", `${state.config.threads} CPU workers · ${state.config.sample}`);
    setRunning(false);
    syncDeterministicControl();
    drawEmbedding();
  } catch (error) {
    document.querySelector(".page-shell").hidden = true;
    elements.loadError.hidden = false;
    elements.loadError.textContent = `Could not reach the live computation backend (${error.message}). Start scripts/19_serve_live_demo.py in the documented environment.`;
  }
}

elements.form.addEventListener("submit", runPipeline);
elements.algorithm.addEventListener("change", updateAlgorithmNote);
elements.deterministic.addEventListener("change", syncDeterministicControl);
new ResizeObserver(drawEmbedding).observe(elements.canvas);
load();
