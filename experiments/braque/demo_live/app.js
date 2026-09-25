"use strict";

const PARAM_NAMES = ["n_neighbors", "min_dist", "min_cluster_size"];
const HISTORY_LIMIT = 20;

const state = {
  config: null,
  result: null,
  running: false,
  runStarted: null,
  timer: null,
  positions: [],
  history: [],
  runCounter: 0,
  lastCompleted: null, // {jobId, runNumber, settings}
  changedSet: new Set(),
};

const $ = (id) => document.getElementById(id);
const elements = {
  app: $("app"),
  form: $("run-form"),
  algorithmOptions: $("algorithm-options"),
  cellCount: $("cell-count-select"),
  deterministic: $("deterministic-toggle"),
  seed: $("seed-input"),
  resetButton: $("reset-button"),
  runButton: $("run-button"),
  algorithmNote: $("algorithm-note"),
  inputScope: $("input-scope"),
  serviceState: $("service-state"),
  runStatus: $("run-status"),
  statusTitle: $("status-title"),
  statusDetail: $("status-detail"),
  liveElapsed: $("live-elapsed"),
  plotPanel: document.querySelector(".plot-panel"),
  insights: document.querySelector(".insights"),
  plotTitle: $("plot-title"),
  plotSubtitle: $("plot-subtitle"),
  plotEmpty: $("plot-empty"),
  canvas: $("embedding-canvas"),
  changedToggle: $("changed-toggle"),
  legendChanged: $("legend-changed"),
  legendColourNote: $("legend-colour-note"),
  changeCard: $("change-card"),
  changePercent: $("change-percent"),
  changeCaption: $("change-caption"),
  changeMeter: $("change-meter-fill"),
  changeDetail: $("change-detail"),
  changeSettings: $("change-settings"),
  metricReduction: $("metric-reduction"),
  metricReductionNote: $("metric-reduction-note"),
  metricHdbscan: $("metric-hdbscan"),
  metricClusters: $("metric-clusters"),
  metricNoise: $("metric-noise"),
  metricNoiseCount: $("metric-noise-count"),
  metricFingerprint: $("metric-fingerprint"),
  metricFingerprintNote: $("metric-fingerprint-note"),
  clusterBars: $("cluster-bars"),
  clusterCountNote: $("cluster-count-note"),
  effectiveConfig: $("effective-config"),
  comparisonMethod: $("comparison-method"),
  history: $("run-history"),
  tooltip: $("tooltip"),
  loadError: $("load-error"),
};

const PHASES = {
  queued: ["Queued", "Waiting for the execution slot."],
  preparing_input: ["Selecting cells", "Fixed subset, independent of the algorithm seed."],
  embedding: ["Computing the embedding", "The dimensionality-reduction algorithm is running."],
  hdbscan: ["Clustering with HDBSCAN", "Clustering the new 2D coordinates."],
  comparing: ["Comparing with the previous run", "Matching clusters by maximum overlap."],
};

/* ---------- formatting ---------- */

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

function percent(value, digits = 1) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
}

function integer(value) {
  return Number.isFinite(value) ? (Math.round(value) || 0).toLocaleString("en-US") : "—";
}

function paramText(name, value) {
  if (!Number.isFinite(value)) return "—";
  return name === "min_dist" ? String(Number(value.toFixed(2))) : String(Math.round(value));
}

/* ---------- heat colour for the change share ---------- */

const HEAT_STOPS = [
  [0.0, [255, 250, 240]],
  [0.22, [253, 225, 192]],
  [0.5, [246, 160, 96]],
  [0.78, [228, 96, 46]],
  [1.0, [186, 38, 30]],
];

function heatLevel(share) {
  // 0 → no heat; 50 % or more of the cells changed → full heat.
  const t = Math.min(Math.max(share / 0.5, 0), 1);
  return Math.pow(t, 0.72);
}

function heatRgb(t) {
  for (let i = 1; i < HEAT_STOPS.length; i += 1) {
    const [t1, c1] = HEAT_STOPS[i];
    const [t0, c0] = HEAT_STOPS[i - 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
    }
  }
  return HEAT_STOPS[HEAT_STOPS.length - 1][1];
}

function heatStyle(share) {
  const t = heatLevel(share);
  const [r, g, b] = heatRgb(t);
  const hot = t >= 0.74; // white text only on the darker end of the scale
  const accent = heatRgb(Math.max(t, 0.62));
  return {
    t,
    hot,
    background: `rgb(${r}, ${g}, ${b})`,
    text: hot ? "#ffffff" : `rgb(${Math.round(96 + 40 * t)}, ${Math.round(36 - 8 * t)}, 14)`,
    accent: hot ? "rgba(255, 255, 255, .92)" : `rgb(${accent.join(", ")})`,
    border: `rgba(${accent.join(", ")}, ${0.35 + 0.5 * t})`,
    glow: `0 0 0 ${1 + 2 * t}px rgba(${accent.join(", ")}, ${0.18 + 0.4 * t}), 0 8px ${14 + 16 * t}px rgba(${accent.join(", ")}, ${0.1 + 0.3 * t})`,
  };
}

/* ---------- backend ---------- */

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

/* ---------- controls ---------- */

function paramInputs(name) {
  return { number: $(`${name}-number`), range: $(`${name}-range`), box: document.querySelector(`.param[data-param="${name}"]`) };
}

function clampParam(name, raw) {
  const spec = state.config.parameters[name];
  let value = Number(raw);
  if (!Number.isFinite(value)) value = spec.default;
  value = Math.min(spec.max, Math.max(spec.min, value));
  if (spec.type === "int") value = Math.round(value);
  else value = Math.round(value / spec.step) * spec.step;
  return Number(value.toFixed(4));
}

function setParam(name, raw) {
  const value = clampParam(name, raw);
  const { number, range, box } = paramInputs(name);
  number.value = paramText(name, value);
  range.value = String(value);
  box.classList.toggle("modified", value !== state.config.parameters[name].default);
  return value;
}

function readParam(name) {
  return clampParam(name, paramInputs(name).range.value);
}

function selectedAlgorithm() {
  const checked = elements.algorithmOptions.querySelector("input:checked");
  return checked ? checked.value : Object.keys(state.config.algorithms)[0];
}

function buildControls() {
  const { config } = state;
  elements.algorithmOptions.innerHTML = Object.entries(config.algorithms).map(([value, item], index) => `
    <label><input type="radio" name="algorithm" value="${escapeHtml(value)}" ${index === 0 ? "checked" : ""}><span>${escapeHtml(item.label)}</span></label>`).join("");
  elements.cellCount.innerHTML = config.cell_count_options
    .map((count) => `<option value="${count}">${integer(count)} cells</option>`).join("");
  elements.cellCount.value = String(config.default_cell_count);
  elements.seed.value = String(config.default_seed);

  for (const name of PARAM_NAMES) {
    const spec = config.parameters[name];
    const { number, range } = paramInputs(name);
    for (const input of [number, range]) {
      input.min = String(spec.min);
      input.max = String(spec.max);
      input.step = String(spec.step);
    }
    range.addEventListener("input", () => { setParam(name, range.value); });
    number.addEventListener("change", () => { setParam(name, number.value); });
    number.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault(); // commit the value without starting a run
      setParam(name, number.value);
    });
    setParam(name, spec.default);
  }
  const defaults = PARAM_NAMES.map((name) => `${name} ${paramText(name, config.parameters[name].default)}`).join(" · ");
  elements.resetButton.title = `Reset to ${defaults}`;
  elements.algorithmOptions.addEventListener("change", updateAlgorithmNote);
}

function resetParameters() {
  for (const name of PARAM_NAMES) setParam(name, state.config.parameters[name].default);
}

function updateAlgorithmNote() {
  if (!state.config) return;
  const algorithm = selectedAlgorithm();
  const item = state.config.algorithms[algorithm];
  const deterministic = elements.deterministic.checked;
  let detail;
  if (deterministic && algorithm === "umap_learn") {
    detail = "Fixed random_state; umap-learn runs with n_jobs=1.";
  } else if (deterministic) {
    detail = `Fixed root seed; ibUMAP's deterministic execution path (${state.config.threads} graph workers).`;
  } else {
    detail = `No seed; up to ${state.config.threads} CPU workers.`;
  }
  elements.algorithmNote.innerHTML = `<strong>${escapeHtml(item.label)}.</strong> ${escapeHtml(detail)} ${state.config.model.n_epochs} epochs, spectral init.`;
  elements.runButton.querySelector("span").textContent = `Run ${item.label} + HDBSCAN`;
}

function syncDeterministicControl() {
  elements.seed.disabled = !elements.deterministic.checked || state.running;
  updateAlgorithmNote();
}

function setRunning(running) {
  state.running = running;
  for (const control of elements.form.querySelectorAll("input, select")) control.disabled = running;
  elements.seed.disabled = running || !elements.deterministic.checked;
  elements.resetButton.disabled = running;
  elements.runButton.disabled = running || !state.config;
  elements.runButton.classList.toggle("running", running);
  elements.plotPanel.classList.toggle("stale", running && Boolean(state.result));
  elements.insights.classList.toggle("stale", running && Boolean(state.result));
}

function currentSettings() {
  const deterministic = elements.deterministic.checked;
  return {
    algorithm: selectedAlgorithm(),
    cell_count: Number.parseInt(elements.cellCount.value, 10),
    deterministic,
    seed: deterministic ? Number.parseInt(elements.seed.value, 10) : null,
    n_neighbors: readParam("n_neighbors"),
    min_dist: readParam("min_dist"),
    min_cluster_size: readParam("min_cluster_size"),
  };
}

function settingsFromRequest(request) {
  return {
    algorithm: request.algorithm,
    cell_count: request.cell_count,
    deterministic: request.deterministic,
    seed: request.deterministic ? request.seed : null,
    n_neighbors: request.n_neighbors,
    min_dist: request.min_dist,
    min_cluster_size: request.min_cluster_size,
  };
}

function modeText(settings) {
  return settings.deterministic ? `seed ${settings.seed}` : "unseeded";
}

function settingsDiff(previous, current) {
  const label = (key) => state.config.algorithms[key].label;
  const rows = [
    ["algorithm", label(previous.algorithm), label(current.algorithm)],
    ["cells", integer(previous.cell_count), integer(current.cell_count)],
    ["n_neighbors", paramText("n_neighbors", previous.n_neighbors), paramText("n_neighbors", current.n_neighbors)],
    ["min_dist", paramText("min_dist", previous.min_dist), paramText("min_dist", current.min_dist)],
    ["min_cluster_size", paramText("min_cluster_size", previous.min_cluster_size), paramText("min_cluster_size", current.min_cluster_size)],
    ["mode", modeText(previous), modeText(current)],
  ];
  return rows.filter(([, a, b]) => a !== b);
}

/* ---------- status ---------- */

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

function setStatus(kind, title, detail) {
  elements.runStatus.className = `run-status ${kind}`;
  elements.statusTitle.textContent = title;
  elements.statusDetail.textContent = detail;
}

function renderPhase(phase) {
  const [title, detail] = PHASES[phase] || ["Computing", "The backend is processing this run."];
  setStatus("running", title, detail);
}

/* ---------- plot ---------- */

function clusterColor(label, alpha = 0.78) {
  if (label === -1) return `rgba(130, 139, 143, ${alpha})`;
  const hue = (label * 137.508 + 190) % 360;
  const lightness = 38 + (label % 3) * 6;
  return `hsla(${hue.toFixed(1)}, 58%, ${lightness}%, ${alpha})`;
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
  elements.plotEmpty.hidden = Boolean(state.result);
  if (!state.result) return;

  const coordinates = state.result.coordinates_uint16;
  const labels = state.result.display_labels || state.result.labels;
  const count = labels.length;
  let minX = Infinity; let maxX = -Infinity; let minY = Infinity; let maxY = -Infinity;
  for (let i = 0; i < count; i += 1) {
    const x = coordinates[i * 2]; const y = coordinates[i * 2 + 1];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const pad = 16;
  const spanX = Math.max(maxX - minX, 1);
  const spanY = Math.max(maxY - minY, 1);
  const scale = Math.min((width - pad * 2) / spanX, (height - pad * 2) / spanY);
  const offsetX = (width - spanX * scale) / 2;
  const offsetY = (height - spanY * scale) / 2;
  const base = count <= 2500 ? 1.9 : count <= 5000 ? 1.5 : count <= 10000 ? 1.15 : 0.9;
  const radius = base * Math.min(1.35, Math.max(0.85, Math.min(width, height) / 620));

  for (let i = 0; i < count; i += 1) {
    state.positions[i] = [
      offsetX + (coordinates[i * 2] - minX) * scale,
      offsetY + (maxY - coordinates[i * 2 + 1]) * scale,
    ];
  }

  const highlight = elements.changedToggle.checked && state.changedSet.size > 0;
  const dot = (index, fill, r = radius) => {
    const [x, y] = state.positions[index];
    context.beginPath();
    context.arc(x, y, r, 0, Math.PI * 2);
    context.fillStyle = fill;
    context.fill();
  };

  if (highlight) {
    for (let i = 0; i < count; i += 1) {
      if (!state.changedSet.has(i)) dot(i, labels[i] === -1 ? "rgba(150, 156, 160, .16)" : clusterColor(labels[i], 0.16));
    }
    context.lineWidth = 0.9;
    context.strokeStyle = "rgba(22, 36, 45, .85)";
    for (const i of state.changedSet) {
      const [x, y] = state.positions[i];
      context.beginPath();
      context.arc(x, y, radius + 0.9, 0, Math.PI * 2);
      context.fillStyle = labels[i] === -1 ? "rgba(130, 139, 143, .95)" : clusterColor(labels[i], 0.95);
      context.fill();
      context.stroke();
    }
    return;
  }

  for (let i = 0; i < count; i += 1) if (labels[i] === -1) dot(i, clusterColor(-1, 0.34));
  for (let i = 0; i < count; i += 1) if (labels[i] !== -1) dot(i, clusterColor(labels[i], 0.7));
}

/* ---------- result panels ---------- */

function renderClusterBars(result) {
  const rows = result.cluster_sizes;
  const maximum = Math.max(...rows.map((row) => row.count), 1);
  elements.clusterBars.innerHTML = rows.map((row) => {
    const name = row.label === -1 ? "Noise (−1)" : `Cluster ${row.label}`;
    return `<div class="cluster-row">
      <span><i style="background:${clusterColor(row.label, 0.9)}"></i>${name}</span>
      <div><b style="width:${(row.count / maximum) * 100}%;background:${clusterColor(row.label, 0.8)}"></b></div>
      <strong>${integer(row.count)}</strong>
    </div>`;
  }).join("");
  const clusters = rows.filter((row) => row.label !== -1).length;
  elements.clusterCountNote.textContent = `${clusters} clusters${rows.some((row) => row.label === -1) ? " + noise" : ""}`;
}

function renderEffective(result) {
  const effective = result.effective;
  const software = state.config.software || {};
  const entries = [
    ["Engine", effective.engine],
    ["Mode", effective.deterministic ? `deterministic · seed ${effective.random_state}` : "unseeded"],
    ["CPU workers", effective.n_jobs],
    ["Cells", `${integer(result.input.cell_count)} × ${integer(result.input.feature_count)} features`],
    ["UMAP objective", `n_neighbors ${effective.n_neighbors} · min_dist ${paramText("min_dist", effective.min_dist)} · ${effective.n_epochs} epochs`],
    ["HDBSCAN", `min_cluster_size ${effective.hdbscan_min_cluster_size} · min_samples ${effective.hdbscan_min_samples} · ε ${effective.hdbscan_cluster_selection_epsilon}`],
    ["Timing", `reduction ${seconds(result.timing_seconds.reduction)} · HDBSCAN ${seconds(result.timing_seconds.hdbscan)} · total ${seconds(result.timing_seconds.total)}`],
    ["Fingerprint", result.summary.result_fingerprint],
    ["Software", `umap-learn ${software.umap_learn} · hdbscan ${software.hdbscan} · ibumap ${software.ibumap}`],
  ];
  elements.effectiveConfig.innerHTML = entries.map(([term, value]) => `<div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
}

function applyChangeStyle(style) {
  const card = elements.changeCard;
  card.style.background = style ? style.background : "";
  card.style.color = style ? style.text : "";
  card.style.borderColor = style ? style.border : "";
  card.style.boxShadow = style ? style.glow : "";
  card.classList.toggle("hot", Boolean(style && style.hot));
  elements.changePercent.style.color = style ? style.text : "";
  elements.changeCard.querySelector(".eyebrow").style.color = style && !style.hot ? style.text : "";
  elements.changeMeter.style.color = style ? (style.hot ? "#ffffff" : style.text) : "";
}

function renderComparison(result, previousRun, currentSettingsValue) {
  const comparison = result.comparison;
  const card = elements.changeCard;
  elements.changeSettings.textContent = "";
  if (!comparison) {
    card.dataset.state = "empty";
    applyChangeStyle(null);
    elements.changePercent.textContent = "—";
    elements.changeCaption.textContent = "of cells changed cluster";
    elements.changeMeter.style.width = "0";
    elements.changeDetail.textContent = "Available from the second run.";
    return;
  }
  if (!comparison.available) {
    card.dataset.state = "unavailable";
    applyChangeStyle(null);
    elements.changePercent.textContent = "—";
    elements.changeMeter.style.width = "0";
    elements.changeDetail.textContent = comparison.reason || "The previous run cannot be compared.";
    return;
  }

  const share = comparison.assignment_disagreement;
  const zero = comparison.changed_count === 0;
  card.dataset.state = zero ? "zero" : "value";
  applyChangeStyle(zero ? null : heatStyle(share));
  elements.changePercent.textContent = percent(share);
  elements.changeCaption.textContent = zero ? "identical partition" : "of cells changed cluster";
  elements.changeMeter.style.width = `${Math.min(100, share * 100)}%`;

  const shared = comparison.shared_cell_count;
  const partial = shared !== comparison.current_cell_count || shared !== comparison.previous_cell_count;
  const scope = partial ? `${integer(shared)} shared cells` : `${integer(shared)} cells`;
  elements.changeDetail.innerHTML =
    `${integer(comparison.changed_count)} of ${scope} vs run #${previousRun.runNumber} · ARI ${comparison.ari_including_noise.toFixed(3)}<br>`
    + `between clusters ${percent(comparison.cluster_to_cluster_disagreement)} · to/from noise ${percent(comparison.noise_status_disagreement)}`;

  const diff = settingsDiff(previousRun.settings, currentSettingsValue);
  if (!diff.length) {
    elements.changeSettings.textContent = currentSettingsValue.deterministic
      ? `Same settings and seed as run #${previousRun.runNumber}.`
      : `Same settings as run #${previousRun.runNumber} (unseeded rerun).`;
  } else {
    elements.changeSettings.textContent = `Changed: ${diff.map(([name, a, b]) => `${name} ${a} → ${b}`).join(" · ")}`;
  }

  card.classList.remove("flash");
  void card.offsetWidth; // restart the animation
  card.classList.add("flash");
}

function renderHistory() {
  if (!state.history.length) {
    elements.history.innerHTML = '<tr><td colspan="12" class="empty-copy">No runs yet.</td></tr>';
    return;
  }
  elements.history.innerHTML = state.history.map((item) => {
    let delta = '<span class="empty-copy">—</span>';
    if (Number.isFinite(item.change)) {
      const style = item.change === 0 ? null : heatStyle(item.change);
      const css = style ? `background:${style.background};color:${style.text}` : "background:#eef6f1;color:#417c64";
      delta = `<span class="delta" style="${css}">${percent(item.change)}</span> <small>vs #${item.comparedWith}</small>`;
    }
    const s = item.settings;
    return `<tr>
      <td><span class="run-index">${item.runNumber}</span></td>
      <td>${escapeHtml(state.config.algorithms[s.algorithm].label)}</td>
      <td>${escapeHtml(modeText(s))}</td>
      <td class="num">${integer(s.cell_count)}</td>
      <td class="num">${paramText("n_neighbors", s.n_neighbors)}</td>
      <td class="num">${paramText("min_dist", s.min_dist)}</td>
      <td class="num">${paramText("min_cluster_size", s.min_cluster_size)}</td>
      <td class="num">${seconds(item.runtime)}</td>
      <td class="num">${integer(item.clusters)}</td>
      <td class="num">${percent(item.noise)}</td>
      <td class="num">${delta}</td>
      <td><code>${escapeHtml(item.fingerprint)}</code></td>
    </tr>`;
  }).join("");
}

function renderResult(job) {
  const result = job.result;
  const previousRun = state.lastCompleted;
  const settings = settingsFromRequest(result.request);
  state.runCounter += 1;
  const runNumber = state.runCounter;
  state.result = result;
  const comparison = result.comparison && result.comparison.available ? result.comparison : null;
  state.changedSet = new Set(comparison ? comparison.changed_indices : []);

  const label = state.config.algorithms[result.request.algorithm].label;
  elements.metricReduction.textContent = seconds(result.timing_seconds.reduction);
  elements.metricReductionNote.textContent = `${label} fit_transform`;
  elements.metricHdbscan.textContent = seconds(result.timing_seconds.hdbscan);
  elements.metricClusters.textContent = integer(result.summary.cluster_count_excluding_noise);
  elements.metricNoise.textContent = percent(result.summary.noise_fraction);
  elements.metricNoiseCount.textContent = `${integer(result.summary.noise_count)} of ${integer(result.input.cell_count)} cells`;
  const fingerprint = result.summary.result_fingerprint.slice(0, 10);
  elements.metricFingerprint.textContent = fingerprint;
  const sameFingerprint = previousRun && previousRun.fingerprint === fingerprint;
  elements.metricFingerprintNote.textContent = previousRun ? (sameFingerprint ? "identical to previous run" : "differs from previous run") : "embedding + labels";
  elements.metricFingerprintNote.classList.toggle("same", Boolean(sameFingerprint));

  elements.plotTitle.textContent = `Run #${runNumber} · ${label} → HDBSCAN`;
  elements.plotSubtitle.textContent = [
    `${integer(result.input.cell_count)} cells`,
    `n_neighbors ${paramText("n_neighbors", result.effective.n_neighbors)}`,
    `min_dist ${paramText("min_dist", result.effective.min_dist)}`,
    `min_cluster_size ${result.effective.hdbscan_min_cluster_size}`,
    modeText(settings),
  ].join(" · ");

  elements.changedToggle.disabled = state.changedSet.size === 0;
  if (elements.changedToggle.disabled) elements.changedToggle.checked = false;
  elements.changedToggle.closest("label").title = comparison
    ? `${integer(state.changedSet.size)} cells changed vs run #${previousRun.runNumber}`
    : "Available from the second run";
  elements.legendChanged.hidden = !elements.changedToggle.checked;
  elements.legendColourNote.hidden = !comparison;

  renderComparison(result, previousRun, settings);
  renderClusterBars(result);
  renderEffective(result);
  drawEmbedding();

  state.history.unshift({
    runNumber,
    settings,
    runtime: result.timing_seconds.reduction,
    clusters: result.summary.cluster_count_excluding_noise,
    noise: result.summary.noise_fraction,
    change: comparison ? comparison.assignment_disagreement : null,
    comparedWith: comparison ? previousRun.runNumber : null,
    fingerprint,
  });
  state.history = state.history.slice(0, HISTORY_LIMIT);
  renderHistory();
  state.lastCompleted = { jobId: job.job_id, runNumber, settings, fingerprint };
}

/* ---------- run ---------- */

async function pollJob(statusUrl) {
  while (true) {
    const job = await requestJson(statusUrl);
    if (job.status === "completed") return job;
    if (job.status === "failed") throw new Error(`${job.error.type}: ${job.error.message}`);
    renderPhase(job.phase);
    await new Promise((resolve) => window.setTimeout(resolve, 500));
  }
}

async function runPipeline(event) {
  event.preventDefault();
  if (state.running || !state.config) return;
  for (const name of PARAM_NAMES) setParam(name, paramInputs(name).number.value);
  const settings = currentSettings();
  if (settings.deterministic && (!Number.isInteger(settings.seed) || settings.seed < 0 || settings.seed > 2147483647)) {
    elements.seed.focus();
    setStatus("failed", "Invalid seed", "Use an integer between 0 and 2147483647.");
    return;
  }
  const request = {
    ...settings,
    seed: settings.deterministic ? settings.seed : null,
    compare_to: state.lastCompleted ? state.lastCompleted.jobId : null,
  };
  setRunning(true);
  setStatus("running", "Submitting", "Validating the request.");
  startClock();
  try {
    const submitted = await requestJson("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    const job = await pollJob(submitted.status_url);
    renderResult(job);
    stopClock(job.server_elapsed_seconds);
    setStatus("completed", `Run #${state.runCounter} completed`, "Plot and metrics come from this run.");
  } catch (error) {
    stopClock();
    setStatus("failed", "Run failed", error.message);
  } finally {
    setRunning(false);
  }
}

/* ---------- tooltip ---------- */

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
  const labels = state.result.display_labels || state.result.labels;
  const label = labels[index];
  const changed = state.changedSet.has(index) ? " · changed vs previous run" : "";
  elements.tooltip.hidden = false;
  elements.tooltip.style.left = `${Math.min(window.innerWidth - 270, event.clientX + 12)}px`;
  elements.tooltip.style.top = `${Math.max(8, event.clientY - 40)}px`;
  elements.tooltip.textContent = `${state.result.cell_ids[index]} · ${label === -1 ? "noise" : `cluster ${label}`}${changed}`;
});
elements.canvas.addEventListener("pointerleave", () => { elements.tooltip.hidden = true; });

/* ---------- start ---------- */

async function load() {
  try {
    state.config = await requestJson("/api/config");
    buildControls();
    elements.inputScope.innerHTML = `${integer(state.config.full_cell_count)} prepared cells · ${integer(state.config.feature_count)} features<br>fixed subset sampling (seed ${state.config.input.subset_seed})`;
    if (state.config.comparison_method) elements.comparisonMethod.textContent = state.config.comparison_method;
    setServiceState(true, "Backend ready", `${state.config.threads} CPU workers · sample ${state.config.sample}`);
    setRunning(false);
    syncDeterministicControl();
    drawEmbedding();
  } catch (error) {
    elements.app.hidden = true;
    document.getElementById("history-section").hidden = true;
    elements.loadError.hidden = false;
    elements.loadError.textContent = `Could not reach the live computation backend (${error.message}). Start scripts/09_serve_live_demo.py in the documented environment.`;
  }
}

elements.form.addEventListener("submit", runPipeline);
elements.deterministic.addEventListener("change", syncDeterministicControl);
elements.resetButton.addEventListener("click", resetParameters);
elements.changedToggle.addEventListener("change", () => {
  elements.legendChanged.hidden = !elements.changedToggle.checked;
  drawEmbedding();
});
new ResizeObserver(drawEmbedding).observe(elements.canvas);
load();
