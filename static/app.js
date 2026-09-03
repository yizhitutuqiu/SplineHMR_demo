const state = {
  sequences: [],
  meta: null,
  stroke: [],
  sampled: [],
  drawing: false,
  currentFrame: 0,
  lastEdit: null,
};

const els = {
  sequence: document.getElementById("sequenceSelect"),
  joint: document.getElementById("jointSelect"),
  tpose: document.getElementById("tposeView"),
  jointLabel: document.getElementById("jointLabel"),
  timing: document.getElementById("timingSelect"),
  frameStart: document.getElementById("frameStart"),
  frameEnd: document.getElementById("frameEnd"),
  reset: document.getElementById("resetStrokeBtn"),
  build: document.getElementById("buildBtn"),
  video: document.getElementById("video"),
  wrap: document.getElementById("videoWrap"),
  canvas: document.getElementById("overlay"),
  playPause: document.getElementById("playPauseBtn"),
  slider: document.getElementById("frameSlider"),
  frameLabel: document.getElementById("frameLabel"),
  status: document.getElementById("status"),
  report: document.getElementById("report"),
  runOpt: document.getElementById("runOptCheck"),
  maxIter: document.getElementById("maxIter"),
  render: document.getElementById("renderCheck"),
  kpPreviewCard: document.getElementById("kpPreviewCard"),
  kpPreviewVideo: document.getElementById("kpPreviewVideo"),
  kpPreviewLabel: document.getElementById("kpPreviewLabel"),
  renderPreviewCard: document.getElementById("renderPreviewCard"),
  renderPreviewVideo: document.getElementById("renderPreviewVideo"),
  renderPreviewLabel: document.getElementById("renderPreviewLabel"),
};

const ctx = els.canvas.getContext("2d");


function hideKeypointPreview() {
  if (els.kpPreviewVideo) {
    els.kpPreviewVideo.pause();
    els.kpPreviewVideo.removeAttribute("src");
    els.kpPreviewVideo.load();
  }
  if (els.kpPreviewCard) els.kpPreviewCard.classList.add("hidden");
  if (els.kpPreviewLabel) els.kpPreviewLabel.textContent = "not built yet";
}

function showKeypointPreview(edit) {
  const url = edit?.urls?.keypoints_2d_preview;
  if (!url || !els.kpPreviewVideo || !els.kpPreviewCard) return;
  els.kpPreviewVideo.src = `${url}?t=${Date.now()}`;
  els.kpPreviewVideo.load();
  els.kpPreviewVideo.play().catch(() => {});
  els.kpPreviewCard.classList.remove("hidden");
  if (els.kpPreviewLabel) {
    els.kpPreviewLabel.textContent = `${edit.sequence} · ${edit.joint} · ${edit.num_frames} frames`;
  }
}

function hideRenderPreview() {
  if (els.renderPreviewVideo) {
    els.renderPreviewVideo.pause();
    els.renderPreviewVideo.removeAttribute("src");
    els.renderPreviewVideo.load();
  }
  if (els.renderPreviewCard) els.renderPreviewCard.classList.add("hidden");
  if (els.renderPreviewLabel) els.renderPreviewLabel.textContent = "not rendered yet";
}

function showRenderPreview(result) {
  const url = result?.render_urls?.annotated_overlay_compare_video
    || result?.render_urls?.overlay_compare_video
    || result?.render_urls?.compare_video
    || result?.render_urls?.after_video;
  if (!url || !els.renderPreviewVideo || !els.renderPreviewCard) return false;
  els.renderPreviewVideo.src = `${url}?t=${Date.now()}`;
  els.renderPreviewVideo.load();
  els.renderPreviewVideo.play().catch(() => {});
  els.renderPreviewCard.classList.remove("hidden");
  els.renderPreviewCard.scrollIntoView({behavior: "smooth", block: "nearest"});
  if (els.renderPreviewLabel) {
    els.renderPreviewLabel.textContent = `${result.sequence} · ${result.num_frames} frames`;
  }
  return true;
}

function setStatus(text) {
  els.status.textContent = text;
}

function setReport(obj) {
  els.report.textContent = JSON.stringify(obj, null, 2);
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok || data.status === "error") {
    const req = data.request_id ? `[request_id=${data.request_id}] ` : "";
    const msg = data.detail || data.error || `Request failed: ${url}`;
    throw new Error(`${req}${msg}`);
  }
  return data;
}

function resizeCanvasToVideo() {
  const meta = state.meta?.video;
  if (!meta) return;
  els.canvas.width = meta.width;
  els.canvas.height = meta.height;
  const aspect = meta.width / Math.max(1, meta.height);
  const maxByViewportHeight = Math.max(220, Math.floor(window.innerHeight * 0.72 * aspect));
  els.wrap.style.setProperty("--video-aspect", `${meta.width} / ${meta.height}`);
  els.wrap.style.setProperty("--video-max-width", `${maxByViewportHeight}px`);
  drawOverlay();
}

function currentFrameIndex() {
  const fps = state.meta?.video?.fps || 30;
  const n = state.meta?.num_frames || 1;
  return Math.max(0, Math.min(n - 1, Math.round(els.video.currentTime * fps)));
}

function pointFromEvent(ev) {
  const rect = els.canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) / rect.width;
  const y = (ev.clientY - rect.top) / rect.height;
  return {
    norm: [Math.max(0, Math.min(1, x)), Math.max(0, Math.min(1, y))],
    px: [x * els.canvas.width, y * els.canvas.height],
  };
}

function normToPx(p) {
  return [p[0] * els.canvas.width, p[1] * els.canvas.height];
}

function drawPolyline(points, color, width = 3, close = false) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
  if (close) ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

function speedColor(value01) {
  const v = Math.max(0, Math.min(1, value01));
  const hue = 220 - 220 * v;
  return `hsl(${hue}, 88%, 55%)`;
}

function drawSpeedPolyline(points, width = 5) {
  if (!points || points.length < 2) return;
  const speeds = [];
  for (let i = 1; i < points.length; i++) {
    const dx = points[i][0] - points[i - 1][0];
    const dy = points[i][1] - points[i - 1][1];
    speeds.push(Math.hypot(dx, dy));
  }
  const sorted = [...speeds].sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length * 0.05)] ?? 0;
  const hi = sorted[Math.floor(sorted.length * 0.95)] ?? 1;
  const denom = Math.max(1e-6, hi - lo);
  ctx.save();
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  for (let i = 1; i < points.length; i++) {
    ctx.strokeStyle = speedColor((speeds[i - 1] - lo) / denom);
    ctx.beginPath();
    ctx.moveTo(points[i - 1][0], points[i - 1][1]);
    ctx.lineTo(points[i][0], points[i][1]);
    ctx.stroke();
  }
  ctx.restore();
}

function drawPoint(p, color, r = 5, stroke = "rgba(0,0,0,.55)") {
  ctx.save();
  ctx.beginPath();
  ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = stroke;
  ctx.stroke();
  ctx.restore();
}

function selectedJointName() {
  return els.joint.value || "left_wrist";
}

function selectedJointIndex() {
  const name = selectedJointName();
  const joint = state.meta?.joints?.find((j) => j.name === name);
  return joint ? Number(joint.index) : 9;
}

function selectedJointLabel() {
  const idx = selectedJointIndex();
  const joint = state.meta?.joints?.find((j) => Number(j.index) === idx);
  return joint ? `${joint.label} (${joint.index})` : `Joint ${idx}`;
}

function setSelectedJoint(nameOrIndex) {
  if (!state.meta) return;
  let joint = null;
  if (typeof nameOrIndex === "number") {
    joint = state.meta.joints.find((j) => Number(j.index) === nameOrIndex);
  } else {
    joint = state.meta.joints.find((j) => j.name === nameOrIndex);
  }
  if (!joint) return;
  els.joint.value = joint.name;
  if (els.jointLabel) els.jointLabel.textContent = `Selected: ${joint.label} (${joint.index})`;
  updateTposeActive();
  drawOverlay();
}

function drawSkeleton(frame) {
  const meta = state.meta;
  if (!meta || !meta.keypoints_2d?.[frame]) return;
  const kps = meta.keypoints_2d[frame];
  ctx.save();
  ctx.globalAlpha = 0.8;
  ctx.strokeStyle = "rgba(255,255,255,.82)";
  ctx.lineWidth = 2;
  for (const [a, b] of meta.edges || []) {
    const pa = kps[a], pb = kps[b];
    if (!pa || !pb) continue;
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
    ctx.stroke();
  }
  for (let i = 0; i < kps.length; i++) {
    drawPoint(kps[i], i === selectedJointIndex() ? "#27ae60" : "rgba(255,255,255,.85)", i === selectedJointIndex() ? 7 : 3);
  }
  ctx.restore();
}

function drawOriginalTrajectory() {
  const meta = state.meta;
  if (!meta) return;
  const j = selectedJointIndex();
  const start = Math.max(0, Number(els.frameStart.value || 0));
  const endVal = els.frameEnd.value === "" ? meta.num_frames : Number(els.frameEnd.value);
  const end = Math.max(start + 2, Math.min(meta.num_frames, endVal));
  const pts = [];
  for (let t = start; t < end; t++) {
    const p = meta.keypoints_2d?.[t]?.[j];
    if (p) pts.push(p);
  }
  drawSpeedPolyline(pts, 5);
  if (pts.length) {
    drawPoint(pts[0], "#2f80ed", 6);
    drawPoint(pts[pts.length - 1], "#eb5757", 6);
  }
}

function drawUserStroke() {
  const pts = state.stroke.map(normToPx);
  drawPolyline(pts, "#f2c94c", 4);
  if (pts.length) {
    drawPoint(pts[0], "#f2c94c", 6);
    drawPoint(pts[pts.length - 1], "#f2994a", 6);
  }
}

function drawSampledTargets() {
  if (!state.sampled?.length) return;
  drawPolyline(state.sampled, "rgba(39,174,96,.75)", 3);
  const stride = Math.max(1, Math.floor(state.sampled.length / 40));
  for (let i = 0; i < state.sampled.length; i += stride) {
    drawPoint(state.sampled[i], "white", 3, "rgba(39,174,96,.9)");
  }
}

function drawOverlay() {
  if (!els.canvas.width || !els.canvas.height) return;
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (!state.meta) return;
  const frame = currentFrameIndex();
  drawSkeleton(frame);
  drawOriginalTrajectory();
  drawUserStroke();
  drawSampledTargets();
  const p = state.meta.keypoints_2d?.[frame]?.[selectedJointIndex()];
  if (p) drawPoint(p, "#27ae60", 9);
}

const TPOSE_POS = {
  0: [130, 45],
  1: [118, 38],
  2: [142, 38],
  3: [105, 42],
  4: [155, 42],
  5: [80, 95],
  6: [180, 95],
  7: [45, 100],
  8: [215, 100],
  9: [20, 105],
  10: [240, 105],
  11: [102, 190],
  12: [158, 190],
  13: [98, 255],
  14: [162, 255],
  15: [95, 315],
  16: [165, 315],
};

function renderTpose() {
  if (!els.tpose || !state.meta) return;
  els.tpose.innerHTML = "";
  const ns = "http://www.w3.org/2000/svg";
  for (const [a, b] of state.meta.edges || []) {
    if (!TPOSE_POS[a] || !TPOSE_POS[b]) continue;
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", TPOSE_POS[a][0]);
    line.setAttribute("y1", TPOSE_POS[a][1]);
    line.setAttribute("x2", TPOSE_POS[b][0]);
    line.setAttribute("y2", TPOSE_POS[b][1]);
    line.setAttribute("class", "tpose-bone");
    els.tpose.appendChild(line);
  }
  for (const joint of state.meta.joints || []) {
    const idx = Number(joint.index);
    const pos = TPOSE_POS[idx];
    if (!pos) continue;
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("cx", pos[0]);
    circle.setAttribute("cy", pos[1]);
    circle.setAttribute("r", idx === selectedJointIndex() ? 10 : 8);
    circle.setAttribute("class", "tpose-joint");
    circle.dataset.jointName = joint.name;
    circle.dataset.jointIndex = String(idx);
    circle.setAttribute("tabindex", "0");
    circle.setAttribute("role", "button");
    circle.setAttribute("aria-label", joint.label);
    circle.addEventListener("click", () => setSelectedJoint(joint.name));
    circle.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") setSelectedJoint(joint.name);
    });
    els.tpose.appendChild(circle);
    if ([0, 5, 6, 9, 10, 15, 16].includes(idx)) {
      const text = document.createElementNS(ns, "text");
      text.setAttribute("x", pos[0]);
      text.setAttribute("y", pos[1] - 13);
      text.setAttribute("class", "tpose-joint-label");
      text.textContent = String(idx);
      els.tpose.appendChild(text);
    }
  }
  updateTposeActive();
}

function updateTposeActive() {
  if (!els.tpose) return;
  const idx = selectedJointIndex();
  els.tpose.querySelectorAll(".tpose-joint").forEach((node) => {
    const active = Number(node.dataset.jointIndex) === idx;
    node.classList.toggle("active", active);
    node.setAttribute("r", active ? "10" : "8");
  });
}

async function loadSequences() {
  const data = await fetchJson("/api/sequences");
  state.sequences = data.sequences || [];
  els.sequence.innerHTML = "";
  for (const seq of state.sequences) {
    const opt = document.createElement("option");
    opt.value = seq.name;
    opt.textContent = `${seq.name} (${seq.frame_count}f, ${seq.width}×${seq.height})`;
    els.sequence.appendChild(opt);
  }
  if (!state.sequences.length) {
    setStatus("No sequences found. Expected SplineHMR/inputs/<seq>/0_input_video.mp4 and hmr4d_results.pt.");
    return;
  }
  await loadSequence(els.sequence.value || state.sequences[0].name);
}

async function loadSequence(name) {
  setStatus(`Loading ${name} metadata and SMPL reprojection...`);
  state.stroke = [];
  state.sampled = [];
  state.lastEdit = null;
  hideKeypointPreview();
  hideRenderPreview();
  const data = await fetchJson(`/api/sequence/${encodeURIComponent(name)}/meta`);
  state.meta = data.meta;
  els.video.src = state.meta.video_url;
  els.video.load();
  els.slider.min = 0;
  els.slider.max = Math.max(0, state.meta.num_frames - 1);
  els.slider.value = 0;
  els.frameEnd.placeholder = String(state.meta.num_frames);
  els.joint.innerHTML = "";
  for (const j of state.meta.joints) {
    const opt = document.createElement("option");
    opt.value = j.name;
    opt.textContent = `${j.label} (${j.index})`;
    els.joint.appendChild(opt);
  }
  els.joint.value = "left_wrist";
  renderTpose();
  setSelectedJoint("left_wrist");
  resizeCanvasToVideo();
  setStatus(`${name} loaded. Trajectory source: ${state.meta.trajectory_source}${state.meta.warning ? " — " + state.meta.warning : ""}`);
}

async function buildEdit() {
  if (!state.meta) return;
  if (state.stroke.length < 2) {
    setStatus("Please draw a trajectory first.");
    return;
  }
  els.build.disabled = true;
  setStatus("Building edited Spline-Opt keypoints...");
  try {
    const payload = {
      sequence: state.meta.name,
      joint: els.joint.value,
      timing_mode: els.timing.value,
      frame_start: Number(els.frameStart.value || 0),
      frame_end: els.frameEnd.value === "" ? null : Number(els.frameEnd.value),
      stroke_points_norm: state.stroke,
      trajectory_source: "smpl_reproj",
    };
    const data = await fetchJson("/api/edit", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    state.lastEdit = data.edit;
    hideRenderPreview();
    setReport(data.edit.timing_report);

    state.sampled = data.edit.sampled_target_trajectory_px || [];
    drawOverlay();
    showKeypointPreview(data.edit);

    setStatus(`Edit package written: ${data.edit.paths.output_dir}`);
    if (els.runOpt.checked) {
      await runSplineOpt(data.edit.paths.edit_request);
    }
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
  } finally {
    els.build.disabled = false;
  }
}

async function runSplineOpt(editRequestPath) {
  setStatus("Running Spline-Opt. This may take a while...");
  const data = await fetchJson("/api/run_spline_opt", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      edit_request_path: editRequestPath,
      max_iter: els.maxIter.value === "" ? "default" : Number(els.maxIter.value || 60),
      render: Boolean(els.render.checked),
      device: "cuda",
    }),
  });
  const shown = showRenderPreview(data.result);
  setStatus(shown ? `Spline-Opt done. Render result is shown below.` : `Spline-Opt done: ${data.result.output_dir}`);
  setReport(data.result);
}

els.sequence.addEventListener("change", () => loadSequence(els.sequence.value));
els.joint.addEventListener("change", () => setSelectedJoint(els.joint.value));
els.frameStart.addEventListener("change", drawOverlay);
els.frameEnd.addEventListener("change", drawOverlay);
els.reset.addEventListener("click", () => {
  state.stroke = [];
  state.sampled = [];
  state.lastEdit = null;
  setReport({});
  hideKeypointPreview();
  hideRenderPreview();
  drawOverlay();
});
els.build.addEventListener("click", buildEdit);
els.playPause.addEventListener("click", () => {
  if (els.video.paused) els.video.play();
  else els.video.pause();
});
els.slider.addEventListener("input", () => {
  const fps = state.meta?.video?.fps || 30;
  els.video.currentTime = Number(els.slider.value) / fps;
  drawOverlay();
});
els.video.addEventListener("loadedmetadata", resizeCanvasToVideo);
els.video.addEventListener("timeupdate", () => {
  const f = currentFrameIndex();
  els.slider.value = f;
  els.frameLabel.textContent = `frame ${f}`;
  drawOverlay();
});

els.canvas.addEventListener("pointerdown", (ev) => {
  ev.preventDefault();
  state.drawing = true;
  state.sampled = [];
  state.stroke.push(pointFromEvent(ev).norm);
  els.canvas.setPointerCapture(ev.pointerId);
  drawOverlay();
});

els.canvas.addEventListener("pointermove", (ev) => {
  if (!state.drawing) return;
  ev.preventDefault();
  state.stroke.push(pointFromEvent(ev).norm);
  drawOverlay();
});

function stopDrawing(ev) {
  if (!state.drawing) return;
  state.drawing = false;
  try { els.canvas.releasePointerCapture(ev.pointerId); } catch (_) {}
  drawOverlay();
}

els.canvas.addEventListener("pointerup", stopDrawing);
els.canvas.addEventListener("pointercancel", stopDrawing);
window.addEventListener("resize", () => {
  resizeCanvasToVideo();
  renderTpose();
});

loadSequences().catch((err) => {
  console.error(err);
  setStatus(`Failed to initialize: ${err.message}`);
});

