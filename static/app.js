const state = {
  sequences: [],
  meta: null,
  jointEdits: {},
  drawing: false,
  currentFrame: 0,
  lastEdit: null,
  lastRenderResult: null,
  selectedJointName: null,
};

const els = {
  sequence: document.getElementById("sequenceSelect"),
  joint: document.getElementById("jointSelect"),
  tpose: document.getElementById("tposeView"),
  jointLabel: document.getElementById("jointLabel"),
  timing: document.getElementById("timingSelect"),
  keyframeMode: document.getElementById("keyframeModeCheck"),
  interpolationLabel: document.getElementById("interpolationLabel"),
  interpolation: document.getElementById("interpolationSelect"),
  keyframePanel: document.getElementById("keyframePanel"),
  keyframeList: document.getElementById("keyframeList"),
  frameStart: document.getElementById("frameStart"),
  frameEnd: document.getElementById("frameEnd"),
  reset: document.getElementById("resetStrokeBtn"),
  build: document.getElementById("buildBtn"),
  displayVideo: document.getElementById("displayVideo"),
  displayWrap: document.getElementById("displayVideoWrap"),
  displayCanvas: document.getElementById("displayOverlay"),
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
  bsDegree: document.getElementById("bsDegree"),
  bsMPerT: document.getElementById("bsMPerT"),
  bsConfThr: document.getElementById("bsConfThr"),
  bsConfPower: document.getElementById("bsConfPower"),
  bsPriorBody: document.getElementById("bsPriorBody"),
  bsSmoothW: document.getElementById("bsSmoothW"),
  bsLr: document.getElementById("bsLr"),
  bsLineSearch: document.getElementById("bsLineSearch"),
  bsLearnKnots: document.getElementById("bsLearnKnots"),
  bsKnotMinGap: document.getElementById("bsKnotMinGap"),
  bsKnotPosW: document.getElementById("bsKnotPosW"),
  bsKnotGapW: document.getElementById("bsKnotGapW"),
  kpPreviewCard: document.getElementById("kpPreviewCard"),
  kpPreviewVideo: document.getElementById("kpPreviewVideo"),
  kpPreviewLabel: document.getElementById("kpPreviewLabel"),
  kpPreviewDownload: document.getElementById("kpPreviewDownload"),
  renderPreviewCard: document.getElementById("renderPreviewCard"),
  renderPreviewVideo: document.getElementById("renderPreviewVideo"),
  renderPreviewLabel: document.getElementById("renderPreviewLabel"),
  renderPreviewDownload: document.getElementById("renderPreviewDownload"),
  render2dOverlay: document.getElementById("render2dOverlayCheck"),
  renderDownloads: document.getElementById("renderDownloads"),
};

const ctx = els.canvas.getContext("2d");
const displayCtx = els.displayCanvas.getContext("2d");

function filenameFromUrl(url, fallback = "splinehmr_video.mp4") {
  try {
    const clean = new URL(url, window.location.href).pathname;
    return clean.split("/").filter(Boolean).pop() || fallback;
  } catch (_) {
    return fallback;
  }
}

function setDownloadLink(el, url, filename) {
  if (!el) return;
  if (!url) {
    el.classList.add("hidden");
    el.removeAttribute("href");
    return;
  }
  el.href = url;
  el.download = filename || filenameFromUrl(url);
  el.classList.remove("hidden");
}

function renderDownloadButtons(container, urls) {
  if (!container) return;
  container.innerHTML = "";
  const labels = [
    ["annotated_overlay_compare_video", "Annotated"],
    ["overlay_compare_video", "Overlay"],
    ["before_video", "Before"],
    ["after_video", "After"],
    ["compare_video", "Side-by-side"],
  ];
  let n = 0;
  for (const [key, label] of labels) {
    const url = urls?.[key];
    if (!url) continue;
    const a = document.createElement("a");
    a.className = "download-btn small";
    a.href = url;
    a.download = filenameFromUrl(url, `${key}.mp4`);
    a.textContent = `Download ${label}`;
    container.appendChild(a);
    n += 1;
  }
  container.classList.toggle("hidden", n === 0);
}

function hideKeypointPreview() {
  if (els.kpPreviewVideo) {
    els.kpPreviewVideo.pause();
    els.kpPreviewVideo.removeAttribute("src");
    els.kpPreviewVideo.load();
  }
  if (els.kpPreviewCard) els.kpPreviewCard.classList.add("hidden");
  if (els.kpPreviewLabel) els.kpPreviewLabel.textContent = "not built yet";
  setDownloadLink(els.kpPreviewDownload, null);
}

function showKeypointPreview(edit) {
  const url = edit?.urls?.keypoints_2d_preview;
  if (!url || !els.kpPreviewVideo || !els.kpPreviewCard) return;
  const cacheBustedUrl = `${url}?t=${Date.now()}`;
  els.kpPreviewVideo.src = cacheBustedUrl;
  const jointText = edit.num_edited_joints > 1 ? `${edit.num_edited_joints} joints` : edit.joint;
  setDownloadLink(els.kpPreviewDownload, url, `${edit.sequence}_${jointText}_keypoints_2d_preview.mp4`);
  els.kpPreviewVideo.load();
  els.kpPreviewVideo.play().catch(() => {});
  els.kpPreviewCard.classList.remove("hidden");
  if (els.kpPreviewLabel) {
    els.kpPreviewLabel.textContent = `${edit.sequence} · ${jointText} · ${edit.num_frames} frames`;
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
  state.lastRenderResult = null;
  setDownloadLink(els.renderPreviewDownload, null);
  renderDownloadButtons(els.renderDownloads, null);
}

function selectedRenderVideo(result) {
  const urls = result?.render_urls || {};
  const want2d = Boolean(els.render2dOverlay?.checked);
  if (want2d && urls.annotated_overlay_compare_video) {
    return {url: urls.annotated_overlay_compare_video, kind: "with_2d", suffix: "with_2d_trajectories"};
  }
  if (!want2d && urls.overlay_compare_video) {
    return {url: urls.overlay_compare_video, kind: "clean", suffix: "clean_overlay"};
  }
  if (urls.annotated_overlay_compare_video) return {url: urls.annotated_overlay_compare_video, kind: "with_2d", suffix: "with_2d_trajectories"};
  if (urls.overlay_compare_video) return {url: urls.overlay_compare_video, kind: "clean", suffix: "clean_overlay"};
  if (urls.compare_video) return {url: urls.compare_video, kind: "compare", suffix: "side_by_side"};
  if (urls.after_video) return {url: urls.after_video, kind: "after", suffix: "after"};
  return {url: null, kind: "missing", suffix: "render"};
}

function refreshShownRenderVideo({cacheBust = false, scroll = false} = {}) {
  const result = state.lastRenderResult;
  const selected = selectedRenderVideo(result);
  const url = selected.url;
  if (!url || !els.renderPreviewVideo || !els.renderPreviewCard) return false;
  const currentTime = Number.isFinite(els.renderPreviewVideo.currentTime) ? els.renderPreviewVideo.currentTime : 0;
  const wasPaused = els.renderPreviewVideo.paused;
  els.renderPreviewVideo.src = cacheBust ? `${url}?t=${Date.now()}` : url;
  setDownloadLink(els.renderPreviewDownload, url, `${result.sequence}_spline_opt_${selected.suffix}.mp4`);
  renderDownloadButtons(els.renderDownloads, result?.render_urls);
  els.renderPreviewVideo.load();
  els.renderPreviewVideo.addEventListener("loadedmetadata", () => {
    try { els.renderPreviewVideo.currentTime = Math.min(currentTime, els.renderPreviewVideo.duration || currentTime); } catch (_) {}
    if (!wasPaused) els.renderPreviewVideo.play().catch(() => {});
  }, {once: true});
  if (cacheBust || !wasPaused) els.renderPreviewVideo.play().catch(() => {});
  els.renderPreviewCard.classList.remove("hidden");
  if (scroll) els.renderPreviewCard.scrollIntoView({behavior: "smooth", block: "nearest"});
  if (els.renderPreviewLabel) {
    const overlayText = selected.kind === "with_2d" ? "2D trajectories on" : "2D trajectories off";
    els.renderPreviewLabel.textContent = `${result.sequence} · ${result.num_frames} frames · ${overlayText}`;
  }
  return true;
}

function showRenderPreview(result) {
  state.lastRenderResult = result;
  return refreshShownRenderVideo({cacheBust: true, scroll: true});
}

function editModeFromUI() {
  return els.keyframeMode?.checked ? "keyframe" : "stroke";
}

function selectedJointName() {
  return state.selectedJointName || els.joint.value || "left_wrist";
}

function getJointInfo(nameOrIndex) {
  if (!state.meta) return null;
  if (typeof nameOrIndex === "number") return state.meta.joints.find((j) => Number(j.index) === nameOrIndex) || null;
  return state.meta.joints.find((j) => j.name === nameOrIndex) || null;
}

function selectedJointIndex() {
  const joint = getJointInfo(selectedJointName());
  return joint ? Number(joint.index) : 9;
}

function selectedJointLabel() {
  const joint = getJointInfo(selectedJointName());
  return joint ? `${joint.label} (${joint.index})` : `Joint ${selectedJointIndex()}`;
}

function defaultJointEdit(mode = editModeFromUI()) {
  return {
    mode,
    timing: els.timing?.value || "original_speed",
    interpolation: els.interpolation?.value || "linear",
    stroke: [],
    keyframes: [],
    sampled: [],
    timingReport: null,
  };
}

function getJointEdit(name = selectedJointName(), create = true) {
  if (!name) return null;
  if (!state.jointEdits[name] && create) state.jointEdits[name] = defaultJointEdit(editModeFromUI());
  return state.jointEdits[name] || null;
}

function jointHasContent(spec) {
  if (!spec) return false;
  return Boolean((spec.stroke && spec.stroke.length) || (spec.keyframes && spec.keyframes.length) || (spec.sampled && spec.sampled.length));
}

function editedJointNames({includeSampled = true} = {}) {
  return Object.entries(state.jointEdits)
    .filter(([, spec]) => Boolean((spec.stroke && spec.stroke.length) || (spec.keyframes && spec.keyframes.length) || (includeSampled && spec.sampled && spec.sampled.length)))
    .map(([name]) => name);
}

function selectedFrameRange() {
  const n = state.meta?.num_frames || 1;
  const start = Math.max(0, Number(els.frameStart.value || 0));
  const endExclusive = els.frameEnd.value === "" ? n : Math.min(n, Number(els.frameEnd.value));
  const end = Math.max(start + 1, endExclusive) - 1;
  return {start, end, endExclusive: end + 1};
}

function sortedKeyframes(spec = getJointEdit(selectedJointName(), false)) {
  return [...(spec?.keyframes || [])].sort((a, b) => Number(a.frame) - Number(b.frame));
}

function keyframeHasFrame(frame, spec = getJointEdit(selectedJointName(), false)) {
  return Boolean(spec?.keyframes?.some((kf) => Number(kf.frame) === Number(frame)));
}

function clearGeneratedPreviews() {
  state.lastEdit = null;
  hideKeypointPreview();
  hideRenderPreview();
}

function resetCurrentEdit({keepMode = true} = {}) {
  const name = selectedJointName();
  const oldMode = getJointEdit(name, false)?.mode || editModeFromUI();
  state.jointEdits[name] = defaultJointEdit(keepMode ? oldMode : editModeFromUI());
  clearGeneratedPreviews();
  setReport({});
  updateModeUI();
  updateTposeActive();
  drawAllOverlays();
  setStatus(`Cleared edit for ${selectedJointLabel()}. Other joints are preserved.`);
}

function clearAllEdits() {
  state.jointEdits = {};
  clearGeneratedPreviews();
  setReport({});
  updateModeUI();
  updateTposeActive();
  drawAllOverlays();
}

function updateModeUI() {
  const spec = getJointEdit(selectedJointName(), true);
  const isKeyframe = spec?.mode === "keyframe";
  if (els.keyframeMode) els.keyframeMode.checked = isKeyframe;
  if (els.timing && spec?.timing) els.timing.value = spec.timing;
  if (els.interpolation && spec?.interpolation) els.interpolation.value = spec.interpolation;
  if (els.interpolationLabel) els.interpolationLabel.classList.toggle("hidden", !isKeyframe);
  if (els.keyframePanel) {
    els.keyframePanel.classList.remove("hidden");
    els.keyframePanel.classList.toggle("inactive", !isKeyframe);
    const summary = els.keyframePanel.querySelector("summary");
    if (summary) {
      summary.textContent = isKeyframe
        ? `Keyframes · ${selectedJointLabel()}`
        : `Keyframes · off`;
    }
  }
  if (els.timing) els.timing.disabled = isKeyframe;
  renderKeyframeList();
}

function setCurrentEditMode(mode) {
  const spec = getJointEdit(selectedJointName(), true);
  if (spec.mode !== mode) {
    spec.mode = mode;
    spec.stroke = [];
    spec.keyframes = [];
    spec.sampled = [];
    spec.timingReport = null;
    clearGeneratedPreviews();
  }
  updateModeUI();
  updateTposeActive();
  drawAllOverlays();
}

function addOrReplaceKeyframe(pointNorm) {
  if (!state.meta) return;
  const spec = getJointEdit(selectedJointName(), true);
  spec.mode = "keyframe";
  const {start, end} = selectedFrameRange();
  const frame = currentFrameIndex();
  if (frame < start || frame > end) {
    setStatus(`Current frame ${frame} is outside selected range [${start}, ${end}].`);
    return;
  }
  const idx = spec.keyframes.findIndex((kf) => Number(kf.frame) === frame);
  const item = {frame, point_norm: pointNorm};
  if (idx >= 0) spec.keyframes[idx] = item;
  else spec.keyframes.push(item);
  spec.sampled = [];
  clearGeneratedPreviews();
  renderKeyframeList();
  updateTposeActive();
  drawAllOverlays();
  setStatus(`Set ${selectedJointLabel()} keyframe at frame ${frame}. Need frame ${start} and ${end}.`);
}

function removeKeyframe(frame) {
  const spec = getJointEdit(selectedJointName(), false);
  if (!spec) return;
  spec.keyframes = spec.keyframes.filter((kf) => Number(kf.frame) !== Number(frame));
  spec.sampled = [];
  clearGeneratedPreviews();
  renderKeyframeList();
  updateTposeActive();
  drawAllOverlays();
}

function renderKeyframeList() {
  if (!els.keyframeList) return;
  const spec = getJointEdit(selectedJointName(), false);
  if (!spec || spec.mode !== "keyframe") {
    els.keyframeList.textContent = "Switch Edit mode to Keyframes to add points.";
    return;
  }
  const {start, end} = selectedFrameRange();
  const kfs = sortedKeyframes(spec);
  if (!kfs.length) {
    els.keyframeList.textContent = `Need frame ${start} and ${end}.`;
    return;
  }
  els.keyframeList.innerHTML = "";
  const summary = document.createElement("div");
  summary.className = "keyframe-summary";
  summary.textContent = `${selectedJointLabel()}: ${kfs.length} keyframe(s). Required: ${keyframeHasFrame(start, spec) ? "✓" : "✗"} frame ${start}, ${keyframeHasFrame(end, spec) ? "✓" : "✗"} frame ${end}`;
  els.keyframeList.appendChild(summary);
  for (const kf of kfs) {
    const row = document.createElement("div");
    row.className = "keyframe-row";
    const px = normToPx(kf.point_norm);
    const text = document.createElement("span");
    text.textContent = `f${kf.frame}: (${Math.round(px[0])}, ${Math.round(px[1])})`;
    const jump = document.createElement("button");
    jump.type = "button";
    jump.textContent = "Go";
    jump.addEventListener("click", () => {
      const fps = state.meta?.video?.fps || 30;
      setCurrentFrame(Number(kf.frame), fps);
    });
    const del = document.createElement("button");
    del.type = "button";
    del.textContent = "×";
    del.title = "Remove keyframe";
    del.addEventListener("click", () => removeKeyframe(kf.frame));
    row.appendChild(text);
    row.appendChild(jump);
    row.appendChild(del);
    els.keyframeList.appendChild(row);
  }
}

function validateKeyframesForBuild(jointName, spec) {
  const {start, end} = selectedFrameRange();
  if (!keyframeHasFrame(start, spec) || !keyframeHasFrame(end, spec)) {
    throw new Error(`${jointName}: keyframe mode requires keyframes at frame ${start} and frame ${end}.`);
  }
  if ((spec.keyframes || []).length < 2) throw new Error(`${jointName}: keyframe mode requires at least two keyframes.`);
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
  for (const canvas of [els.canvas, els.displayCanvas]) {
    canvas.width = meta.width;
    canvas.height = meta.height;
  }
  const aspect = meta.width / Math.max(1, meta.height);
  const maxByViewportHeight = Math.max(220, Math.floor(window.innerHeight * 0.68 * aspect));
  for (const wrap of [els.wrap, els.displayWrap]) {
    wrap.style.setProperty("--video-aspect", `${meta.width} / ${meta.height}`);
    wrap.style.setProperty("--video-max-width", `${maxByViewportHeight}px`);
  }
  drawAllOverlays();
}

function currentFrameIndex() {
  const fps = state.meta?.video?.fps || 30;
  const n = state.meta?.num_frames || 1;
  return Math.max(0, Math.min(n - 1, Math.round(els.video.currentTime * fps)));
}

function setCurrentFrame(frame, fps = state.meta?.video?.fps || 30) {
  const n = state.meta?.num_frames || 1;
  const f = Math.max(0, Math.min(n - 1, Math.round(Number(frame))));
  els.slider.value = f;
  els.video.currentTime = f / fps;
  if (els.displayVideo) els.displayVideo.currentTime = f / fps;
  els.frameLabel.textContent = `frame ${f}`;
  drawAllOverlays();
}

function syncDisplayVideo() {
  if (!els.displayVideo || !els.video) return;
  if (Math.abs((els.displayVideo.currentTime || 0) - (els.video.currentTime || 0)) > 0.05) {
    try { els.displayVideo.currentTime = els.video.currentTime; } catch (_) {}
  }
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

function drawPolylineWith(ctxLike, points, color, width = 3, close = false) {
  if (!points || points.length < 2) return;
  ctxLike.save();
  ctxLike.strokeStyle = color;
  ctxLike.lineWidth = width;
  ctxLike.lineJoin = "round";
  ctxLike.lineCap = "round";
  ctxLike.beginPath();
  ctxLike.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctxLike.lineTo(points[i][0], points[i][1]);
  if (close) ctxLike.closePath();
  ctxLike.stroke();
  ctxLike.restore();
}

function drawPolyline(points, color, width = 3, close = false) {
  drawPolylineWith(ctx, points, color, width, close);
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

function drawPointWith(ctxLike, p, color, r = 5, stroke = "rgba(0,0,0,.55)") {
  ctxLike.save();
  ctxLike.beginPath();
  ctxLike.arc(p[0], p[1], r, 0, Math.PI * 2);
  ctxLike.fillStyle = color;
  ctxLike.fill();
  ctxLike.lineWidth = 2;
  ctxLike.strokeStyle = stroke;
  ctxLike.stroke();
  ctxLike.restore();
}

function drawPoint(p, color, r = 5, stroke = "rgba(0,0,0,.55)") {
  drawPointWith(ctx, p, color, r, stroke);
}

function drawSkeletonOn(ctxLike, frame, {interactive = false} = {}) {
  const meta = state.meta;
  if (!meta || !meta.keypoints_2d?.[frame]) return;
  const kps = meta.keypoints_2d[frame];
  const selectedIdx = selectedJointIndex();
  const editedIdx = new Set(editedJointNames().map((name) => Number(getJointInfo(name)?.index)).filter((x) => Number.isFinite(x)));
  ctxLike.save();
  ctxLike.globalAlpha = interactive ? 0.8 : 0.72;
  ctxLike.strokeStyle = "rgba(255,255,255,.82)";
  ctxLike.lineWidth = 2;
  for (const [a, b] of meta.edges || []) {
    const pa = kps[a], pb = kps[b];
    if (!pa || !pb) continue;
    ctxLike.beginPath();
    ctxLike.moveTo(pa[0], pa[1]);
    ctxLike.lineTo(pb[0], pb[1]);
    ctxLike.stroke();
  }
  for (let i = 0; i < kps.length; i++) {
    const isSelected = i === selectedIdx;
    const isEdited = editedIdx.has(i);
    const color = isSelected ? "#27ae60" : (isEdited ? "#f2c94c" : "rgba(255,255,255,.85)");
    const radius = isSelected ? 7 : (isEdited ? 5 : 3);
    drawPointWith(ctxLike, kps[i], color, radius);
  }
  ctxLike.restore();
}

function drawOriginalTrajectoryForJoint(jointName, active = false) {
  const meta = state.meta;
  const joint = getJointInfo(jointName);
  if (!meta || !joint) return;
  const j = Number(joint.index);
  const start = Math.max(0, Number(els.frameStart.value || 0));
  const endVal = els.frameEnd.value === "" ? meta.num_frames : Number(els.frameEnd.value);
  const end = Math.max(start + 2, Math.min(meta.num_frames, endVal));
  const pts = [];
  for (let t = start; t < end; t++) {
    const p = meta.keypoints_2d?.[t]?.[j];
    if (p) pts.push(p);
  }
  if (active) {
    drawSpeedPolyline(pts, 5);
    if (pts.length) {
      drawPoint(pts[0], "#2f80ed", 6);
      drawPoint(pts[pts.length - 1], "#eb5757", 6);
    }
  } else {
    drawPolyline(pts, "rgba(47,128,237,.28)", 3);
  }
}

function drawUserStrokeForSpec(spec, active = false) {
  const pts = (spec?.stroke || []).map(normToPx);
  drawPolyline(pts, active ? "#f2c94c" : "rgba(242,201,76,.45)", active ? 4 : 3);
  if (pts.length) {
    drawPoint(pts[0], "#f2c94c", active ? 6 : 4);
    drawPoint(pts[pts.length - 1], "#f2994a", active ? 6 : 4);
  }
}

function drawKeyframesForSpec(spec, active = false) {
  const kfs = sortedKeyframes(spec);
  if (!kfs.length) return;
  const pts = kfs.map((kf) => normToPx(kf.point_norm));
  drawPolyline(pts, active ? "rgba(242,201,76,.62)" : "rgba(242,201,76,.35)", active ? 2 : 1.5);
  const current = currentFrameIndex();
  ctx.save();
  ctx.font = "bold 12px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  for (const kf of kfs) {
    const p = normToPx(kf.point_norm);
    const isCurrent = Number(kf.frame) === current;
    drawPoint(p, isCurrent ? "#eb5757" : "#f2c94c", active ? (isCurrent ? 8 : 6) : 4, "rgba(0,0,0,.7)");
    if (active) {
      ctx.fillStyle = "rgba(0,0,0,.75)";
      ctx.fillText(`f${kf.frame}`, p[0], p[1] - 10);
    }
  }
  ctx.restore();
}

function drawSampledTargetsForSpec(spec, active = false) {
  if (!spec?.sampled?.length) return;
  drawPolyline(spec.sampled, active ? "rgba(39,174,96,.82)" : "rgba(39,174,96,.42)", active ? 3 : 2);
  const stride = Math.max(1, Math.floor(spec.sampled.length / (active ? 40 : 24)));
  for (let i = 0; i < spec.sampled.length; i += stride) {
    drawPoint(spec.sampled[i], active ? "white" : "rgba(255,255,255,.75)", active ? 3 : 2, "rgba(39,174,96,.9)");
  }
}

function drawEditOverlays() {
  const selected = selectedJointName();
  const names = editedJointNames();
  for (const name of names) {
    if (name !== selected) drawOriginalTrajectoryForJoint(name, false);
  }
  drawOriginalTrajectoryForJoint(selected, true);
  for (const name of names) {
    const spec = getJointEdit(name, false);
    const active = name === selected;
    if (spec?.mode === "stroke") drawUserStrokeForSpec(spec, active);
    else if (spec?.mode === "keyframe") drawKeyframesForSpec(spec, active);
    drawSampledTargetsForSpec(spec, active);
  }
  const frame = currentFrameIndex();
  const p = state.meta?.keypoints_2d?.[frame]?.[selectedJointIndex()];
  if (p) drawPoint(p, "#27ae60", 9);
}

function drawEditOverlay() {
  if (!els.canvas.width || !els.canvas.height) return;
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (!state.meta) return;
  const frame = currentFrameIndex();
  drawSkeletonOn(ctx, frame, {interactive: true});
  drawEditOverlays();
}

function drawDisplayOverlay() {
  if (!els.displayCanvas.width || !els.displayCanvas.height) return;
  displayCtx.clearRect(0, 0, els.displayCanvas.width, els.displayCanvas.height);
  if (!state.meta || state.meta.source_render_url) return;
  const frame = currentFrameIndex();
  drawSkeletonOn(displayCtx, frame, {interactive: false});
}

function drawAllOverlays() {
  drawEditOverlay();
  drawDisplayOverlay();
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
  const editedIdx = new Set(editedJointNames().map((name) => Number(getJointInfo(name)?.index)).filter((x) => Number.isFinite(x)));
  els.tpose.querySelectorAll(".tpose-joint").forEach((node) => {
    const j = Number(node.dataset.jointIndex);
    const active = j === idx;
    node.classList.toggle("active", active);
    node.classList.toggle("edited", !active && editedIdx.has(j));
    node.setAttribute("r", active ? "10" : (editedIdx.has(j) ? "9" : "8"));
  });
}

function setSelectedJoint(nameOrIndex) {
  if (!state.meta) return;
  const joint = getJointInfo(nameOrIndex);
  if (!joint) return;
  state.selectedJointName = joint.name;
  els.joint.value = joint.name;
  getJointEdit(joint.name, true);
  if (els.jointLabel) {
    const nEdited = editedJointNames().length;
    els.jointLabel.textContent = `${joint.label} (${joint.index}) · ${nEdited} edited`;
  }
  updateModeUI();
  updateTposeActive();
  drawAllOverlays();
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
  clearAllEdits();
  state.meta = null;
  state.selectedJointName = null;
  const data = await fetchJson(`/api/sequence/${encodeURIComponent(name)}/meta`);
  state.meta = data.meta;
  els.video.src = state.meta.video_url;
  els.displayVideo.src = state.meta.source_render_url || state.meta.video_url;
  els.video.load();
  els.displayVideo.load();
  els.slider.min = 0;
  els.slider.max = Math.max(0, state.meta.num_frames - 1);
  els.slider.value = 0;
  els.frameLabel.textContent = "frame 0";
  els.frameEnd.placeholder = String(state.meta.num_frames);
  els.joint.innerHTML = "";
  for (const j of state.meta.joints) {
    const opt = document.createElement("option");
    opt.value = j.name;
    opt.textContent = `${j.label} (${j.index})`;
    els.joint.appendChild(opt);
  }
  els.joint.value = "left_wrist";
  state.selectedJointName = "left_wrist";
  getJointEdit("left_wrist", true);
  renderTpose();
  setSelectedJoint("left_wrist");
  resizeCanvasToVideo();
  updateModeUI();
  setStatus(`${name} loaded. Trajectory source: ${state.meta.trajectory_source}${state.meta.warning ? " — " + state.meta.warning : ""}`);
}

function buildPayloadEdits() {
  const edits = [];
  const namesWithAnyContent = Object.entries(state.jointEdits)
    .filter(([, spec]) => Boolean((spec.stroke && spec.stroke.length) || (spec.keyframes && spec.keyframes.length)))
    .map(([name]) => name);
  for (const name of namesWithAnyContent) {
    const spec = state.jointEdits[name];
    if (spec.mode === "keyframe") {
      validateKeyframesForBuild(name, spec);
      edits.push({
        joint: name,
        edit_mode: "keyframe",
        interpolation_mode: spec.interpolation || "linear",
        timing_mode: "original_speed",
        keyframes: sortedKeyframes(spec),
      });
    } else {
      if ((spec.stroke || []).length < 2) throw new Error(`${name}: please draw at least two points.`);
      edits.push({
        joint: name,
        edit_mode: "stroke",
        timing_mode: spec.timing || els.timing.value,
        stroke_points_norm: spec.stroke,
      });
    }
  }
  if (!edits.length) throw new Error("Please draw a stroke or set keyframes for at least one joint first.");
  return edits;
}

async function buildEdit() {
  if (!state.meta) return;
  let edits = [];
  try {
    edits = buildPayloadEdits();
  } catch (err) {
    setStatus(`Error: ${err.message}`);
    return;
  }
  els.build.disabled = true;
  setStatus(`Building edited Spline-Opt keypoints for ${edits.length} joint(s)...`);
  try {
    const payload = {
      sequence: state.meta.name,
      joint: edits[0].joint,
      edits,
      edit_mode: edits[0].edit_mode,
      timing_mode: els.timing.value,
      interpolation_mode: els.interpolation?.value || "linear",
      frame_start: Number(els.frameStart.value || 0),
      frame_end: els.frameEnd.value === "" ? null : Number(els.frameEnd.value),
      trajectory_source: "smpl_reproj",
    };
    const data = await fetchJson("/api/edit", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    state.lastEdit = data.edit;
    hideRenderPreview();
    setReport(data.edit.timing_reports || data.edit.timing_report || {});

    for (const spec of Object.values(state.jointEdits)) spec.sampled = [];
    for (const item of data.edit.joint_edits || []) {
      const spec = getJointEdit(item.joint, true);
      spec.sampled = item.sampled_target_trajectory_px || [];
      spec.timingReport = item.timing_report || null;
    }
    drawAllOverlays();
    updateTposeActive();
    setSelectedJoint(selectedJointName());
    showKeypointPreview(data.edit);

    setStatus(`Edit package written for ${data.edit.num_edited_joints || 1} joint(s): ${data.edit.paths.output_dir}`);
    if (els.runOpt.checked) await runSplineOpt(data.edit.paths.edit_request);
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
  } finally {
    els.build.disabled = false;
  }
}


function readOptionalNumber(el) {
  if (!el || String(el.value || "").trim() === "") return undefined;
  const value = Number(el.value);
  if (!Number.isFinite(value)) throw new Error(`Invalid numeric value: ${el.value}`);
  return value;
}

function readOptionalInt(el) {
  const value = readOptionalNumber(el);
  if (value === undefined) return undefined;
  return Math.round(value);
}

function collectBsplineOverrides() {
  const overrides = {};
  const degree = readOptionalInt(els.bsDegree);
  if (degree !== undefined) overrides.degree = degree;
  const mPerT = String(els.bsMPerT?.value || "").trim();
  if (mPerT) overrides.m_per_t = /^\d+$/.test(mPerT) ? Number(mPerT) : mPerT;
  const confThr = readOptionalNumber(els.bsConfThr);
  if (confThr !== undefined) overrides.trajectory_edit_conf_thr = confThr;
  const confPower = readOptionalNumber(els.bsConfPower);
  if (confPower !== undefined) overrides.trajectory_edit_conf_power = confPower;
  const priorBody = readOptionalNumber(els.bsPriorBody);
  if (priorBody !== undefined) overrides.trajectory_edit_prior_w_body_pose = priorBody;
  const smoothW = readOptionalNumber(els.bsSmoothW);
  if (smoothW !== undefined) overrides.smooth_w = smoothW;
  const lr = readOptionalNumber(els.bsLr);
  if (lr !== undefined) overrides.lr = lr;
  const lineSearch = String(els.bsLineSearch?.value || "").trim();
  if (lineSearch) overrides.line_search_fn = lineSearch === "none" ? null : lineSearch;
  if (els.bsLearnKnots?.checked) overrides.learn_knots = true;
  const knotMinGap = readOptionalNumber(els.bsKnotMinGap);
  if (knotMinGap !== undefined) overrides.knot_min_gap = knotMinGap;
  const knotPosW = readOptionalNumber(els.bsKnotPosW);
  if (knotPosW !== undefined) overrides.knot_pos_w = knotPosW;
  const knotGapW = readOptionalNumber(els.bsKnotGapW);
  if (knotGapW !== undefined) overrides.knot_gap_w = knotGapW;
  return overrides;
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
      bspline_overrides: collectBsplineOverrides(),
    }),
  });
  const shown = showRenderPreview(data.result);
  setStatus(shown ? "Spline-Opt done. Render result is shown below." : `Spline-Opt done: ${data.result.output_dir}`);
  setReport(data.result);
}

els.sequence.addEventListener("change", () => loadSequence(els.sequence.value));
els.joint.addEventListener("change", () => setSelectedJoint(els.joint.value));
els.frameStart.addEventListener("change", () => { renderKeyframeList(); drawAllOverlays(); clearGeneratedPreviews(); });
els.frameEnd.addEventListener("change", () => { renderKeyframeList(); drawAllOverlays(); clearGeneratedPreviews(); });
els.keyframeMode.addEventListener("change", () => {
  setCurrentEditMode(editModeFromUI());
  setStatus(editModeFromUI() === "keyframe" ? `Keyframe mode for ${selectedJointLabel()}: scrub to a frame and click target points.` : `Stroke mode for ${selectedJointLabel()}: draw the desired trajectory on the editable canvas.`);
});
els.timing.addEventListener("change", () => {
  const spec = getJointEdit(selectedJointName(), true);
  spec.timing = els.timing.value;
  spec.sampled = [];
  clearGeneratedPreviews();
  drawAllOverlays();
});
els.interpolation.addEventListener("change", () => {
  const spec = getJointEdit(selectedJointName(), true);
  spec.interpolation = els.interpolation.value;
  spec.sampled = [];
  clearGeneratedPreviews();
  renderKeyframeList();
  drawAllOverlays();
});
if (els.render2dOverlay) {
  els.render2dOverlay.addEventListener("change", () => {
    refreshShownRenderVideo({cacheBust: false, scroll: false});
  });
}
els.reset.addEventListener("click", () => resetCurrentEdit({keepMode: true}));
els.build.addEventListener("click", buildEdit);
els.playPause.addEventListener("click", () => {
  if (els.video.paused) {
    syncDisplayVideo();
    els.video.play();
    els.displayVideo.play().catch(() => {});
  } else {
    els.video.pause();
    els.displayVideo.pause();
  }
});
els.slider.addEventListener("input", () => {
  setCurrentFrame(Number(els.slider.value));
  renderKeyframeList();
});
els.video.addEventListener("loadedmetadata", resizeCanvasToVideo);
els.displayVideo.addEventListener("loadedmetadata", resizeCanvasToVideo);
els.video.addEventListener("play", () => { syncDisplayVideo(); els.displayVideo.play().catch(() => {}); });
els.video.addEventListener("pause", () => els.displayVideo.pause());
els.video.addEventListener("seeked", syncDisplayVideo);
els.video.addEventListener("timeupdate", () => {
  syncDisplayVideo();
  const f = currentFrameIndex();
  els.slider.value = f;
  els.frameLabel.textContent = `frame ${f}`;
  drawAllOverlays();
});

els.canvas.addEventListener("pointerdown", (ev) => {
  ev.preventDefault();
  const pt = pointFromEvent(ev);
  const spec = getJointEdit(selectedJointName(), true);
  spec.sampled = [];
  clearGeneratedPreviews();
  if (spec.mode === "keyframe") {
    addOrReplaceKeyframe(pt.norm);
    return;
  }
  spec.mode = "stroke";
  state.drawing = true;
  spec.stroke.push(pt.norm);
  els.canvas.setPointerCapture(ev.pointerId);
  updateTposeActive();
  drawAllOverlays();
});

els.canvas.addEventListener("pointermove", (ev) => {
  const spec = getJointEdit(selectedJointName(), false);
  if (!state.drawing || !spec || spec.mode !== "stroke") return;
  ev.preventDefault();
  spec.stroke.push(pointFromEvent(ev).norm);
  drawAllOverlays();
});

function stopDrawing(ev) {
  if (!state.drawing) return;
  state.drawing = false;
  try { els.canvas.releasePointerCapture(ev.pointerId); } catch (_) {}
  updateTposeActive();
  setSelectedJoint(selectedJointName());
  drawAllOverlays();
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
