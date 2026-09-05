from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import CACHE_ROOT, INPUTS_ROOT, OUTPUTS_ROOT, SPLINEHMR_ROOT, WORK_ROOT
from .sequence import read_video_meta

GVHMR_ROOT = WORK_ROOT / "third_party" / "GVHMR"
UPLOAD_ROOT = OUTPUTS_ROOT / "gvhmr_uploads"

_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_JOBS: dict[str, "GVHMRJob"] = {}
_JOBS_LOCK = threading.Lock()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def safe_stem(name: str) -> str:
    stem = Path(name or "upload").stem.strip() or "upload"
    stem = _SAFE_RE.sub("_", stem).strip("._-") or "upload"
    return stem[:56]


def resolve_conda_exe() -> str:
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        "/root/miniconda3/bin/conda",
        "/root/autodl-tmp/conda/bin/conda",
        "/opt/conda/bin/conda",
    ]
    for item in candidates:
        if not item:
            continue
        path = Path(item).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    raise FileNotFoundError("Could not find conda executable. Start the demo from a shell with conda available, or install GVHMR in /root/miniconda3.")


@dataclass
class GVHMRJob:
    job_id: str
    sequence: str
    upload_path: Path
    work_dir: Path
    gvhmr_output_root: Path
    gvhmr_sequence_dir: Path
    status: str = "queued"
    stage: str = "Queued"
    progress: int = 0
    message: str = "Queued."
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    returncode: int | None = None
    error: str | None = None
    log_path: Path | None = None
    recent_log: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    pid: int | None = None
    stop_requested: bool = False
    process: subprocess.Popen | None = field(default=None, repr=False, compare=False)

    def update(self, *, status: str | None = None, stage: str | None = None, progress: int | None = None, message: str | None = None) -> None:
        if status is not None:
            self.status = status
        if stage is not None:
            self.stage = stage
        if progress is not None:
            self.progress = max(0, min(100, int(progress)))
        if message is not None:
            self.message = message
        self.updated_at = _now()
        self._write_state()

    def append_log(self, line: str) -> None:
        clean = line.rstrip("\n")
        if not clean:
            return
        # tqdm often rewrites the same terminal line; keep a compact readable tail.
        clean = clean.replace("\r", "\n").split("\n")[-1].strip()
        if not clean:
            return
        self.recent_log.append(clean)
        self.recent_log = self.recent_log[-18:]
        self._infer_progress(clean)
        self._write_state()

    def _infer_progress(self, line: str) -> None:
        low = line.lower()
        stage = None
        progress = None
        if "[copy video]" in low or low.startswith("copy"):
            stage, progress = "Copy input video", max(self.progress, 8)
        elif "[preprocess]" in low:
            stage, progress = "Preprocess", max(self.progress, 15)
        elif "bbx" in low or "track" in low or "vitpose" in low or "slam" in low or "dpvo" in low:
            stage, progress = "Preprocess", max(self.progress, 25)
        elif "[hmr4d] predicting" in low:
            stage, progress = "HMR4D inference", max(self.progress, 55)
        elif "[hmr4d] elapsed" in low:
            stage, progress = "HMR4D inference", max(self.progress, 70)
        elif "render incam" in low or "rendering incam" in low:
            stage, progress = "Render initial SMPL-X", max(self.progress, 78)
        elif "render global" in low or "rendering global" in low:
            stage, progress = "Render global view", max(self.progress, 88)
        elif "merge videos" in low:
            stage, progress = "Merge videos", max(self.progress, 94)
        pct = re.search(r"(\d{1,3})%", line)
        if pct and self.stage.lower().startswith("preprocess"):
            raw = max(0, min(100, int(pct.group(1))))
            progress = max(self.progress, 15 + int(raw * 0.35))
        if stage or progress is not None:
            self.update(stage=stage, progress=progress, message=line[-500:])

    def _write_state(self) -> None:
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            (self.work_dir / "job_state.json").write_text(json.dumps(self.to_dict(include_log=True), indent=2), encoding="utf-8")
        except Exception:
            pass

    def to_dict(self, *, include_log: bool = True) -> dict[str, Any]:
        data = {
            "job_id": self.job_id,
            "sequence": self.sequence,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "returncode": self.returncode,
            "error": self.error,
            "log_path": str(self.log_path) if self.log_path else None,
            "result": self.result,
            "pid": self.pid,
            "stop_requested": self.stop_requested,
        }
        if include_log:
            data["recent_log"] = self.recent_log
        return data


def _register(job: GVHMRJob) -> None:
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job


def get_job(job_id: str) -> GVHMRJob | None:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def stop_job(job_id: str) -> bool:
    job = get_job(job_id)
    if job is None:
        return False
    job.stop_requested = True
    proc = job.process
    if proc is not None and proc.poll() is None:
        job.update(status="stopping", stage="Stopping", message="Stopping GVHMR subprocess...")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    job.returncode = proc.poll() if proc is not None else job.returncode
    job.update(status="canceled", stage="Canceled", message="GVHMR job was stopped by user.")
    return True


def create_upload_job(filename: str, data: bytes) -> GVHMRJob:
    if not GVHMR_ROOT.exists():
        raise FileNotFoundError(f"GVHMR source not found: {GVHMR_ROOT}")
    if not data:
        raise ValueError("Uploaded video is empty.")
    stem = safe_stem(filename)
    job_id = f"gvhmr_{_stamp()}_{uuid.uuid4().hex[:8]}"
    sequence = f"upload_{_stamp()}_{stem}"
    work_dir = UPLOAD_ROOT / job_id
    upload_dir = work_dir / "upload"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f"{sequence}.mp4"
    upload_path.write_bytes(data)
    # Check early so a bad upload fails before occupying GPU time.
    meta = read_video_meta(upload_path)
    if meta.frame_count <= 0 or meta.width <= 0 or meta.height <= 0:
        raise RuntimeError("Uploaded file is not a readable video.")
    gvhmr_output_root = work_dir / "gvhmr_outputs"
    gvhmr_sequence_dir = gvhmr_output_root / sequence
    job = GVHMRJob(
        job_id=job_id,
        sequence=sequence,
        upload_path=upload_path,
        work_dir=work_dir,
        gvhmr_output_root=gvhmr_output_root,
        gvhmr_sequence_dir=gvhmr_sequence_dir,
        log_path=work_dir / "gvhmr.log",
        message=f"Uploaded {filename}: {meta.frame_count} frames, {meta.width}×{meta.height}.",
    )
    _register(job)
    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()
    return job


def _copy_result_into_splinehmr(job: GVHMRJob) -> dict[str, Any]:
    src = job.gvhmr_sequence_dir
    required = [src / "0_input_video.mp4", src / "hmr4d_results.pt", src / "preprocess" / "bbx.pt", src / "preprocess" / "vitpose.pt"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("GVHMR finished but required outputs are missing: " + ", ".join(missing))

    dst = INPUTS_ROOT / job.sequence
    dst_pre = dst / "preprocess"
    dst_pre.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "0_input_video.mp4", dst / "0_input_video.mp4")
    shutil.copy2(src / "hmr4d_results.pt", dst / "hmr4d_results.pt")
    shutil.copy2(src / "preprocess" / "bbx.pt", dst_pre / "bbx.pt")
    shutil.copy2(src / "preprocess" / "vitpose.pt", dst_pre / "vitpose.pt")

    render_src = src / "1_incam.mp4"
    render_dst = SPLINEHMR_ROOT / "outputs" / job.sequence / "spline-opt" / "render_before.mp4"
    if render_src.exists():
        render_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(render_src, render_dst)

    # Avoid stale projection cache if a user reuses names after manual edits.
    for cache in CACHE_ROOT.glob(f"{job.sequence}_*_coco17.pt"):
        try:
            cache.unlink()
        except FileNotFoundError:
            pass

    meta = read_video_meta(dst / "0_input_video.mp4")
    return {
        "sequence": job.sequence,
        "input_dir": str(dst),
        "input_video": str(dst / "0_input_video.mp4"),
        "hmr4d_results": str(dst / "hmr4d_results.pt"),
        "source_render": str(render_dst) if render_dst.exists() else None,
        "frame_count": meta.frame_count,
        "fps": meta.fps,
        "width": meta.width,
        "height": meta.height,
    }


def _run_job(job: GVHMRJob) -> None:
    job.update(status="running", stage="Starting GVHMR", progress=3, message="Launching GVHMR in conda env gvhmr...")
    conda_exe = resolve_conda_exe()
    cmd = [
        conda_exe,
        "run",
        "--no-capture-output",
        "-n",
        "gvhmr",
        "python",
        "tools/demo/demo.py",
        "--video",
        str(job.upload_path),
        "--output_root",
        str(job.gvhmr_output_root),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        job.work_dir.mkdir(parents=True, exist_ok=True)
        with job.log_path.open("w", encoding="utf-8", errors="replace") as logf:
            logf.write("$ " + " ".join(cmd) + "\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(GVHMR_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,
            )
            job.process = proc
            job.pid = proc.pid
            job._write_state()
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                job.append_log(line)
            job.returncode = proc.wait()
        if job.stop_requested:
            job.update(status="canceled", stage="Canceled", message="GVHMR job was stopped by user.")
            return
        if job.returncode != 0:
            tail = "\n".join(job.recent_log[-8:])
            try:
                job.update(stage="Register sequence", progress=97, message="GVHMR returned non-zero, but checking whether core outputs are usable...")
                result = _copy_result_into_splinehmr(job)
                result["warning"] = (
                    f"GVHMR returned code {job.returncode} after producing required core outputs. "
                    "The final optional video merge may have failed, commonly because ffmpeg is not installed. "
                    "SplineHMR editing can continue with 0_input_video.mp4, hmr4d_results.pt, preprocess files, and 1_incam.mp4."
                )
                result["gvhmr_tail"] = tail
                job.result = result
                job.update(
                    status="done",
                    stage="Done with warning",
                    progress=100,
                    message=f"GVHMR core outputs registered despite final non-critical failure. Loaded sequence {job.sequence}.",
                )
                return
            except Exception as copy_exc:
                raise RuntimeError(f"GVHMR failed with return code {job.returncode}.\n{tail}\nAdditionally failed to register partial outputs: {copy_exc!r}") from copy_exc
        job.update(stage="Register sequence", progress=97, message="Copying GVHMR outputs into SplineHMR demo inputs...")
        result = _copy_result_into_splinehmr(job)
        job.result = result
        job.update(status="done", stage="Done", progress=100, message=f"GVHMR done. Loaded sequence {job.sequence}.")
    except Exception as exc:
        if job.stop_requested:
            job.update(status="canceled", stage="Canceled", message="GVHMR job was stopped by user.")
            return
        job.error = repr(exc)
        try:
            if job.log_path:
                with job.log_path.open("a", encoding="utf-8", errors="replace") as logf:
                    logf.write(f"\n[Wrapper Error] {type(exc).__name__}: {exc}\n")
                    logf.write(f"[Wrapper cwd] {GVHMR_ROOT} exists={GVHMR_ROOT.exists()}\n")
        except Exception:
            pass
        job.update(status="error", stage="Error", progress=max(job.progress, 1), message=f"{type(exc).__name__}: {exc}")
