from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import OUTPUTS_ROOT

JOB_ROOT = OUTPUTS_ROOT / "spline_opt_jobs"
_JOBS: dict[str, "SplineOptJob"] = {}
_JOBS_LOCK = threading.Lock()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


@dataclass
class SplineOptJob:
    job_id: str
    work_dir: Path
    request_path: Path
    result_path: Path
    error_path: Path
    log_path: Path
    status: str = "queued"
    stage: str = "Queued"
    progress: int = 0
    message: str = "Queued."
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    returncode: int | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    recent_log: list[str] = field(default_factory=list)
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
        clean = clean.replace("\r", "\n").split("\n")[-1].strip()
        if not clean:
            return
        self.recent_log.append(clean)
        self.recent_log = self.recent_log[-24:]
        self._infer_progress(clean)
        self._write_state()

    def _infer_progress(self, line: str) -> None:
        low = line.lower()
        stage = None
        progress = None
        message = line[-500:]
        m = re.search(r"optimizer closure=(\d+)/(\d+).*?(?:loss=([0-9.eE+-]+))?", line)
        if m:
            i = int(m.group(1)); n = max(1, int(m.group(2)))
            stage = "Spline-Opt optimizer"
            progress = max(self.progress, 8 + int(min(1.0, i / n) * 58))
            if m.group(3):
                message = f"optimizer closure {i}/{n}, loss={m.group(3)}"
            else:
                message = f"optimizer closure {i}/{n}"
        elif "baseline" in low:
            stage, progress = "Spline-Opt baseline", max(self.progress, 8)
        elif "optimize_time_sec" in low or "final mean" in low or "final mpjpe" in low:
            stage, progress = "Spline-Opt optimizer done", max(self.progress, 68)
        m = re.search(r"\[Spline-Opt\] rendering frame=(\d+)/(\d+)", line)
        if m:
            i = int(m.group(1)); n = max(1, int(m.group(2)))
            stage = "Rendering before/after videos"
            progress = max(self.progress, 70 + int(min(1.0, i / n) * 18))
            message = f"rendering frame {i}/{n}"
        m = re.search(r"\[Spline-Opt\] annotate rendering frame=(\d+)/(\d+)", line)
        if m:
            i = int(m.group(1)); n = max(1, int(m.group(2)))
            stage = "Rendering 2D trajectory overlay"
            progress = max(self.progress, 89 + int(min(1.0, i / n) * 8))
            message = f"annotating frame {i}/{n}"
        if stage or progress is not None:
            self.update(stage=stage, progress=progress, message=message)

    def _write_state(self) -> None:
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            (self.work_dir / "job_state.json").write_text(json.dumps(self.to_dict(include_log=True), indent=2), encoding="utf-8")
        except Exception:
            pass

    def to_dict(self, *, include_log: bool = True) -> dict[str, Any]:
        data = {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "returncode": self.returncode,
            "error": self.error,
            "result": self.result,
            "pid": self.pid,
            "stop_requested": self.stop_requested,
            "log_path": str(self.log_path),
            "request_path": str(self.request_path),
        }
        if include_log:
            data["recent_log"] = self.recent_log
        return data


def _register(job: SplineOptJob) -> None:
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job


def get_spline_opt_job(job_id: str) -> SplineOptJob | None:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def start_spline_opt_job(payload: dict[str, Any], *, request_id: str) -> SplineOptJob:
    job_id = f"splineopt_{_stamp()}_{uuid.uuid4().hex[:8]}"
    work_dir = JOB_ROOT / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    req = {
        "request_id": request_id,
        "edit_request_path": payload["edit_request_path"],
        "device": str(payload.get("device", "cuda")),
        "max_iter": (None if payload.get("max_iter", None) in (None, "", "default") else int(payload.get("max_iter"))),
        "render": bool(payload.get("render", False)),
        "crf": int(payload.get("crf", 23)),
        "bspline_overrides": payload.get("bspline_overrides", None),
        "result_path": str(work_dir / "result.json"),
        "error_path": str(work_dir / "error.json"),
    }
    request_path = work_dir / "request.json"
    request_path.write_text(json.dumps(req, indent=2), encoding="utf-8")
    job = SplineOptJob(
        job_id=job_id,
        work_dir=work_dir,
        request_path=request_path,
        result_path=Path(req["result_path"]),
        error_path=Path(req["error_path"]),
        log_path=work_dir / "spline_opt.log",
        message="Starting Spline-Opt subprocess...",
    )
    _register(job)
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


def stop_spline_opt_job(job_id: str) -> bool:
    job = get_spline_opt_job(job_id)
    if job is None:
        return False
    job.stop_requested = True
    proc = job.process
    if proc is not None and proc.poll() is None:
        job.update(status="stopping", stage="Stopping", message="Stopping Spline-Opt subprocess...")
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
    job.update(status="canceled", stage="Canceled", message="Spline-Opt job was stopped by user.")
    return True


def _run_job(job: SplineOptJob) -> None:
    cmd = [sys.executable, "-u", "-m", "splinehmr_demo.spline_opt_runner", "--job-request", str(job.request_path)]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    job.update(status="running", stage="Starting Spline-Opt", progress=3, message="Launching Spline-Opt subprocess...")
    try:
        with job.log_path.open("w", encoding="utf-8", errors="replace") as logf:
            logf.write("$ " + " ".join(cmd) + "\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
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
            job.update(status="canceled", stage="Canceled", message="Spline-Opt job was stopped by user.")
            return
        if job.returncode != 0:
            err = None
            if job.error_path.exists():
                try:
                    err = json.loads(job.error_path.read_text())
                except Exception:
                    err = None
            job.error = (err or {}).get("error") or f"Spline-Opt failed with return code {job.returncode}."
            job.update(status="error", stage="Error", progress=max(job.progress, 1), message=job.error)
            return
        if not job.result_path.exists():
            raise FileNotFoundError(f"Spline-Opt result file missing: {job.result_path}")
        job.result = json.loads(job.result_path.read_text())
        job.update(status="done", stage="Done", progress=100, message="Spline-Opt finished.")
    except Exception as exc:
        if job.stop_requested:
            job.update(status="canceled", stage="Canceled", message="Spline-Opt job was stopped by user.")
            return
        job.error = repr(exc)
        try:
            with job.log_path.open("a", encoding="utf-8", errors="replace") as logf:
                logf.write(f"\n[Wrapper Error] {type(exc).__name__}: {exc}\n")
        except Exception:
            pass
        job.update(status="error", stage="Error", progress=max(job.progress, 1), message=f"{type(exc).__name__}: {exc}")
