from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from splinehmr_demo.edit import create_edit
from splinehmr_demo.paths import DEMO_ROOT, OUTPUTS_ROOT
from splinehmr_demo.sequence import input_video_path, list_sequences, sequence_meta, source_render_path
from splinehmr_demo.spline_opt_bridge import run_spline_opt_for_edit


STATIC_ROOT = DEMO_ROOT / "static"
REQUEST_LOG_ROOT = OUTPUTS_ROOT / "request_logs"


def make_request_id(prefix: str = "req") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def write_request_log(request_id: str, payload: dict, response: dict | None = None, error: str | None = None, detail: str | None = None) -> Path:
    REQUEST_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    data = {
        "request_id": request_id,
        "payload": payload,
        "response": response,
        "error": error,
        "detail": detail,
    }
    path = REQUEST_LOG_ROOT / f"{request_id}_{'error' if error else 'ok'}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _json_bytes(data: dict | list, status: str = "ok") -> bytes:
    return json.dumps(data, indent=2).encode("utf-8")


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "SplineHMRDemo/0.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[SplineHMR-demo] {self.address_string()} - {fmt % args}")

    def _send_json(self, data: dict | list, code: int = 200) -> None:
        body = _json_bytes(data)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, code: int, message: str, detail: str | None = None, request_id: str | None = None) -> None:
        payload = {"status": "error", "error": message}
        if request_id:
            payload["request_id"] = request_id
        if detail:
            payload["detail"] = detail
        self._send_json(payload, code)

    def _read_json_body(self) -> dict:
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_file(self, path: Path, *, content_type: str | None = None, head_only: bool = False) -> None:
        path = path.resolve()
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        size = path.stat().st_size
        ctype = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            start_s, _, end_s = range_header[len("bytes=") :].partition("-")
            try:
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
                start = max(0, min(start, size - 1))
                end = max(start, min(end, size - 1))
            except ValueError:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            if not head_only:
                with path.open("rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            with path.open("rb") as f:
                self.wfile.write(f.read())

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/":
                return self._send_file(STATIC_ROOT / "index.html", content_type="text/html; charset=utf-8", head_only=True)
            if path.startswith("/static/"):
                rel = path[len("/static/") :]
                return self._send_file(STATIC_ROOT / rel, head_only=True)
            if path.startswith("/media/sequence/"):
                parts = path.split("/")
                if len(parts) >= 5 and parts[4] == "0_input_video.mp4":
                    return self._send_file(input_video_path(parts[3]), content_type="video/mp4", head_only=True)
            if path.startswith("/media/source_render/"):
                parts = path.split("/")
                if len(parts) >= 5 and parts[4] == "render_before.mp4":
                    render = source_render_path(parts[3])
                    if render is None:
                        return self.send_error(HTTPStatus.NOT_FOUND, "Source render not found")
                    return self._send_file(render, content_type="video/mp4", head_only=True)
            if path.startswith("/outputs/"):
                rel = Path(path[len("/outputs/") :])
                target = (OUTPUTS_ROOT / rel).resolve()
                if not str(target).startswith(str(OUTPUTS_ROOT.resolve())):
                    return self._send_error_json(403, "Forbidden")
                return self._send_file(target, head_only=True)
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_error_json(500, repr(exc), traceback.format_exc())

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        try:
            if path == "/":
                return self._send_file(STATIC_ROOT / "index.html", content_type="text/html; charset=utf-8")
            if path.startswith("/static/"):
                rel = path[len("/static/") :]
                return self._send_file(STATIC_ROOT / rel)
            if path == "/api/sequences":
                return self._send_json({"status": "ok", "sequences": list_sequences()})
            if path.startswith("/api/sequence/") and path.endswith("/meta"):
                seq = path.split("/")[3]
                device = query.get("device", ["cuda"])[0]
                return self._send_json({"status": "ok", "meta": sequence_meta(seq, device=device)})
            if path.startswith("/media/sequence/"):
                parts = path.split("/")
                if len(parts) >= 5 and parts[4] == "0_input_video.mp4":
                    return self._send_file(input_video_path(parts[3]), content_type="video/mp4")
            if path.startswith("/media/source_render/"):
                parts = path.split("/")
                if len(parts) >= 5 and parts[4] == "render_before.mp4":
                    render = source_render_path(parts[3])
                    if render is None:
                        return self.send_error(HTTPStatus.NOT_FOUND, "Source render not found")
                    return self._send_file(render, content_type="video/mp4")
            if path.startswith("/outputs/"):
                rel = Path(path[len("/outputs/") :])
                target = (OUTPUTS_ROOT / rel).resolve()
                if not str(target).startswith(str(OUTPUTS_ROOT.resolve())):
                    return self._send_error_json(403, "Forbidden")
                return self._send_file(target)
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_error_json(500, repr(exc), traceback.format_exc())

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        request_id = make_request_id("edit" if parsed.path == "/api/edit" else "run" if parsed.path == "/api/run_spline_opt" else "req")
        payload: dict = {}
        try:
            payload = self._read_json_body()
            payload.setdefault("request_id", request_id)
            if parsed.path == "/api/edit":
                result = create_edit(payload, run_id=payload.get("edit_id", None), request_id=request_id)
                response = {"status": "ok", "request_id": request_id, "edit": result}
                write_request_log(request_id, payload, response=response)
                return self._send_json(response)
            if parsed.path == "/api/run_spline_opt":
                result = run_spline_opt_for_edit(
                    edit_request_path=payload["edit_request_path"],
                    device=str(payload.get("device", "cuda")),
                    max_iter=(None if payload.get("max_iter", None) in (None, "", "default") else int(payload.get("max_iter"))),
                    render=bool(payload.get("render", False)),
                    crf=int(payload.get("crf", 23)),
                    bspline_overrides=payload.get("bspline_overrides", None),
                    request_id=request_id,
                )
                response = {"status": "ok", "request_id": request_id, "result": result}
                write_request_log(request_id, payload, response=response)
                return self._send_json(response)
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            detail = traceback.format_exc()
            log_path = write_request_log(request_id, payload, error=repr(exc), detail=detail)
            try:
                edit_path = payload.get("edit_request_path") if isinstance(payload, dict) else None
                if edit_path:
                    edit_dir = Path(edit_path).resolve().parent
                    (edit_dir / f"{request_id}_error.json").write_text(
                        json.dumps({"request_id": request_id, "error": repr(exc), "detail": detail, "global_log": str(log_path)}, indent=2),
                        encoding="utf-8",
                    )
            except Exception:
                pass
            self._send_error_json(500, repr(exc), detail, request_id=request_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory-guided Spline-Opt MVP web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    os.chdir(str(DEMO_ROOT))
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"[SplineHMR-demo] serving on http://{args.host}:{args.port}")
    print("[SplineHMR-demo] if running on a remote server, use SSH port forwarding:")
    print(f"  ssh -L {args.port}:127.0.0.1:{args.port} <user>@<server>")
    server.serve_forever()


if __name__ == "__main__":
    main()

