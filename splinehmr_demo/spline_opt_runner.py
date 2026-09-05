from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from .spline_opt_bridge import run_spline_opt_for_edit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Spline-Opt web-demo job.")
    parser.add_argument("--job-request", required=True)
    args = parser.parse_args()
    req_path = Path(args.job_request)
    req = json.loads(req_path.read_text())
    result_path = Path(req["result_path"])
    error_path = Path(req["error_path"])
    try:
        result = run_spline_opt_for_edit(
            edit_request_path=req["edit_request_path"],
            device=str(req.get("device", "cuda")),
            max_iter=req.get("max_iter", None),
            render=bool(req.get("render", False)),
            crf=int(req.get("crf", 23)),
            bspline_overrides=req.get("bspline_overrides", None),
            request_id=req.get("request_id", None),
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:
        error = {"error": repr(exc), "detail": traceback.format_exc()}
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(json.dumps(error, indent=2), encoding="utf-8")
        print(error["detail"], flush=True)
        raise


if __name__ == "__main__":
    main()
