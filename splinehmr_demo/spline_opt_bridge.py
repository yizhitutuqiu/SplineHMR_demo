from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .paths import OUTPUTS_ROOT, SPLINEHMR_ROOT, add_splinehmr_to_path
from .sequence import input_video_path, load_bbx_xys, load_hmr


def _video_fps(video_path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return fps


def _extract_static_conf_logits(hmr: dict[str, Any]) -> torch.Tensor | None:
    net = hmr.get("net_outputs", None)
    if not isinstance(net, dict):
        return None
    x = net.get("static_conf_logits", None)
    if x is None and isinstance(net.get("model_output", None), dict):
        x = net["model_output"].get("static_conf_logits", None)
    if not torch.is_tensor(x):
        return None
    if x.ndim == 3 and int(x.shape[0]) == 1:
        return x[0]
    return x


def _slice_frame_range(obj: Any, start: int, end: int) -> Any:
    if torch.is_tensor(obj):
        if obj.ndim > 0 and int(obj.shape[0]) >= end:
            return obj[start:end].detach().cpu()
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _slice_frame_range(v, start, end) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_slice_frame_range(v, start, end) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_slice_frame_range(v, start, end) for v in obj)
    return obj


def _make_before_pack(hmr: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    return _slice_frame_range(hmr, start, end)


def _slice_static_conf_logits(hmr: dict[str, Any], start: int, end: int) -> torch.Tensor | None:
    x = _extract_static_conf_logits(hmr)
    if not torch.is_tensor(x):
        return None
    if x.ndim > 0 and int(x.shape[0]) >= end:
        return x[start:end].detach().cpu()
    return x.detach().cpu()


def _draw_polyline_bgr(
    frame: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    *,
    width: int = 3,
    alpha: float = 0.85,
) -> None:
    import cv2

    pts = np.asarray(points, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return
    overlay = frame.copy()
    cv2.polylines(overlay, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, color, width, cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0.0, dst=frame)


def _draw_trajectory_legend(frame: np.ndarray) -> None:
    import cv2

    x0, y0 = 18, 112
    box_w, box_h = 330, 104
    panel = frame[y0 : y0 + box_h, x0 : x0 + box_w].copy()
    panel[:] = np.array([0, 0, 0], dtype=np.uint8)
    frame[y0 : y0 + box_h, x0 : x0 + box_w] = (0.52 * frame[y0 : y0 + box_h, x0 : x0 + box_w] + 0.48 * panel).astype(np.uint8)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1)
    cv2.line(frame, (x0 + 16, y0 + 28), (x0 + 54, y0 + 28), (235, 130, 35), 4, cv2.LINE_AA)
    cv2.putText(frame, "Original 2D trajectory", (x0 + 66, y0 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (x0 + 16, y0 + 60), (x0 + 54, y0 + 60), (0, 220, 255), 4, cv2.LINE_AA)
    cv2.putText(frame, "Edited target trajectory", (x0 + 66, y0 + 67), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (x0 + 35, y0 + 88), 9, (0, 245, 120), -1, cv2.LINE_AA)
    cv2.circle(frame, (x0 + 35, y0 + 88), 12, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Edited joint / current frame", (x0 + 66, y0 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_render_with_2d_targets(
    *,
    input_video_path: Path,
    render_video_path: Path,
    output_path: Path,
    original_joint_xy: np.ndarray,
    edited_joint_xy: np.ndarray,
    video_size: list[int] | tuple[int, int],
) -> Path:
    """Overlay original and edited selected-joint trajectories on a render result video."""
    import cv2

    cap = cv2.VideoCapture(str(render_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open render video for annotation: {render_video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or _video_fps(input_video_path) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_w, src_h = int(video_size[0]), int(video_size[1])
    sx = width / max(1, src_w)
    sy = height / max(1, src_h)

    orig = np.asarray(original_joint_xy, dtype=np.float32).copy()
    edit = np.asarray(edited_joint_xy, dtype=np.float32).copy()
    orig[:, 0] *= sx
    orig[:, 1] *= sy
    edit[:, 0] *= sx
    edit[:, 1] *= sy
    n = min(len(orig), len(edit), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or min(len(orig), len(edit))))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for annotation: {output_path}")

    try:
        for i in range(n):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            _draw_polyline_bgr(frame, orig[:n], (235, 130, 35), width=4, alpha=0.78)
            _draw_polyline_bgr(frame, edit[:n], (0, 220, 255), width=5, alpha=0.86)
            for pts, color in ((orig, (235, 130, 35)), (edit, (0, 220, 255))):
                p0 = tuple(np.round(pts[0]).astype(int).tolist())
                p1 = tuple(np.round(pts[n - 1]).astype(int).tolist())
                cv2.circle(frame, p0, 6, color, -1, cv2.LINE_AA)
                cv2.circle(frame, p1, 6, color, -1, cv2.LINE_AA)
            po = tuple(np.round(orig[i]).astype(int).tolist())
            pe = tuple(np.round(edit[i]).astype(int).tolist())
            cv2.circle(frame, po, 8, (235, 130, 35), -1, cv2.LINE_AA)
            cv2.circle(frame, po, 11, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, pe, 10, (0, 245, 120), -1, cv2.LINE_AA)
            cv2.circle(frame, pe, 14, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, po, pe, (255, 255, 255), 1, cv2.LINE_AA)
            _draw_trajectory_legend(frame)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    return output_path


def run_spline_opt_for_edit(
    *,
    edit_request_path: str | Path,
    device: str = "cuda",
    max_iter: int | None = None,
    render: bool = False,
    crf: int = 23,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Run existing Spline-Opt with an edited COCO17 tensor produced by this demo.

    This function intentionally keeps the bridge thin: it reuses the open-source
    SplineHMR optimizer and writes outputs next to the edit package.
    """
    add_splinehmr_to_path()
    from multi_view_smpl_optimizer.utils.refiner_main import BsplineRefineConfig, refine_body_pose_bspline_lbfgs
    from splinehmr.demo import _load_yaml, _resolve_m_per_t
    from splinehmr.io import make_refined_hmr_pack
    from splinehmr.render import render_outputs

    edit_request_path = Path(edit_request_path)
    req = json.loads(edit_request_path.read_text())
    run_request_id = request_id or f"run_{req.get('edit_id', edit_request_path.parent.name)}"
    sequence = req["sequence"]
    out_dir = Path(req["paths"]["output_dir"]) / "spline_opt"
    out_dir.mkdir(parents=True, exist_ok=True)

    hmr = load_hmr(sequence)
    bbx_xys = load_bbx_xys(sequence)
    edited_coco17 = torch.load(req["paths"]["keypoints_2d_edit"], map_location="cpu").float()
    T = int(edited_coco17.shape[0])
    frame_start = int(req.get("frame_start", 0))
    frame_end = frame_start + T
    params = hmr["smpl_params_incam"]
    video_path = input_video_path(sequence)
    fps = _video_fps(video_path)

    optim_cfg_path = SPLINEHMR_ROOT / "configs" / "spline_opt.yaml"
    optim_cfg = _load_yaml(optim_cfg_path)
    cfg_cfg = dict(optim_cfg.get("cfg", {}) or {})
    if max_iter is not None:
        cfg_cfg["max_iter"] = int(max_iter)

    bs_cfg = BsplineRefineConfig(
        degree=int(cfg_cfg.get("degree", 3)),
        m_per_t=_resolve_m_per_t(cfg_cfg.get("m_per_t", "fps_div_2"), fps),
        conf_thr=float(cfg_cfg.get("conf_thr", 0.3)),
        amp_body_pose=float(cfg_cfg.get("amp_body_pose", 1.0)),
        amp_global_orient=float(cfg_cfg.get("amp_global_orient", 1.0)),
        amp_transl=float(cfg_cfg.get("amp_transl", 1.0)),
        prior_w_body_pose=float(cfg_cfg.get("prior_w_body_pose", 0.1)),
        prior_w_global_orient=float(cfg_cfg.get("prior_w_global_orient", 0.2)),
        prior_w_transl=float(cfg_cfg.get("prior_w_transl", 0.2)),
        mv_consistency_w=float(cfg_cfg.get("mv_consistency_w", 10.0)),
        ankle_ground_align_w=float(cfg_cfg.get("ankle_ground_align_w", 10.0)),
        static_motion_w=float(cfg_cfg.get("static_motion_w", 2.0)),
        static_joint_w=tuple(cfg_cfg.get("static_joint_w", (1.0, 1.0, 1.0, 1.0, 0.1, 0.1))),
        static_softmax_tau=float(cfg_cfg.get("static_softmax_tau", 1.5)),
        static_use_smpl24=bool(cfg_cfg.get("static_use_smpl24", True)),
        smooth_w=float(cfg_cfg.get("smooth_w", 0.02)),
        max_iter=int(cfg_cfg.get("max_iter", 60)),
        lr=float(cfg_cfg.get("lr", 1.0)),
        line_search_fn=cfg_cfg.get("line_search_fn", "strong_wolfe"),
        verbose=bool(cfg_cfg.get("verbose", True)),
        learn_knots=bool(cfg_cfg.get("learn_knots", False)),
        knot_min_gap=float(cfg_cfg.get("knot_min_gap", 1e-3)),
        knot_pos_w=float(cfg_cfg.get("knot_pos_w", 1.0)),
        knot_gap_w=float(cfg_cfg.get("knot_gap_w", 0.2)),
        knot_smooth_w=float(cfg_cfg.get("knot_smooth_w", 0.0)),
        optimize_pose_in_rot6d=bool(cfg_cfg.get("optimize_pose_in_rot6d", True)),
    ).resolve()

    refined = refine_body_pose_bspline_lbfgs(
        body_pose=params["body_pose"][frame_start:frame_end],
        betas=params["betas"][frame_start:frame_end],
        global_orient=params["global_orient"][frame_start:frame_end],
        transl=params["transl"][frame_start:frame_end],
        K_fullimg=hmr["K_fullimg"][frame_start:frame_end],
        bbx_xys=bbx_xys[frame_start:frame_end],
        coco17=edited_coco17,
        cfg=bs_cfg,
        device=device,
        optimize_body_pose=True,
        optimize_global_orient=bool(optim_cfg.get("optimize_global_orient", True)),
        optimize_transl=bool(optim_cfg.get("optimize_transl", True)),
        pose_limit_in_loss=bool(optim_cfg.get("pose_limit_in_loss", False)),
        static_conf_logits=_slice_static_conf_logits(hmr, frame_start, frame_end),
    )
    refined.setdefault("stats", {})
    refined["stats"]["trajectory_edit_request"] = str(edit_request_path)
    refined["stats"]["run_request_id"] = run_request_id
    invalid_tensors = []
    for key in ["body_pose_refined", "global_orient_refined", "transl_refined"]:
        value = refined.get(key)
        if torch.is_tensor(value) and (torch.isnan(value).any() or torch.isinf(value).any()):
            invalid_tensors.append(key)
    if invalid_tensors:
        err = {
            "request_id": run_request_id,
            "edit_request_path": str(edit_request_path),
            "invalid_tensors": invalid_tensors,
            "stats": refined.get("stats", {}),
        }
        (out_dir / f"{run_request_id}_nan_debug.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
        raise RuntimeError(
            "Spline-Opt returned NaN/Inf for " + ", ".join(invalid_tensors) +
            f". request_id={run_request_id}. See {out_dir / (run_request_id + '_nan_debug.json')}"
        )

    before_pack = _make_before_pack(hmr, frame_start, frame_end)
    after_pack = make_refined_hmr_pack(before_pack, refined, T, "spline-opt")

    before_path = out_dir / "hmr4d_results_before.pt"
    after_path = out_dir / "hmr4d_results.pt"
    torch.save(before_pack, before_path)
    torch.save(after_pack, after_path)

    render_paths: dict[str, str] = {}
    render_urls: dict[str, str] = {}
    if render:
        try:
            paths = render_outputs(
                video_path=video_path,
                before_pack=before_pack,
                after_pack=after_pack,
                out_dir=out_dir,
                T=T,
                device=device,
                crf=int(crf),
                fast_render=True,
                render_model="smplx2smpl",
                before_render_model="smplx2smpl",
                after_render_model="smplx2smpl",
            )
            render_paths = {k: str(v) for k, v in paths.items()}
            annotated_path = annotate_render_with_2d_targets(
                input_video_path=video_path,
                render_video_path=paths["overlay_compare_video"],
                output_path=out_dir / "render_overlay_compare_annotated.mp4",
                original_joint_xy=np.load(req["paths"]["original_joint_trajectory"]),
                edited_joint_xy=np.load(req["paths"]["sampled_target_trajectory"]),
                video_size=req.get("video_size", [1, 1]),
            )
            paths["annotated_overlay_compare_video"] = annotated_path
            render_paths = {k: str(v) for k, v in paths.items()}
            for k, v in paths.items():
                try:
                    rel = Path(v).resolve().relative_to(OUTPUTS_ROOT.resolve())
                    render_urls[k] = "/outputs/" + rel.as_posix()
                except Exception:
                    pass
        except Exception as exc:
            render_paths = {"warning": f"Rendering failed: {exc!r}"}

    report = {
        "status": "done",
        "request_id": run_request_id,
        "sequence": sequence,
        "num_frames": T,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "output_dir": str(out_dir),
        "before_params": str(before_path),
        "after_params": str(after_path),
        "renders": render_paths,
        "render_urls": render_urls,
        "stats": refined.get("stats", {}),
    }
    (out_dir / "spline_opt_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

