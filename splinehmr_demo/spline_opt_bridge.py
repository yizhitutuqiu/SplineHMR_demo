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


def _draw_masked_polyline_bgr(
    frame: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    width: int = 3,
    alpha: float = 0.85,
) -> None:
    pts = np.asarray(points, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if len(pts) != len(mask):
        mask = np.ones((len(pts),), dtype=bool)
    start = None
    for i, keep in enumerate(mask.tolist() + [False]):
        if keep and start is None:
            start = i
        elif (not keep) and start is not None:
            if i - start >= 2:
                _draw_polyline_bgr(frame, pts[start:i], color, width=width, alpha=alpha)
            start = None


def _draw_side_legend_panel(
    canvas: np.ndarray,
    *,
    x0: int,
    y0: int,
    panel_w: int,
    n_joints: int,
) -> None:
    import cv2

    h = canvas.shape[0]
    panel_bg = (18, 18, 18)
    canvas[:, x0:] = np.array(panel_bg, dtype=np.uint8)
    cv2.line(canvas, (x0, 0), (x0, h), (58, 58, 58), 1, cv2.LINE_AA)

    pad = 18
    x = x0 + pad
    y = max(26, int(y0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.58
    text_scale = 0.46
    white = (245, 245, 245)
    muted = (182, 182, 182)

    cv2.putText(canvas, "Legend", (x, y), font, title_scale, white, 2, cv2.LINE_AA)
    y += 30

    cv2.rectangle(canvas, (x, y - 13), (x + 20, y + 7), (255, 0, 0), -1)
    cv2.putText(canvas, "Before optimization", (x + 32, y + 3), font, text_scale, white, 1, cv2.LINE_AA)
    y += 28
    cv2.rectangle(canvas, (x, y - 13), (x + 20, y + 7), (0, 255, 0), -1)
    cv2.putText(canvas, "After optimization", (x + 32, y + 3), font, text_scale, white, 1, cv2.LINE_AA)
    y += 34

    cv2.line(canvas, (x, y), (x + 30, y), (255, 120, 30), 3, cv2.LINE_AA)
    cv2.putText(canvas, "Original 2D traj.", (x + 42, y + 5), font, text_scale, white, 1, cv2.LINE_AA)
    y += 28
    cv2.line(canvas, (x, y), (x + 30, y), (0, 220, 255), 4, cv2.LINE_AA)
    cv2.putText(canvas, "Edited target traj.", (x + 42, y + 5), font, text_scale, white, 1, cv2.LINE_AA)
    y += 28
    cv2.circle(canvas, (x + 15, y), 8, (0, 245, 120), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x + 15, y), 11, white, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Current edited joint", (x + 42, y + 5), font, text_scale, white, 1, cv2.LINE_AA)
    y += 34

    cv2.putText(canvas, f"Edited joints: {int(n_joints)}", (x, y), font, text_scale, muted, 1, cv2.LINE_AA)
    y += 23
    cv2.putText(canvas, "Trajectories are drawn", (x, y), font, 0.42, muted, 1, cv2.LINE_AA)
    y += 20
    cv2.putText(canvas, "beside this panel's", (x, y), font, 0.42, muted, 1, cv2.LINE_AA)
    y += 20
    cv2.putText(canvas, "source video region.", (x, y), font, 0.42, muted, 1, cv2.LINE_AA)

def annotate_render_with_2d_targets(
    *,
    input_video_path: Path,
    render_video_path: Path,
    output_path: Path,
    original_joint_xy: np.ndarray,
    edited_joint_xy: np.ndarray,
    video_size: list[int] | tuple[int, int],
    edited_joint_masks: np.ndarray | None = None,
) -> Path:
    """Overlay original/edited trajectories, with all legends placed in a right-side panel.

    Accepts either one trajectory with shape (T,2) or multiple trajectories with shape (N,T,2).
    The source render frame is kept untouched by legend text; only trajectory curves/points are drawn on it.
    """
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
    if orig.ndim == 2:
        orig = orig[None, ...]
    if edit.ndim == 2:
        edit = edit[None, ...]
    if orig.ndim != 3 or edit.ndim != 3:
        raise ValueError(f"Expected trajectory arrays with shape (T,2) or (N,T,2), got {orig.shape} and {edit.shape}")
    n_joints = min(int(orig.shape[0]), int(edit.shape[0]))
    orig = orig[:n_joints]
    edit = edit[:n_joints]
    if edited_joint_masks is None:
        masks = np.ones((n_joints, int(orig.shape[1])), dtype=bool)
    else:
        masks = np.asarray(edited_joint_masks, dtype=bool)
        if masks.ndim == 1:
            masks = masks[None, ...]
        masks = masks[:n_joints]
        if masks.shape[1] != orig.shape[1]:
            masks = np.ones((n_joints, int(orig.shape[1])), dtype=bool)
    orig[:, :, 0] *= sx
    orig[:, :, 1] *= sy
    edit[:, :, 0] *= sx
    edit[:, :, 1] *= sy
    n = min(int(orig.shape[1]), int(edit.shape[1]), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or min(orig.shape[1], edit.shape[1])))

    panel_w = max(260, min(360, int(width * 0.42)))
    out_width = width + panel_w
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for annotation: {output_path}")

    orig_palette = [(255, 120, 30), (255, 80, 150), (180, 110, 255), (80, 180, 255), (255, 210, 90)]
    edit_palette = [(0, 220, 255), (0, 245, 120), (80, 255, 220), (70, 180, 255), (255, 255, 80)]
    annotate_log_stride = max(1, int(n) // 20)
    print(f"[Spline-Opt] annotate rendering frame=0/{int(n)}", flush=True)
    try:
        for i in range(n):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if i == 0 or (i + 1) % annotate_log_stride == 0 or (i + 1) == n:
                print(f"[Spline-Opt] annotate rendering frame={i + 1}/{int(n)}", flush=True)
            canvas = np.zeros((height, out_width, 3), dtype=np.uint8)
            canvas[:, :width] = frame
            for j in range(n_joints):
                oc = orig_palette[j % len(orig_palette)]
                ec = edit_palette[j % len(edit_palette)]
                mask = masks[j, :n]
                _draw_masked_polyline_bgr(canvas[:, :width], orig[j, :n], mask, oc, width=3, alpha=0.75)
                _draw_masked_polyline_bgr(canvas[:, :width], edit[j, :n], mask, ec, width=4, alpha=0.86)
                active_idx = np.flatnonzero(mask)
                if len(active_idx) > 0:
                    for idx_pt, pts, color in ((active_idx[0], orig[j], oc), (active_idx[-1], orig[j], oc), (active_idx[0], edit[j], ec), (active_idx[-1], edit[j], ec)):
                        p = tuple(np.round(pts[int(idx_pt)]).astype(int).tolist())
                        cv2.circle(canvas, p, 5, color, -1, cv2.LINE_AA)
                if i < len(mask) and bool(mask[i]):
                    po = tuple(np.round(orig[j, i]).astype(int).tolist())
                    pe = tuple(np.round(edit[j, i]).astype(int).tolist())
                    cv2.circle(canvas, po, 6, oc, -1, cv2.LINE_AA)
                    cv2.circle(canvas, po, 9, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(canvas, pe, 9, ec, -1, cv2.LINE_AA)
                    cv2.circle(canvas, pe, 13, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.line(canvas, po, pe, (255, 255, 255), 1, cv2.LINE_AA)
            _draw_side_legend_panel(canvas, x0=width, y0=18, panel_w=panel_w, n_joints=n_joints)
            writer.write(canvas)
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
    bspline_overrides: dict[str, Any] | None = None,
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
    applied_bspline_overrides: dict[str, Any] = {}
    allowed_bspline_keys = {
        "degree",
        "m_per_t",
        "trajectory_edit_conf_thr",
        "trajectory_edit_conf_power",
        "trajectory_edit_prior_w_body_pose",
        "smooth_w",
        "lr",
        "line_search_fn",
        "learn_knots",
        "knot_min_gap",
        "knot_pos_w",
        "knot_gap_w",
        "knot_smooth_w",
    }
    if bspline_overrides:
        if not isinstance(bspline_overrides, dict):
            raise TypeError("bspline_overrides must be a dict")
        for key, value in bspline_overrides.items():
            if key not in allowed_bspline_keys:
                raise ValueError(f"Unsupported B-spline override: {key}")
            if value in ("", None):
                continue
            cfg_cfg[key] = value
            applied_bspline_overrides[key] = value
    if max_iter is not None:
        cfg_cfg["max_iter"] = int(max_iter)
        applied_bspline_overrides["max_iter"] = int(max_iter)

    if applied_bspline_overrides:
        print(f"[Spline-Opt] UI B-spline overrides: {applied_bspline_overrides}", flush=True)
    else:
        print("[Spline-Opt] UI B-spline overrides: <none>; using trajectory-edit defaults", flush=True)

    bs_cfg = BsplineRefineConfig(
        degree=int(cfg_cfg.get("degree", 3)),
        m_per_t=_resolve_m_per_t(cfg_cfg.get("m_per_t", "fps_div_2"), fps),
        conf_thr=float(cfg_cfg.get("trajectory_edit_conf_thr", cfg_cfg.get("conf_thr", 0.3))),
        use_conf_weight=True,
        conf_power=float(cfg_cfg.get("trajectory_edit_conf_power", 4.0)),
        amp_body_pose=float(cfg_cfg.get("amp_body_pose", 1.0)),
        amp_global_orient=float(cfg_cfg.get("amp_global_orient", 1.0)),
        amp_transl=float(cfg_cfg.get("amp_transl", 1.0)),
        prior_w_body_pose=float(cfg_cfg.get("trajectory_edit_prior_w_body_pose", 0.02)),
        prior_w_global_orient=float(cfg_cfg.get("prior_w_global_orient", 0.2)),
        prior_w_transl=float(cfg_cfg.get("prior_w_transl", 0.2)),
        mv_consistency_w=float(cfg_cfg.get("mv_consistency_w", 10.0)),
        ankle_ground_align_w=float(cfg_cfg.get("ankle_ground_align_w", 10.0)),
        static_motion_w=float(cfg_cfg.get("trajectory_edit_static_motion_w", 0.0)),
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
    print(
        "[Spline-Opt] effective B-spline config from UI: "
        f"degree={bs_cfg.degree} m_per_t={bs_cfg.m_per_t} conf_thr={bs_cfg.conf_thr} "
        f"conf_power={bs_cfg.conf_power} prior_body={bs_cfg.prior_w_body_pose} "
        f"smooth_w={bs_cfg.smooth_w} lr={bs_cfg.lr} learn_knots={bs_cfg.learn_knots}",
        flush=True,
    )

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
        optimize_global_orient=bool(optim_cfg.get("trajectory_edit_optimize_global_orient", False)),
        optimize_transl=bool(optim_cfg.get("trajectory_edit_optimize_transl", False)),
        pose_limit_in_loss=bool(optim_cfg.get("pose_limit_in_loss", False)),
        static_conf_logits=(_slice_static_conf_logits(hmr, frame_start, frame_end) if float(cfg_cfg.get("trajectory_edit_static_motion_w", 0.0)) > 0.0 else None),
    )
    refined.setdefault("stats", {})
    refined["stats"]["trajectory_edit_request"] = str(edit_request_path)
    refined["stats"]["run_request_id"] = run_request_id
    refined["stats"]["bspline_overrides"] = applied_bspline_overrides
    refined["stats"]["bspline_effective_config"] = {
        "degree": int(bs_cfg.degree),
        "m_per_t": int(bs_cfg.m_per_t),
        "conf_thr": float(bs_cfg.conf_thr),
        "conf_power": float(bs_cfg.conf_power),
        "prior_w_body_pose": float(bs_cfg.prior_w_body_pose),
        "smooth_w": float(bs_cfg.smooth_w),
        "max_iter": int(bs_cfg.max_iter),
        "lr": float(bs_cfg.lr),
        "line_search_fn": bs_cfg.line_search_fn,
        "learn_knots": bool(bs_cfg.learn_knots),
        "knot_min_gap": float(bs_cfg.knot_min_gap),
        "knot_pos_w": float(bs_cfg.knot_pos_w),
        "knot_gap_w": float(bs_cfg.knot_gap_w),
        "knot_smooth_w": float(bs_cfg.knot_smooth_w),
    }
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
                draw_overlay_legend=False,
            )
            render_paths = {k: str(v) for k, v in paths.items()}
            orig_traj_path = req["paths"].get("original_joint_trajectories") or req["paths"]["original_joint_trajectory"]
            edit_traj_path = req["paths"].get("sampled_target_trajectories") or req["paths"]["sampled_target_trajectory"]
            mask_path = req["paths"].get("edited_joint_masks")
            annotated_path = annotate_render_with_2d_targets(
                input_video_path=video_path,
                render_video_path=paths["overlay_compare_video"],
                output_path=out_dir / "render_overlay_compare_annotated.mp4",
                original_joint_xy=np.load(orig_traj_path),
                edited_joint_xy=np.load(edit_traj_path),
                video_size=req.get("video_size", [1, 1]),
                edited_joint_masks=(np.load(mask_path) if mask_path else None),
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

