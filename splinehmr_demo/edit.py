from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .paths import OUTPUTS_ROOT
from .sequence import COCO17_EDGES, JOINT_NAME_TO_INDEX, get_original_coco17, input_video_path, read_video_meta
from .trajectory import (
    SampledTrajectory,
    TimingMode,
    TimingReport,
    normalized_to_pixels,
    pixels_to_normalized,
    preprocess_stroke,
    sample_keyframes_with_original_speed,
    sample_user_curve_with_timing,
)


@dataclass
class EditConfig:
    sequence: str
    joint: str = "left_wrist"
    edit_mode: str = "stroke"
    timing_mode: TimingMode = "original_speed"
    interpolation_mode: str = "linear"
    frame_start: int = 0
    frame_end: int | None = None
    base_conf: float = 0.5
    edit_conf: float = 1.0
    neighbor_conf: float = 0.75
    smooth_stroke: bool = True
    stroke_min_dist_px: float = 1.5
    stroke_smooth_window: int = 5
    trajectory_source: str = "smpl_reproj"
    anchor_noise_px: float = 0.0
    anchor_noise_seed: int = 0
    device: str = "cuda"


KINEMATIC_NEIGHBORS: dict[int, list[int]] = {
    5: [6, 7, 11],
    6: [5, 8, 12],
    7: [5, 9],
    8: [6, 10],
    9: [7, 5],
    10: [8, 6],
    11: [5, 12, 13],
    12: [6, 11, 14],
    13: [11, 15],
    14: [12, 16],
    15: [13],
    16: [14],
}


def _resolve_joint(joint: str | int) -> tuple[int, str]:
    if isinstance(joint, int):
        idx = int(joint)
        if idx < 0 or idx >= 17:
            raise ValueError(f"COCO17 joint index out of range: {idx}")
        name = next((k for k, v in JOINT_NAME_TO_INDEX.items() if v == idx), str(idx))
        return idx, name
    name = str(joint)
    if name not in JOINT_NAME_TO_INDEX:
        valid = ", ".join(JOINT_NAME_TO_INDEX)
        raise ValueError(f"Unknown joint '{joint}'. Valid joints: {valid}")
    return JOINT_NAME_TO_INDEX[name], name


def _frame_slice(T: int, start: int, end: int | None) -> slice:
    s = max(0, int(start))
    e = T if end is None else min(T, int(end))
    if e <= s + 1:
        raise ValueError(f"Invalid frame range [{s}, {e}); need at least two frames")
    return slice(s, e)


def build_multi_edited_keypoints(
    *,
    original_coco17: torch.Tensor,
    joint_targets: dict[int, np.ndarray],
    joint_masks: dict[int, np.ndarray] | None = None,
    base_conf: float,
    edit_conf: float,
    neighbor_conf: float,
    anchor_noise_px: float = 0.0,
    anchor_noise_seed: int = 0,
) -> torch.Tensor:
    """Build complete Spline-Opt COCO17 input from original reprojection and multiple edited targets."""
    orig = original_coco17.detach().cpu().float()
    T, J, _ = orig.shape
    if J != 17:
        raise ValueError(f"Expected COCO17 with 17 joints, got {J}")
    if not joint_targets:
        raise ValueError("At least one edited joint target is required")
    if joint_masks is None:
        joint_masks = {}

    edited_indices = {int(j) for j in joint_targets.keys()}
    out = torch.zeros((T, J, 3), dtype=torch.float32)
    out[:, :, :2] = orig[:, :, :2]
    if float(anchor_noise_px) > 0.0:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(anchor_noise_seed))
        noise = torch.randn((T, J, 2), generator=gen, dtype=torch.float32) * float(anchor_noise_px)
        for j in edited_indices:
            noise[:, j, :] = 0.0
        out[:, :, :2] = out[:, :, :2] + noise

    out[:, :, 2] = float(base_conf)
    mask_tensors: dict[int, torch.Tensor] = {}
    for selected_joint, target_2d in joint_targets.items():
        j = int(selected_joint)
        target = np.asarray(target_2d, dtype=np.float32)
        if target.shape != (T, 2):
            raise ValueError(f"target_2d for joint {j} must be {(T, 2)}, got {target.shape}")
        mask_np = np.asarray(joint_masks.get(j, np.ones((T,), dtype=bool)), dtype=bool).reshape(-1)
        if mask_np.shape != (T,):
            raise ValueError(f"edit mask for joint {j} must be {(T,)}, got {mask_np.shape}")
        mask_t = torch.from_numpy(mask_np).bool()
        mask_tensors[j] = mask_t
        out[:, j, :2] = torch.from_numpy(target).float()
        out[mask_t, j, 2] = float(edit_conf)

    for selected_joint in edited_indices:
        mask_t = mask_tensors[int(selected_joint)]
        for j in KINEMATIC_NEIGHBORS.get(int(selected_joint), []):
            out[mask_t, j, 2] = torch.clamp(out[mask_t, j, 2], min=float(neighbor_conf))
    for selected_joint in edited_indices:
        mask_t = mask_tensors[int(selected_joint)]
        out[mask_t, int(selected_joint), 2] = float(edit_conf)
    return out


def build_edited_keypoints(
    *,
    original_coco17: torch.Tensor,
    selected_joint: int,
    target_2d: np.ndarray,
    base_conf: float,
    edit_conf: float,
    neighbor_conf: float,
    anchor_noise_px: float = 0.0,
    anchor_noise_seed: int = 0,
) -> torch.Tensor:
    """Backward-compatible single-joint wrapper."""
    return build_multi_edited_keypoints(
        original_coco17=original_coco17,
        joint_targets={int(selected_joint): target_2d},
        joint_masks={int(selected_joint): np.ones((int(original_coco17.shape[0]),), dtype=bool)},
        base_conf=base_conf,
        edit_conf=edit_conf,
        neighbor_conf=neighbor_conf,
        anchor_noise_px=anchor_noise_px,
        anchor_noise_seed=anchor_noise_seed,
    )


def _draw_translucent_polyline(frame: np.ndarray, points: np.ndarray, color: tuple[int, int, int], alpha: float, width: int) -> None:
    pts = np.asarray(points, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return
    overlay = frame.copy()
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [pts_i], False, color, int(width), lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0.0, dst=frame)


def _draw_masked_translucent_polyline(frame: np.ndarray, points: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float, width: int) -> None:
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
                _draw_translucent_polyline(frame, pts[start:i], color, alpha, width)
            start = None


def render_keypoints_preview_video(
    *,
    video_path: Path,
    keypoints_2d: torch.Tensor,
    selected_joint: int | None = None,
    target_trajectory: np.ndarray | None = None,
    selected_joints: list[int] | None = None,
    target_trajectories: dict[int, np.ndarray] | None = None,
    target_masks: dict[int, np.ndarray] | None = None,
    output_path: Path,
    frame_start: int = 0,
    fps: float | None = None,
) -> Path:
    """Render the exact edited COCO17 sequence that will be sent to Spline-Opt."""
    kps = keypoints_2d.detach().cpu().float().numpy()
    T = int(kps.shape[0])
    if target_trajectories is None:
        target_trajectories = {}
    if target_masks is None:
        target_masks = {}
    if selected_joint is not None and target_trajectory is not None:
        sj = int(selected_joint)
        target_trajectories[sj] = np.asarray(target_trajectory, dtype=np.float32)
        target_masks.setdefault(sj, np.ones((T,), dtype=bool))
    if selected_joints is None:
        selected_joints = sorted(int(j) for j in target_trajectories.keys())
    selected_set = {int(j) for j in selected_joints}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    out_fps = float(fps or src_fps or 30.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_start))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

    palette = [(0, 215, 255), (255, 120, 30), (235, 80, 210), (80, 220, 120), (230, 230, 70)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_bg = (18, 18, 18)
    for i in range(T):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        pts = kps[i, :, :2]
        conf = kps[i, :, 2]

        active_selected_set = set()
        for n, j in enumerate(sorted(target_trajectories.keys())):
            mask = np.asarray(target_masks.get(int(j), np.ones((T,), dtype=bool)), dtype=bool).reshape(-1)
            _draw_masked_translucent_polyline(frame, target_trajectories[j], mask, palette[n % len(palette)], alpha=0.58, width=4)
            if i < len(mask) and bool(mask[i]):
                active_selected_set.add(int(j))

        for a, b in COCO17_EDGES:
            if conf[a] <= 0.05 or conf[b] <= 0.05:
                continue
            pa = tuple(np.round(pts[a]).astype(int).tolist())
            pb = tuple(np.round(pts[b]).astype(int).tolist())
            cv2.line(frame, pa, pb, (245, 245, 245), 2, cv2.LINE_AA)

        for j, pxy in enumerate(pts):
            if conf[j] <= 0.05:
                continue
            p = tuple(np.round(pxy).astype(int).tolist())
            if j in active_selected_set:
                cv2.circle(frame, p, 8, (0, 220, 80), -1, cv2.LINE_AA)
                cv2.circle(frame, p, 11, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, p, 5, (80, 80, 80), 1, cv2.LINE_AA)

        text = f"Edited COCO17 preview | {len(selected_set)} edited joint(s) | frame {i + int(frame_start)}"
        (tw, th), base = cv2.getTextSize(text, font, 0.55, 2)
        cv2.rectangle(frame, (8, 8), (tw + 24, th + base + 22), label_bg, -1)
        cv2.putText(frame, text, (16, 16 + th), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    cap.release()
    return output_path


def _normalise_edit_specs(payload: dict[str, Any], cfg: EditConfig) -> list[dict[str, Any]]:
    raw = payload.get("edits", None)
    if raw is None:
        raw = [
            {
                "joint": cfg.joint,
                "edit_mode": cfg.edit_mode,
                "timing_mode": cfg.timing_mode,
                "interpolation_mode": cfg.interpolation_mode,
                "stroke_points_norm": payload.get("stroke_points_norm", None),
                "keyframes": payload.get("keyframes", None),
            }
        ]
    if not isinstance(raw, list) or not raw:
        raise ValueError("payload.edits must be a non-empty list")

    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each edit must be an object")
        joint = str(item.get("joint", cfg.joint))
        _, joint_name = _resolve_joint(joint)
        if joint_name in seen:
            raise ValueError(f"Duplicate edited joint: {joint_name}")
        seen.add(joint_name)
        specs.append(
            {
                "joint": joint_name,
                "edit_mode": str(item.get("edit_mode", cfg.edit_mode)).strip().lower(),
                "timing_mode": str(item.get("timing_mode", cfg.timing_mode)),
                "interpolation_mode": str(item.get("interpolation_mode", cfg.interpolation_mode)),
                "stroke_points_norm": item.get("stroke_points_norm", None),
                "keyframes": item.get("keyframes", None),
            }
        )
    return specs


def _sample_one_edit(
    *,
    spec: dict[str, Any],
    original: torch.Tensor,
    video_size: tuple[int, int],
    frame_start_abs: int,
    frame_end_abs_inclusive: int,
    cfg: EditConfig,
) -> dict[str, Any]:
    selected_joint, joint_name = _resolve_joint(spec["joint"])
    selected_orig = original[:, selected_joint, :2].numpy()
    edit_mode = str(spec["edit_mode"]).strip().lower()
    edit_mask = np.ones((int(selected_orig.shape[0]),), dtype=bool)
    stroke_px: np.ndarray | None = None
    keyframes_abs_px: np.ndarray | None = None
    keyframes_norm_out: np.ndarray | None = None
    keyframe_payload_out: list[dict[str, Any]] | None = None

    if edit_mode == "keyframe":
        keyframes_in = spec.get("keyframes", None)
        if not isinstance(keyframes_in, list) or len(keyframes_in) < 2:
            raise ValueError(f"{joint_name}: keyframe mode requires at least two keyframes")
        rows: list[tuple[int, float, float]] = []
        for item in keyframes_in:
            if not isinstance(item, dict):
                raise ValueError(f"{joint_name}: each keyframe must be an object with frame and point_norm")
            f_abs = int(item.get("frame"))
            pt = item.get("point_norm", item.get("norm", None))
            if pt is None or len(pt) != 2:
                raise ValueError(f"{joint_name}: each keyframe must contain point_norm=[x,y]")
            rows.append((f_abs, float(pt[0]), float(pt[1])))
        dedup: dict[int, tuple[float, float]] = {}
        for f_abs, x, y in rows:
            dedup[int(f_abs)] = (float(x), float(y))
        frames_abs = np.asarray(sorted(dedup.keys()), dtype=np.int64)
        pts_norm = np.asarray([dedup[int(f)] for f in frames_abs], dtype=np.float64)
        if len(frames_abs) < 2:
            raise ValueError(f"{joint_name}: keyframe mode requires at least two unique keyframes")
        if int(frames_abs[0]) < frame_start_abs or int(frames_abs[-1]) > frame_end_abs_inclusive:
            raise ValueError(f"{joint_name}: keyframes must lie inside selected frame range [{frame_start_abs}, {frame_end_abs_inclusive}]")
        pts_px = normalized_to_pixels(pts_norm, video_size)
        key_start_abs = int(frames_abs[0])
        key_end_abs = int(frames_abs[-1])
        key_start_rel = key_start_abs - frame_start_abs
        key_end_rel = key_end_abs - frame_start_abs
        frames_rel = frames_abs - key_start_abs
        local_orig = selected_orig[key_start_rel : key_end_rel + 1]
        local_sampled = sample_keyframes_with_original_speed(
            keyframe_frames=frames_rel.tolist(),
            keyframe_xy=pts_px,
            original_joint_xy=local_orig,
            interpolation=spec["interpolation_mode"],  # type: ignore[arg-type]
        )
        target_full = selected_orig.copy()
        target_full[key_start_rel : key_end_rel + 1] = local_sampled.target_2d
        s_t_full = np.zeros((int(selected_orig.shape[0]),), dtype=np.float32)
        s_t_full[key_start_rel : key_end_rel + 1] = local_sampled.s_t
        target_speed_full = np.linalg.norm(target_full[1:] - target_full[:-1], axis=-1).astype(np.float32)
        original_speed_full = np.linalg.norm(selected_orig[1:] - selected_orig[:-1], axis=-1).astype(np.float32)
        local_report = local_sampled.report.to_dict()
        note = local_report.get("note", "")
        if key_start_abs != frame_start_abs or key_end_abs != frame_end_abs_inclusive:
            note += f" Partial keyframe edit is active only on absolute frames [{key_start_abs}, {key_end_abs}]; outside this interval, this joint keeps the original 2D reprojection with weak confidence."
        sampled = SampledTrajectory(
            target_2d=target_full.astype(np.float32),
            s_t=s_t_full.astype(np.float32),
            target_speed_px=target_speed_full,
            original_speed_px=original_speed_full,
            report=TimingReport(
                mode=(local_sampled.report.mode if (key_start_abs == frame_start_abs and key_end_abs == frame_end_abs_inclusive) else local_sampled.report.mode + "_partial"),
                num_frames=int(selected_orig.shape[0]),
                original_total_length_px=float(local_report.get("original_total_length_px", 0.0)),
                user_curve_length_px=float(local_report.get("user_curve_length_px", 0.0)),
                speed_scale=float(local_report.get("speed_scale", float("nan"))),
                absolute_speed_exact=bool(local_report.get("absolute_speed_exact", False)),
                reaches_curve_end=bool(local_report.get("reaches_curve_end", True)),
                note=note,
            ),
        )
        edit_mask = np.zeros((int(selected_orig.shape[0]),), dtype=bool)
        edit_mask[key_start_rel : key_end_rel + 1] = True
        keyframes_abs_px = np.concatenate([frames_abs[:, None].astype(np.float64), pts_px], axis=1)
        keyframes_norm_out = np.concatenate([frames_abs[:, None].astype(np.float64), pts_norm], axis=1)
        keyframe_payload_out = [
            {"frame": int(f), "frame_relative": int(f - frame_start_abs), "point_norm": [float(pn[0]), float(pn[1])], "point_px": [float(pp[0]), float(pp[1])]}
            for f, pn, pp in zip(frames_abs.tolist(), pts_norm, pts_px)
        ]
    elif edit_mode == "stroke":
        stroke_norm = spec.get("stroke_points_norm", None)
        if not stroke_norm:
            raise ValueError(f"{joint_name}: stroke_points_norm is required")
        stroke_px = normalized_to_pixels(stroke_norm, video_size)
        stroke_px = preprocess_stroke(
            stroke_px,
            min_dist=cfg.stroke_min_dist_px,
            smooth=cfg.smooth_stroke,
            smooth_window=cfg.stroke_smooth_window,
        )
        sampled = sample_user_curve_with_timing(
            user_curve_xy=stroke_px,
            original_joint_xy=selected_orig,
            mode=spec["timing_mode"],  # type: ignore[arg-type]
        )
    else:
        raise ValueError(f"{joint_name}: unknown edit_mode '{edit_mode}'. Expected stroke or keyframe")

    return {
        "joint": joint_name,
        "joint_index": int(selected_joint),
        "edit_mode": edit_mode,
        "timing_mode": spec["timing_mode"],
        "interpolation_mode": spec["interpolation_mode"] if edit_mode == "keyframe" else None,
        "original_joint_trajectory_px": selected_orig.astype(float).tolist(),
        "sampled_target_trajectory_px": sampled.target_2d.astype(float).tolist(),
        "sampled_target_trajectory_norm": pixels_to_normalized(sampled.target_2d, video_size).astype(float).tolist(),
        "timing_report": sampled.report.to_dict(),
        "active_frame_start": int(np.flatnonzero(edit_mask)[0]) + frame_start_abs if np.any(edit_mask) else frame_start_abs,
        "active_frame_end": int(np.flatnonzero(edit_mask)[-1]) + frame_start_abs + 1 if np.any(edit_mask) else frame_start_abs,
        "edit_mask": edit_mask.astype(bool),
        "stroke_px": stroke_px,
        "keyframes_px": keyframes_abs_px,
        "keyframes_norm": keyframes_norm_out,
        "keyframes": keyframe_payload_out,
        "sampled": sampled,
    }


def create_edit(
    payload: dict[str, Any],
    *,
    run_id: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Create a single- or multi-joint edit package from browser input."""
    cfg = EditConfig(
        sequence=str(payload.get("sequence", "climbing_3mb")),
        joint=str(payload.get("joint", "left_wrist")),
        edit_mode=str(payload.get("edit_mode", "stroke")),
        timing_mode=str(payload.get("timing_mode", "original_speed")),  # type: ignore[arg-type]
        interpolation_mode=str(payload.get("interpolation_mode", "linear")),
        frame_start=int(payload.get("frame_start", 0)),
        frame_end=payload.get("frame_end", None),
        base_conf=float(payload.get("base_conf", 0.5)),
        edit_conf=float(payload.get("edit_conf", 1.0)),
        neighbor_conf=float(payload.get("neighbor_conf", 0.75)),
        smooth_stroke=bool(payload.get("smooth_stroke", True)),
        stroke_min_dist_px=float(payload.get("stroke_min_dist_px", 1.5)),
        stroke_smooth_window=int(payload.get("stroke_smooth_window", 5)),
        trajectory_source=str(payload.get("trajectory_source", "smpl_reproj")),
        anchor_noise_px=float(payload.get("anchor_noise_px", 0.0)),
        anchor_noise_seed=int(payload.get("anchor_noise_seed", 0)),
        device=str(payload.get("device", "cuda")),
    )
    edit_specs = _normalise_edit_specs(payload, cfg)

    video_meta = read_video_meta(input_video_path(cfg.sequence))
    video_size = (video_meta.width, video_meta.height)
    original, actual_source, source_warning = get_original_coco17(
        cfg.sequence,
        source=cfg.trajectory_source,
        device=cfg.device,
        allow_fallback=True,
    )
    T_total = min(int(original.shape[0]), int(video_meta.frame_count))
    sl = _frame_slice(T_total, cfg.frame_start, cfg.frame_end)
    original = original[sl].clone()
    T = int(original.shape[0])
    frame_start_abs = int(sl.start or 0)
    frame_end_abs_inclusive = int((sl.stop or T_total) - 1)

    processed = [
        _sample_one_edit(
            spec=spec,
            original=original,
            video_size=video_size,
            frame_start_abs=frame_start_abs,
            frame_end_abs_inclusive=frame_end_abs_inclusive,
            cfg=cfg,
        )
        for spec in edit_specs
    ]
    joint_targets = {int(item["joint_index"]): item["sampled"].target_2d for item in processed}
    joint_masks = {int(item["joint_index"]): item["edit_mask"] for item in processed}
    edited = build_multi_edited_keypoints(
        original_coco17=original,
        joint_targets=joint_targets,
        joint_masks=joint_masks,
        base_conf=cfg.base_conf,
        edit_conf=cfg.edit_conf,
        neighbor_conf=cfg.neighbor_conf,
        anchor_noise_px=cfg.anchor_noise_px,
        anchor_noise_seed=cfg.anchor_noise_seed,
    )

    req_id = request_id or str(payload.get("request_id") or f"edit_{int(time.time())}")
    first_joint_name = str(processed[0]["joint"])
    edit_name = first_joint_name if len(processed) == 1 else f"multi_{len(processed)}joints"
    edit_id = run_id or f"{cfg.sequence}_{edit_name}_{req_id}"
    out_dir = OUTPUTS_ROOT / "trajectory_edit" / edit_id
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_traj_stack = np.stack([np.asarray(item["original_joint_trajectory_px"], dtype=np.float32) for item in processed], axis=0)
    target_traj_stack = np.stack([item["sampled"].target_2d.astype(np.float32) for item in processed], axis=0)
    joint_indices = np.asarray([int(item["joint_index"]) for item in processed], dtype=np.int64)
    joint_names = [str(item["joint"]) for item in processed]
    edit_mask_stack = np.stack([item["edit_mask"].astype(np.bool_) for item in processed], axis=0)

    np.save(out_dir / "original_joint_trajectories.npy", orig_traj_stack)
    np.save(out_dir / "sampled_target_trajectories.npy", target_traj_stack)
    np.save(out_dir / "edited_joint_masks.npy", edit_mask_stack)
    np.save(out_dir / "edited_joint_indices.npy", joint_indices)
    np.save(out_dir / "original_joint_trajectory.npy", orig_traj_stack[0])
    np.save(out_dir / "sampled_target_trajectory.npy", target_traj_stack[0])
    np.save(out_dir / "timing_s_t.npy", processed[0]["sampled"].s_t.astype(np.float32))

    for item in processed:
        safe_joint = str(item["joint"])
        if item["stroke_px"] is not None:
            np.save(out_dir / f"{safe_joint}_stroke_points_px.npy", item["stroke_px"].astype(np.float32))
            np.save(out_dir / f"{safe_joint}_stroke_points_norm.npy", pixels_to_normalized(item["stroke_px"], video_size).astype(np.float32))
        if item["keyframes_px"] is not None:
            np.save(out_dir / f"{safe_joint}_keyframes_px.npy", item["keyframes_px"].astype(np.float32))
        if item["keyframes_norm"] is not None:
            np.save(out_dir / f"{safe_joint}_keyframes_norm.npy", item["keyframes_norm"].astype(np.float32))
        np.save(out_dir / f"{safe_joint}_edit_mask.npy", item["edit_mask"].astype(np.bool_))

    torch.save(edited, out_dir / "keypoints_2d_edit.pt")
    torch.save(original, out_dir / "keypoints_2d_original.pt")
    generate_preview = bool(payload.get("generate_preview", payload.get("preview_2d", True)))
    preview_path: Path | None = None
    preview_url: str | None = None
    if generate_preview:
        preview_path = render_keypoints_preview_video(
            video_path=input_video_path(cfg.sequence),
            keypoints_2d=edited,
            selected_joints=[int(x) for x in joint_indices.tolist()],
            target_trajectories={int(item["joint_index"]): item["sampled"].target_2d for item in processed},
            target_masks={int(item["joint_index"]): item["edit_mask"] for item in processed},
            output_path=out_dir / "keypoints_2d_preview.mp4",
            frame_start=frame_start_abs,
            fps=float(video_meta.fps),
        )
        preview_url = f"/outputs/trajectory_edit/{edit_id}/keypoints_2d_preview.mp4"

    joint_edit_reports: list[dict[str, Any]] = []
    for n, item in enumerate(processed):
        safe_joint = str(item["joint"])
        paths = {
            "stroke_points_px": (str(out_dir / f"{safe_joint}_stroke_points_px.npy") if item["stroke_px"] is not None else None),
            "stroke_points_norm": (str(out_dir / f"{safe_joint}_stroke_points_norm.npy") if item["stroke_px"] is not None else None),
            "keyframes_px": (str(out_dir / f"{safe_joint}_keyframes_px.npy") if item["keyframes_px"] is not None else None),
            "keyframes_norm": (str(out_dir / f"{safe_joint}_keyframes_norm.npy") if item["keyframes_norm"] is not None else None),
        }
        joint_edit_reports.append(
            {
                "joint": safe_joint,
                "joint_index": int(item["joint_index"]),
                "edit_mode": item["edit_mode"],
                "timing_mode": item["timing_mode"],
                "interpolation_mode": item["interpolation_mode"],
                "keyframes": item["keyframes"],
                "timing_report": item["timing_report"],
                "active_frame_start": int(item["active_frame_start"]),
                "active_frame_end": int(item["active_frame_end"]),
                "edit_mask": item["edit_mask"].astype(bool).tolist(),
                "sampled_target_trajectory_px": item["sampled_target_trajectory_px"],
                "sampled_target_trajectory_norm": item["sampled_target_trajectory_norm"],
                "original_joint_trajectory_px": item["original_joint_trajectory_px"],
                "paths": paths,
            }
        )

    request = {
        "request_id": req_id,
        "edit_id": edit_id,
        "config": asdict(cfg),
        "sequence": cfg.sequence,
        "joint": first_joint_name,
        "joint_index": int(joint_indices[0]),
        "joints": [{"joint": name, "joint_index": int(idx)} for name, idx in zip(joint_names, joint_indices.tolist())],
        "num_edited_joints": len(processed),
        "edit_mode": processed[0]["edit_mode"],
        "interpolation_mode": processed[0]["interpolation_mode"],
        "keyframes": processed[0]["keyframes"],
        "joint_edits": joint_edit_reports,
        "joint_format": "coco17",
        "frame_start": frame_start_abs,
        "frame_end": int(sl.stop or T),
        "num_frames": int(T),
        "video_size": [int(video_meta.width), int(video_meta.height)],
        "trajectory_source_requested": cfg.trajectory_source,
        "trajectory_source_used": actual_source,
        "source_warning": source_warning,
        "generate_preview": bool(generate_preview),
        "timing_report": processed[0]["timing_report"],
        "timing_reports": {str(item["joint"]): item["timing_report"] for item in processed},
        "sampled_target_trajectory_px": joint_edit_reports[0]["sampled_target_trajectory_px"],
        "sampled_target_trajectory_norm": joint_edit_reports[0]["sampled_target_trajectory_norm"],
        "sampled_target_trajectories_px": {str(item["joint"]): item["sampled_target_trajectory_px"] for item in joint_edit_reports},
        "paths": {
            "output_dir": str(out_dir),
            "edit_request": str(out_dir / "edit_request.json"),
            "keypoints_2d_edit": str(out_dir / "keypoints_2d_edit.pt"),
            "keypoints_2d_original": str(out_dir / "keypoints_2d_original.pt"),
            "sampled_target_trajectory": str(out_dir / "sampled_target_trajectory.npy"),
            "sampled_target_trajectories": str(out_dir / "sampled_target_trajectories.npy"),
            "edited_joint_masks": str(out_dir / "edited_joint_masks.npy"),
            "original_joint_trajectory": str(out_dir / "original_joint_trajectory.npy"),
            "original_joint_trajectories": str(out_dir / "original_joint_trajectories.npy"),
            "edited_joint_indices": str(out_dir / "edited_joint_indices.npy"),
            "stroke_points_px": joint_edit_reports[0]["paths"]["stroke_points_px"],
            "stroke_points_norm": joint_edit_reports[0]["paths"]["stroke_points_norm"],
            "keyframes_px": joint_edit_reports[0]["paths"]["keyframes_px"],
            "keyframes_norm": joint_edit_reports[0]["paths"]["keyframes_norm"],
            "timing_s_t": str(out_dir / "timing_s_t.npy"),
            "keypoints_2d_preview": (str(preview_path) if preview_path is not None else None),
        },
        "urls": {
            "keypoints_2d_preview": preview_url,
        },
    }
    (out_dir / "edit_request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    return request
