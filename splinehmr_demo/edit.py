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
    TimingMode,
    normalized_to_pixels,
    pixels_to_normalized,
    preprocess_stroke,
    sample_user_curve_with_timing,
)


@dataclass
class EditConfig:
    sequence: str
    joint: str
    timing_mode: TimingMode = "original_speed"
    frame_start: int = 0
    frame_end: int | None = None
    base_conf: float = 0.35
    edit_conf: float = 1.0
    neighbor_conf: float = 0.65
    smooth_stroke: bool = True
    stroke_min_dist_px: float = 1.5
    stroke_smooth_window: int = 5
    trajectory_source: str = "smpl_reproj"
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


def build_edited_keypoints(
    *,
    original_coco17: torch.Tensor,
    selected_joint: int,
    target_2d: np.ndarray,
    base_conf: float,
    edit_conf: float,
    neighbor_conf: float,
) -> torch.Tensor:
    """Build complete Spline-Opt COCO17 input from original reprojection and edited joint target."""
    orig = original_coco17.detach().cpu().float()
    T, J, _ = orig.shape
    if J != 17:
        raise ValueError(f"Expected COCO17 with 17 joints, got {J}")
    if target_2d.shape != (T, 2):
        raise ValueError(f"target_2d must be {(T, 2)}, got {target_2d.shape}")

    out = torch.zeros((T, J, 3), dtype=torch.float32)
    out[:, :, :2] = orig[:, :, :2]
    out[:, :, 2] = float(base_conf)
    out[:, selected_joint, :2] = torch.from_numpy(target_2d).float()
    out[:, selected_joint, 2] = float(edit_conf)
    for j in KINEMATIC_NEIGHBORS.get(int(selected_joint), []):
        out[:, j, 2] = max(float(out[:, j, 2].max().item()), float(neighbor_conf))
    return out


def _draw_translucent_polyline(frame: np.ndarray, points: np.ndarray, color: tuple[int, int, int], alpha: float, width: int) -> None:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return
    overlay = frame.copy()
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [pts_i], False, color, int(width), lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0.0, dst=frame)


def render_keypoints_preview_video(
    *,
    video_path: Path,
    keypoints_2d: torch.Tensor,
    selected_joint: int,
    target_trajectory: np.ndarray,
    output_path: Path,
    frame_start: int = 0,
    fps: float | None = None,
) -> Path:
    """Render the exact edited COCO17 sequence that will be sent to Spline-Opt."""
    import cv2

    kps = keypoints_2d.detach().cpu().float().numpy()
    T = int(kps.shape[0])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    out_fps = float(fps or src_fps or 30.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_start))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

    target = np.asarray(target_trajectory, dtype=np.float32)
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_bg = (18, 18, 18)
    for i in range(T):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        pts = kps[i, :, :2]
        conf = kps[i, :, 2]

        _draw_translucent_polyline(frame, target, (0, 215, 255), alpha=0.62, width=4)

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
            if j == int(selected_joint):
                cv2.circle(frame, p, 8, (0, 220, 80), -1, cv2.LINE_AA)
                cv2.circle(frame, p, 10, (10, 80, 10), 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, p, 5, (80, 80, 80), 1, cv2.LINE_AA)

        text = f"Edited COCO17 input preview | frame {i + int(frame_start)}"
        (tw, th), base = cv2.getTextSize(text, font, 0.55, 2)
        cv2.rectangle(frame, (8, 8), (tw + 24, th + base + 22), label_bg, -1)
        cv2.putText(frame, text, (16, 16 + th), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    cap.release()
    return output_path


def create_edit(
    payload: dict[str, Any],
    *,
    run_id: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Create an edit package from a browser stroke.

    Expected payload fields:
        sequence: str
        joint: str
        stroke_points_norm: [[x,y], ...]
        timing_mode: original_speed | absolute_speed | uniform
    """
    cfg = EditConfig(
        sequence=str(payload.get("sequence", "climbing_3mb")),
        joint=str(payload.get("joint", "left_wrist")),
        timing_mode=str(payload.get("timing_mode", "original_speed")),  # type: ignore[arg-type]
        frame_start=int(payload.get("frame_start", 0)),
        frame_end=payload.get("frame_end", None),
        base_conf=float(payload.get("base_conf", 0.35)),
        edit_conf=float(payload.get("edit_conf", 1.0)),
        neighbor_conf=float(payload.get("neighbor_conf", 0.65)),
        smooth_stroke=bool(payload.get("smooth_stroke", True)),
        stroke_min_dist_px=float(payload.get("stroke_min_dist_px", 1.5)),
        stroke_smooth_window=int(payload.get("stroke_smooth_window", 5)),
        trajectory_source=str(payload.get("trajectory_source", "smpl_reproj")),
        device=str(payload.get("device", "cuda")),
    )
    selected_joint, joint_name = _resolve_joint(cfg.joint)
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

    stroke_norm = payload.get("stroke_points_norm", None)
    if not stroke_norm:
        raise ValueError("payload.stroke_points_norm is required")
    stroke_px = normalized_to_pixels(stroke_norm, video_size)
    stroke_px = preprocess_stroke(
        stroke_px,
        min_dist=cfg.stroke_min_dist_px,
        smooth=cfg.smooth_stroke,
        smooth_window=cfg.stroke_smooth_window,
    )
    selected_orig = original[:, selected_joint, :2].numpy()
    sampled = sample_user_curve_with_timing(
        user_curve_xy=stroke_px,
        original_joint_xy=selected_orig,
        mode=cfg.timing_mode,
    )
    edited = build_edited_keypoints(
        original_coco17=original,
        selected_joint=selected_joint,
        target_2d=sampled.target_2d,
        base_conf=cfg.base_conf,
        edit_conf=cfg.edit_conf,
        neighbor_conf=cfg.neighbor_conf,
    )

    req_id = request_id or str(payload.get("request_id") or f"edit_{int(time.time())}")
    edit_id = run_id or f"{cfg.sequence}_{joint_name}_{req_id}"
    out_dir = OUTPUTS_ROOT / "trajectory_edit" / edit_id
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "stroke_points_px.npy", stroke_px.astype(np.float32))
    np.save(out_dir / "stroke_points_norm.npy", pixels_to_normalized(stroke_px, video_size).astype(np.float32))
    np.save(out_dir / "original_joint_trajectory.npy", selected_orig.astype(np.float32))
    np.save(out_dir / "sampled_target_trajectory.npy", sampled.target_2d.astype(np.float32))
    np.save(out_dir / "timing_s_t.npy", sampled.s_t.astype(np.float32))
    torch.save(edited, out_dir / "keypoints_2d_edit.pt")
    torch.save(original, out_dir / "keypoints_2d_original.pt")
    preview_path = render_keypoints_preview_video(
        video_path=input_video_path(cfg.sequence),
        keypoints_2d=edited,
        selected_joint=selected_joint,
        target_trajectory=sampled.target_2d,
        output_path=out_dir / "keypoints_2d_preview.mp4",
        frame_start=int(sl.start or 0),
        fps=float(video_meta.fps),
    )
    preview_url = f"/outputs/trajectory_edit/{edit_id}/keypoints_2d_preview.mp4"

    request = {
        "request_id": req_id,
        "edit_id": edit_id,
        "config": asdict(cfg),
        "sequence": cfg.sequence,
        "joint": joint_name,
        "joint_index": int(selected_joint),
        "joint_format": "coco17",
        "frame_start": int(sl.start or 0),
        "frame_end": int(sl.stop or T),
        "num_frames": int(T),
        "video_size": [int(video_meta.width), int(video_meta.height)],
        "trajectory_source_requested": cfg.trajectory_source,
        "trajectory_source_used": actual_source,
        "source_warning": source_warning,
        "timing_report": sampled.report.to_dict(),
        "sampled_target_trajectory_px": sampled.target_2d.astype(float).tolist(),
        "sampled_target_trajectory_norm": pixels_to_normalized(sampled.target_2d, video_size).astype(float).tolist(),
        "paths": {
            "output_dir": str(out_dir),
            "edit_request": str(out_dir / "edit_request.json"),
            "keypoints_2d_edit": str(out_dir / "keypoints_2d_edit.pt"),
            "keypoints_2d_original": str(out_dir / "keypoints_2d_original.pt"),
            "sampled_target_trajectory": str(out_dir / "sampled_target_trajectory.npy"),
            "original_joint_trajectory": str(out_dir / "original_joint_trajectory.npy"),
            "stroke_points_px": str(out_dir / "stroke_points_px.npy"),
            "timing_s_t": str(out_dir / "timing_s_t.npy"),
            "keypoints_2d_preview": str(preview_path),
        },
        "urls": {
            "keypoints_2d_preview": preview_url,
        },
    }
    (out_dir / "edit_request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    return request

