from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .paths import CACHE_ROOT, INPUTS_ROOT, SPLINEHMR_ROOT, add_splinehmr_to_path


COCO17_JOINTS: list[dict[str, Any]] = [
    {"index": 0, "name": "nose", "label": "Nose"},
    {"index": 1, "name": "left_eye", "label": "Left eye"},
    {"index": 2, "name": "right_eye", "label": "Right eye"},
    {"index": 3, "name": "left_ear", "label": "Left ear"},
    {"index": 4, "name": "right_ear", "label": "Right ear"},
    {"index": 5, "name": "left_shoulder", "label": "Left shoulder"},
    {"index": 6, "name": "right_shoulder", "label": "Right shoulder"},
    {"index": 7, "name": "left_elbow", "label": "Left elbow"},
    {"index": 8, "name": "right_elbow", "label": "Right elbow"},
    {"index": 9, "name": "left_wrist", "label": "Left wrist"},
    {"index": 10, "name": "right_wrist", "label": "Right wrist"},
    {"index": 11, "name": "left_hip", "label": "Left hip"},
    {"index": 12, "name": "right_hip", "label": "Right hip"},
    {"index": 13, "name": "left_knee", "label": "Left knee"},
    {"index": 14, "name": "right_knee", "label": "Right knee"},
    {"index": 15, "name": "left_ankle", "label": "Left ankle"},
    {"index": 16, "name": "right_ankle", "label": "Right ankle"},
]

JOINT_NAME_TO_INDEX = {j["name"]: int(j["index"]) for j in COCO17_JOINTS}

COCO17_EDGES = [
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 6],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
]


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "fps": float(self.fps),
            "frame_count": int(self.frame_count),
            "duration_sec": float(self.duration_sec),
        }


def sequence_dir(name: str) -> Path:
    safe = Path(name).name
    path = INPUTS_ROOT / safe
    if not path.exists():
        raise FileNotFoundError(f"Unknown sequence: {name}")
    return path


def list_sequences() -> list[dict[str, Any]]:
    if not INPUTS_ROOT.exists():
        return []
    out = []
    for d in sorted(p for p in INPUTS_ROOT.iterdir() if p.is_dir()):
        video = d / "0_input_video.mp4"
        hmr = d / "hmr4d_results.pt"
        if video.exists() and hmr.exists():
            meta = read_video_meta(video)
            out.append(
                {
                    "name": d.name,
                    "video": str(video),
                    "frame_count": meta.frame_count,
                    "fps": meta.fps,
                    "width": meta.width,
                    "height": meta.height,
                }
            )
    return out


def read_video_meta(video_path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return VideoMeta(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=(frame_count / fps if fps > 0 else 0.0),
    )


def input_video_path(name: str) -> Path:
    path = sequence_dir(name) / "0_input_video.mp4"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_hmr(name: str) -> dict[str, Any]:
    path = sequence_dir(name) / "hmr4d_results.pt"
    return torch.load(str(path), map_location="cpu")


def load_vitpose(name: str) -> torch.Tensor:
    path = sequence_dir(name) / "preprocess" / "vitpose.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(str(path), map_location="cpu").float()


def load_bbx_xys(name: str) -> torch.Tensor:
    path = sequence_dir(name) / "preprocess" / "bbx.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    bbx = torch.load(str(path), map_location="cpu")
    if not isinstance(bbx, dict) or "bbx_xys" not in bbx:
        raise KeyError(f"{path} must contain bbx_xys")
    return bbx["bbx_xys"].float()


def _projection_cache_path(name: str, source: str) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"{name}_{source}_coco17.pt"


def _load_spline_opt_use_smpl_flag() -> bool:
    cfg_path = SPLINEHMR_ROOT / "configs" / "spline_opt.yaml"
    if not cfg_path.exists():
        return False
    try:
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        return bool((cfg.get("cfg", {}) or {}).get("use_smpl", False))
    except Exception:
        return False


def compute_smpl_reprojected_coco17(name: str, *, device: str = "cuda") -> torch.Tensor:
    """Compute original SMPL/SMPL-X COCO17 reprojection used for trajectory editing."""
    add_splinehmr_to_path()
    from multi_view_smpl_optimizer.utils.bspline_body_pose_refiner import BsplineRefineConfig, _make_coco17_model
    from multi_view_smpl_optimizer.utils.geo_transform import project_p2d

    hmr = load_hmr(name)
    params = hmr["smpl_params_incam"]
    T = int(params["body_pose"].shape[0])
    use_smpl = _load_spline_opt_use_smpl_flag()
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    cfg = BsplineRefineConfig(use_smpl=use_smpl).resolve()
    model = _make_coco17_model(cfg, dev)
    with torch.no_grad():
        joints3d = model(
            body_pose=params["body_pose"][:T].to(dev).float(),
            betas=params["betas"][:T].to(dev).float(),
            global_orient=params["global_orient"][:T].to(dev).float(),
            transl=params["transl"][:T].to(dev).float(),
        )
        joints2d = project_p2d(joints3d, K=hmr["K_fullimg"][:T].to(dev).float())
    conf = torch.ones((T, joints2d.shape[1], 1), dtype=torch.float32)
    return torch.cat([joints2d.detach().cpu().float(), conf], dim=-1)


def get_original_coco17(
    name: str,
    *,
    source: str = "smpl_reproj",
    device: str = "cuda",
    allow_fallback: bool = True,
) -> tuple[torch.Tensor, str, str | None]:
    """Return original COCO17 keypoints for UI and Spline-Opt input.

    Args:
        source: "smpl_reproj" or "vitpose".

    Returns:
        keypoints: (T,17,3)
        actual_source: source actually used
        warning: optional fallback warning
    """
    if source == "vitpose":
        return load_vitpose(name), "vitpose", None

    cache = _projection_cache_path(name, "smpl_reproj")
    if cache.exists():
        return torch.load(str(cache), map_location="cpu").float(), "smpl_reproj", None

    try:
        kp = compute_smpl_reprojected_coco17(name, device=device)
        torch.save(kp, cache)
        return kp, "smpl_reproj", None
    except Exception as exc:
        if not allow_fallback:
            raise
        try:
            kp = load_vitpose(name)
            return kp, "vitpose", f"SMPL reprojection failed; fell back to VitPose. Error: {exc!r}"
        except Exception:
            raise exc


def sequence_meta(name: str, *, device: str = "cuda") -> dict[str, Any]:
    video = input_video_path(name)
    meta = read_video_meta(video)
    kp, source, warning = get_original_coco17(name, device=device)
    T = min(int(kp.shape[0]), meta.frame_count)
    kp = kp[:T]
    return {
        "name": Path(name).name,
        "video_url": f"/media/sequence/{Path(name).name}/0_input_video.mp4",
        "video": meta.to_dict(),
        "num_frames": int(T),
        "joint_format": "coco17",
        "joints": COCO17_JOINTS,
        "edges": COCO17_EDGES,
        "trajectory_source": source,
        "warning": warning,
        "keypoints_2d": kp[:, :, :2].numpy().astype(float).tolist(),
        "keypoints_conf": kp[:, :, 2].numpy().astype(float).tolist(),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

