from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Literal

import numpy as np


TimingMode = Literal["original_speed", "absolute_speed", "uniform"]


@dataclass
class TimingReport:
    mode: str
    num_frames: int
    original_total_length_px: float
    user_curve_length_px: float
    speed_scale: float
    absolute_speed_exact: bool
    reaches_curve_end: bool
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SampledTrajectory:
    target_2d: np.ndarray
    s_t: np.ndarray
    target_speed_px: np.ndarray
    original_speed_px: np.ndarray
    report: TimingReport


def _as_points(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(points), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N,2), got {arr.shape}")
    if len(arr) < 2:
        raise ValueError("At least two stroke points are required")
    if not np.isfinite(arr).all():
        raise ValueError("Stroke points contain NaN or Inf")
    return arr


def remove_near_duplicates(points: np.ndarray, min_dist: float = 1.5) -> np.ndarray:
    points = _as_points(points)
    keep = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - keep[-1]) >= float(min_dist):
            keep.append(p)
    if len(keep) < 2:
        raise ValueError("Stroke is too short after duplicate removal")
    return np.asarray(keep, dtype=np.float64)


def smooth_polyline(points: np.ndarray, window: int = 5) -> np.ndarray:
    points = _as_points(points)
    window = int(window)
    if window <= 1 or len(points) < 5:
        return points
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    out = np.stack(
        [
            np.convolve(padded[:, 0], kernel, mode="valid"),
            np.convolve(padded[:, 1], kernel, mode="valid"),
        ],
        axis=-1,
    )
    out[0] = points[0]
    out[-1] = points[-1]
    return out


def preprocess_stroke(
    points: Iterable[Iterable[float]],
    *,
    min_dist: float = 1.5,
    smooth: bool = True,
    smooth_window: int = 5,
) -> np.ndarray:
    """Clean a user stroke represented in video-pixel coordinates."""
    cleaned = remove_near_duplicates(_as_points(points), min_dist=min_dist)
    if smooth:
        cleaned = smooth_polyline(cleaned, window=smooth_window)
        cleaned = remove_near_duplicates(cleaned, min_dist=max(0.5, min_dist * 0.5))
    return cleaned


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    points = _as_points(points)
    seg = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def polyline_length(points: np.ndarray) -> float:
    return float(cumulative_lengths(points)[-1])


def sample_polyline_at_distances(points: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample a polyline by cumulative arc-length distances.

    Returns:
        sampled_xy: (T,2)
        s_t: normalized curve parameter in [0,1]
    """
    points = _as_points(points)
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    lengths = cumulative_lengths(points)
    total = float(lengths[-1])
    if total <= 1e-8:
        raise ValueError("User curve length is nearly zero")
    d = np.clip(distances, 0.0, total)
    x = np.interp(d, lengths, points[:, 0])
    y = np.interp(d, lengths, points[:, 1])
    return np.stack([x, y], axis=-1), d / total


def original_speed_timing(orig_2d: np.ndarray, *, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, float]:
    """Return normalized cumulative progress using original per-frame speeds."""
    orig = _as_points(orig_2d)
    speed = np.linalg.norm(orig[1:] - orig[:-1], axis=-1)
    accum = np.concatenate([[0.0], np.cumsum(speed)])
    total = float(accum[-1])
    if total < eps:
        s_t = np.linspace(0.0, 1.0, len(orig), dtype=np.float64)
    else:
        s_t = accum / total
    return s_t, speed, total


def uniform_timing(num_frames: int) -> np.ndarray:
    if int(num_frames) < 2:
        raise ValueError("num_frames must be >= 2")
    return np.linspace(0.0, 1.0, int(num_frames), dtype=np.float64)


def sample_user_curve_with_timing(
    *,
    user_curve_xy: np.ndarray,
    original_joint_xy: np.ndarray,
    mode: TimingMode = "original_speed",
) -> SampledTrajectory:
    """Sample a drawn 2D curve into one target point per frame.

    Modes:
        original_speed:
            Preserves the original *speed distribution* exactly and traverses the
            full user curve. If the user curve length differs from the original
            trajectory length, absolute pixel speeds are uniformly scaled by
            L_user / L_original. This is the recommended oral-demo default.

        absolute_speed:
            Uses original per-frame pixel distances directly along the user curve.
            This preserves absolute speed whenever the curve is long enough, but
            may stop before the user endpoint or clamp at the endpoint when the
            curve length differs. The report explicitly records this.

        uniform:
            Ignores original speeds and samples uniformly along the curve.
    """
    curve = _as_points(user_curve_xy)
    orig = _as_points(original_joint_xy)
    if len(orig) < 2:
        raise ValueError("Need at least two original frames")

    T = len(orig)
    curve_len = polyline_length(curve)
    s_orig, orig_speed, orig_len = original_speed_timing(orig)

    if mode == "uniform":
        s_t = uniform_timing(T)
        distances = s_t * curve_len
        sampled, s_used = sample_polyline_at_distances(curve, distances)
        note = "Uniform arc-length sampling; original timing is ignored."
        speed_scale = float("nan")
        exact = False
        reaches = True
    elif mode == "absolute_speed":
        accum = np.concatenate([[0.0], np.cumsum(orig_speed)])
        sampled, s_used = sample_polyline_at_distances(curve, accum)
        reaches = bool(abs(float(s_used[-1]) - 1.0) < 1e-4)
        exact = bool(orig_len <= curve_len + 1e-6)
        speed_scale = 1.0
        if abs(curve_len - orig_len) < 1e-4:
            note = "Absolute per-frame pixel speeds match the original trajectory and the curve endpoint is reached."
        elif curve_len > orig_len:
            note = (
                "Absolute per-frame pixel speeds are preserved, but the user curve is longer than the original "
                "trajectory, so the sampled path stops before the drawn endpoint."
            )
        else:
            note = (
                "The user curve is shorter than the original trajectory. Sampling clamps to the curve endpoint "
                "after the accumulated original distance exceeds the curve length."
            )
    else:
        s_t = s_orig
        distances = s_t * curve_len
        sampled, s_used = sample_polyline_at_distances(curve, distances)
        if orig_len <= 1e-6:
            speed_scale = float("nan")
            note = "Original trajectory is nearly static; fell back to uniform progress."
        else:
            speed_scale = curve_len / orig_len
            note = (
                "Original-speed timing: preserves the original frame-to-frame speed profile and traverses the full "
                "user curve. Absolute pixel speeds are scaled by user_curve_length / original_total_length."
            )
        exact = bool(abs(speed_scale - 1.0) < 1e-3) if np.isfinite(speed_scale) else False
        reaches = True

    target_speed = np.linalg.norm(sampled[1:] - sampled[:-1], axis=-1)
    report = TimingReport(
        mode=str(mode),
        num_frames=int(T),
        original_total_length_px=float(orig_len),
        user_curve_length_px=float(curve_len),
        speed_scale=float(speed_scale) if np.isfinite(speed_scale) else float("nan"),
        absolute_speed_exact=bool(exact),
        reaches_curve_end=bool(reaches),
        note=note,
    )
    return SampledTrajectory(
        target_2d=sampled.astype(np.float32),
        s_t=np.asarray(s_used, dtype=np.float32),
        target_speed_px=target_speed.astype(np.float32),
        original_speed_px=orig_speed.astype(np.float32),
        report=report,
    )


def normalized_to_pixels(points_norm: Iterable[Iterable[float]], video_size: tuple[int, int]) -> np.ndarray:
    """Convert normalized canvas/video coordinates to video pixel coordinates.

    Args:
        points_norm: (N,2), each coordinate in [0,1].
        video_size: (width, height).
    """
    pts = _as_points(points_norm)
    w, h = map(float, video_size)
    out = pts.copy()
    out[:, 0] *= w
    out[:, 1] *= h
    return out


def pixels_to_normalized(points_px: np.ndarray, video_size: tuple[int, int]) -> np.ndarray:
    pts = _as_points(points_px)
    w, h = map(float, video_size)
    out = pts.copy()
    out[:, 0] /= max(w, 1.0)
    out[:, 1] /= max(h, 1.0)
    return out

