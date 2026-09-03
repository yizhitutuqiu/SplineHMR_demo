from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Literal

import numpy as np


TimingMode = Literal["original_speed", "absolute_speed", "uniform"]
KeyframeInterpolationMode = Literal["linear", "bezier", "bspline"]


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


def _segment_speed_progress(original_joint_xy: np.ndarray, start: int, end: int, *, eps: float = 1e-6) -> np.ndarray:
    """Normalized cumulative original speed for inclusive frame segment [start, end]."""
    if end <= start:
        return np.asarray([0.0], dtype=np.float64)
    pts = np.asarray(original_joint_xy[start : end + 1], dtype=np.float64)
    speed = np.linalg.norm(pts[1:] - pts[:-1], axis=-1)
    accum = np.concatenate([[0.0], np.cumsum(speed)])
    total = float(accum[-1])
    if total <= eps:
        return np.linspace(0.0, 1.0, end - start + 1, dtype=np.float64)
    return accum / total


def _catmull_rom_point(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64).reshape(-1, 1)
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


def _bezier_segment_point(points: np.ndarray, i: int, u: np.ndarray) -> np.ndarray:
    n = int(points.shape[0])
    p0 = points[max(0, i - 1)]
    p1 = points[i]
    p2 = points[i + 1]
    p3 = points[min(n - 1, i + 2)]
    # Cubic Bezier controls converted from a cardinal/Catmull-Rom tangent estimate.
    c1 = p1 + (p2 - p0) / 6.0
    c2 = p2 - (p3 - p1) / 6.0
    u = np.asarray(u, dtype=np.float64).reshape(-1, 1)
    v = 1.0 - u
    return (v ** 3) * p1 + 3.0 * (v ** 2) * u * c1 + 3.0 * v * (u ** 2) * c2 + (u ** 3) * p2


def _natural_cubic_spline_eval(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Evaluate a natural cubic interpolating spline y(x) at xq. No scipy dependency."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    xq = np.asarray(xq, dtype=np.float64).reshape(-1)
    n = int(x.shape[0])
    if n < 2:
        raise ValueError("Need at least two spline knots")
    if n == 2:
        return np.interp(xq, x, y)
    h = np.diff(x)
    if np.any(h <= 1e-9):
        raise ValueError("Spline knot parameters must be strictly increasing")
    A = np.zeros((n, n), dtype=np.float64)
    rhs = np.zeros((n,), dtype=np.float64)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    m = np.linalg.solve(A, rhs)
    xqc = np.clip(xq, x[0], x[-1])
    idx = np.searchsorted(x, xqc, side="right") - 1
    idx = np.clip(idx, 0, n - 2)
    hi = x[idx + 1] - x[idx]
    a = (x[idx + 1] - xqc) / hi
    b = (xqc - x[idx]) / hi
    return (
        a * y[idx]
        + b * y[idx + 1]
        + ((a ** 3 - a) * m[idx] + (b ** 3 - b) * m[idx + 1]) * (hi ** 2) / 6.0
    )


def _bspline_like_interp(points: np.ndarray, segment_index: int, u: np.ndarray) -> np.ndarray:
    """Interpolating cubic spline used for the UI's B-spline mode; passes all keyframes."""
    n = int(points.shape[0])
    if n <= 2:
        p1 = points[segment_index]
        p2 = points[segment_index + 1]
        return p1[None, :] * (1.0 - u.reshape(-1, 1)) + p2[None, :] * u.reshape(-1, 1)
    # chord-length parameterization is less loopy than uniform when keyframes are unevenly spaced.
    seg = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    q = np.concatenate([[0.0], np.cumsum(np.maximum(seg, 1e-6))])
    xq = q[segment_index] + np.asarray(u, dtype=np.float64).reshape(-1) * (q[segment_index + 1] - q[segment_index])
    xs = _natural_cubic_spline_eval(q, points[:, 0], xq)
    ys = _natural_cubic_spline_eval(q, points[:, 1], xq)
    return np.stack([xs, ys], axis=-1)


def _eval_keyframe_segment(points: np.ndarray, i: int, u: np.ndarray, mode: KeyframeInterpolationMode) -> np.ndarray:
    if mode == "linear" or len(points) <= 2:
        p1 = points[i]
        p2 = points[i + 1]
        uu = np.asarray(u, dtype=np.float64).reshape(-1, 1)
        return p1[None, :] * (1.0 - uu) + p2[None, :] * uu
    if mode == "bezier":
        return _bezier_segment_point(points, i, u)
    if mode == "bspline":
        return _bspline_like_interp(points, i, u)
    raise ValueError(f"Unknown keyframe interpolation mode: {mode}")


def _approx_curve_length_from_keyframes(points: np.ndarray, mode: KeyframeInterpolationMode) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        dense_u = np.linspace(0.0, 1.0, 64, dtype=np.float64)
        dense = _eval_keyframe_segment(points, i, dense_u, mode)
        total += float(np.linalg.norm(dense[1:] - dense[:-1], axis=-1).sum())
    return total


def sample_keyframes_with_original_speed(
    *,
    keyframe_frames: Iterable[int],
    keyframe_xy: np.ndarray,
    original_joint_xy: np.ndarray,
    interpolation: KeyframeInterpolationMode = "linear",
) -> SampledTrajectory:
    """Interpolate sparse keyframes while preserving original speed profile within each keyframe span.

    `keyframe_frames` are relative frame indices inside `original_joint_xy` and must include 0 and T-1.
    Each keyframe position is hit exactly. For frames between two adjacent keyframes, progress along the
    chosen geometric curve is driven by the original joint's normalized cumulative speed in that same span.
    """
    orig = _as_points(original_joint_xy)
    kxy = _as_points(keyframe_xy)
    frames = np.asarray(list(keyframe_frames), dtype=np.int64).reshape(-1)
    if frames.shape[0] != kxy.shape[0]:
        raise ValueError("keyframe_frames and keyframe_xy must have the same length")
    if frames.shape[0] < 2:
        raise ValueError("At least first and last keyframes are required")
    order = np.argsort(frames)
    frames = frames[order]
    kxy = kxy[order]
    if np.any(np.diff(frames) <= 0):
        raise ValueError("Keyframe frame indices must be unique and increasing")
    T = int(orig.shape[0])
    if int(frames[0]) != 0 or int(frames[-1]) != T - 1:
        raise ValueError(f"Keyframe mode requires keyframes at relative frames 0 and {T - 1}")
    if int(frames[0]) < 0 or int(frames[-1]) >= T:
        raise ValueError(f"Keyframes must lie in [0, {T - 1}]")
    mode = str(interpolation).lower()
    if mode not in ("linear", "bezier", "bspline"):
        raise ValueError(f"Unknown interpolation '{interpolation}'")

    target = np.zeros((T, 2), dtype=np.float64)
    s_all = np.zeros((T,), dtype=np.float64)
    for seg_i in range(len(frames) - 1):
        f0 = int(frames[seg_i])
        f1 = int(frames[seg_i + 1])
        progress = _segment_speed_progress(orig, f0, f1)
        pts = _eval_keyframe_segment(kxy, seg_i, progress, mode)  # (f1-f0+1,2)
        if seg_i > 0:
            target[f0 + 1 : f1 + 1] = pts[1:]
            s_all[f0 + 1 : f1 + 1] = progress[1:]
        else:
            target[f0 : f1 + 1] = pts
            s_all[f0 : f1 + 1] = progress
    # hard snap keyframes to avoid floating-point drift
    for f, p in zip(frames.tolist(), kxy):
        target[int(f)] = p

    orig_speed = np.linalg.norm(orig[1:] - orig[:-1], axis=-1)
    target_speed = np.linalg.norm(target[1:] - target[:-1], axis=-1)
    orig_len = float(orig_speed.sum())
    curve_len = _approx_curve_length_from_keyframes(kxy, mode)  # geometric length of the interpolated keyframe path
    speed_scale = float(curve_len / orig_len) if orig_len > 1e-6 else float("nan")
    report = TimingReport(
        mode=f"keyframe_{mode}_original_speed",
        num_frames=int(T),
        original_total_length_px=float(orig_len),
        user_curve_length_px=float(curve_len),
        speed_scale=float(speed_scale) if np.isfinite(speed_scale) else float("nan"),
        absolute_speed_exact=False,
        reaches_curve_end=True,
        note=(
            f"Keyframe mode with {mode} interpolation: every keyframe is hit exactly; frames between adjacent "
            "keyframes use the original joint's normalized speed profile within that span."
        ),
    )
    return SampledTrajectory(
        target_2d=target.astype(np.float32),
        s_t=s_all.astype(np.float32),
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

