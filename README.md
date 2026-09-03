# SplineHMR Trajectory-Guided Editing Demo

This is an internal MVP for the SplineHMR oral-presentation demo.

The demo lets a user:

1. load a sequence from the sibling `../SplineHMR/inputs` directory;
2. select a COCO17 joint from a clickable right-side T-pose picker;
3. view the original SMPL/SMPL-X reprojection trajectory, colored by speed from slow to fast;
4. draw a desired 2D trajectory in the browser while preserving the video's original aspect ratio;
5. sample the drawn trajectory into one 2D target per frame;
6. build an edited `keypoints_2d` tensor for Spline-Opt;
7. preview the exact edited 2D keypoint sequence as a skeleton video;
8. optionally run Spline-Opt from the UI.

The implementation follows:

```text
../docs/splinehmr_docs_internal/trajectory_guided_spline_opt_demo_design.md
```

## Why this is a separate directory

`SplineHMR_demo` is intentionally separate from the clean open-source repository at `../SplineHMR`.

It can import the sibling repository, but keeps oral-demo UI code, caches, and experiment outputs isolated.

## Start the web demo

Use the already validated `splinehmr` conda environment:

```bash
cd /root/autodl-tmp/work/SplineHMR/SplineHMR_demo
conda run -n splinehmr python app.py --host 127.0.0.1 --port 7860
```

If you are accessing a remote headless Ubuntu server from a MacBook:

```bash
ssh -L 7860:127.0.0.1:7860 root@<server-ip>
```

Then open this on the MacBook:

```text
http://127.0.0.1:7860
```

You can also expose the server directly:

```bash
conda run -n splinehmr python app.py --host 0.0.0.0 --port 7860
```

For development, SSH tunneling is safer.


## UI notes

- The video canvas preserves the original video aspect ratio exactly. Portrait sequences stay portrait and are centered instead of being stretched or cropped.
- The right-side COCO17 T-pose is the primary joint selector. Clicking a joint updates both the T-pose highlight and the video overlay.
- The selected joint trajectory is drawn over the whole selected frame range. Segment color indicates speed: blue is slow, red is fast.
- After clicking Build, an Edited 2D Keypoints Preview panel appears. This preview is generated from the saved `keypoints_2d_edit.pt`, i.e. the exact tensor sent to Spline-Opt.
- The dropdown under the T-pose is kept as a fallback/debug selector and stays synchronized with the clickable joints.

## Timing modes

The demo currently supports three timing modes.

### Original speed profile

This is the default interaction mode for the MVP. It preserves the original relative speed distribution and always traverses the complete user-drawn curve, so the sampled first/last frames land on the drawn start/end points.

It uses normalized cumulative original motion:

```text
s_t = cumulative_original_speed[t] / total_original_length
target_2d[t] = user_curve(s_t)
```

If the drawn curve length differs from the original trajectory length, absolute pixel speeds are uniformly scaled by:

```text
user_curve_length / original_total_length
```

This is usually the best mode for a live oral-demo interaction: the edited motion keeps the original fast/slow temporal rhythm, but it never freezes at the end just because the user drew a shorter curve.

### Absolute pixel speed

This strict/debug mode advances along the user-drawn curve by the same pixel distance as the original SMPL reprojection joint trajectory for each frame interval:

```text
target_step_length[t] = ||orig_2d[t] - orig_2d[t-1]||
```

It preserves absolute 2D speed only when the drawn curve length matches the original trajectory length. If the drawn curve is longer, sampling may stop before the endpoint; if it is shorter, sampling may clamp at the endpoint. The API reports this in `timing_report`.

### Uniform

Uniformly samples the drawn curve by frame index. This is mostly a debugging baseline.

## Output package

Each edit writes:

```text
outputs/trajectory_edit/<edit_id>/
  edit_request.json
  keypoints_2d_edit.pt
  keypoints_2d_original.pt
  original_joint_trajectory.npy
  sampled_target_trajectory.npy
  stroke_points_norm.npy
  stroke_points_px.npy
  timing_s_t.npy
  keypoints_2d_preview.mp4
```

`keypoints_2d_preview.mp4` overlays the exact edited COCO17 tensor on the input video before Spline-Opt runs. Use it to verify that the hand-drawn trajectory and all unedited joints form the intended 2D input sequence.

If Spline-Opt is run, it writes:

```text
outputs/trajectory_edit/<edit_id>/spline_opt/
  hmr4d_results_before.pt
  hmr4d_results.pt
  spline_opt_report.json
  optional render videos
```

When `Render result after optimization` is enabled, the frontend displays `render_overlay_compare_annotated.mp4` when available. This annotated render keeps the red/green before/after SMPL overlay and additionally draws the selected joint's original 2D trajectory, edited target 2D trajectory, and current-frame highlighted edited joint target.

## API endpoints

```text
GET  /api/sequences
GET  /api/sequence/<name>/meta
POST /api/edit
POST /api/run_spline_opt
```

`/api/sequence/<name>/meta` computes and caches original COCO17 reprojection from the existing SMPL/SMPL-X sequence. If body-model reprojection fails, it falls back to `preprocess/vitpose.pt` and returns a warning.

## MVP caveats

- Only one selected COCO17 joint is edited at a time.
- The browser UI currently supports one continuous stroke.
- The first version does not implement manual keyframe timing anchors.
- The first version prioritizes making the interaction and edited-keypoint generation reliable.
- Live Spline-Opt can be slow; for oral presentation, cache polished examples.
- The Spline-Opt runner uses `../SplineHMR/configs/spline_opt.yaml` by default. Leave the max-iter field empty or set it to 60 to match the validated open-source demo configuration.

