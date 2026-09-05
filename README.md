# SplineHMR Trajectory-Guided Editing Demo

This is an internal MVP for the SplineHMR oral-presentation demo.

The demo lets a user:

1. load a sequence from the sibling `../SplineHMR/inputs` directory, or upload a new video and initialize it with GVHMR;
2. select COCO17 joints from a clickable right-side T-pose picker;
3. view the original SMPL/SMPL-X reprojection trajectory of the active joint, colored by speed from slow to fast;
4. edit one or more joints with browser strokes or sparse keyframes while preserving the video's original aspect ratio;
5. sample every edited joint into one 2D target per frame;
6. build a multi-joint edited `keypoints_2d` tensor for Spline-Opt;
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


## Uploading a new video with GVHMR

The upload panel starts a separate GVHMR initialization workflow:

1. the browser uploads the selected video to `outputs/gvhmr_uploads/<job_id>/upload/`;
2. the backend runs:

```bash
conda run --no-capture-output -n gvhmr python tools/demo/demo.py \
  --video <uploaded_video.mp4> \
  --output_root outputs/gvhmr_uploads/<job_id>/gvhmr_outputs
```

with working directory `../third_party/GVHMR`;
3. progress is shown live in the UI using GVHMR log keywords such as preprocess, HMR4D inference, render, and merge;
4. on success, the generated GVHMR package is copied into the normal SplineHMR layout:

```text
../SplineHMR/inputs/<upload_sequence>/
  0_input_video.mp4
  hmr4d_results.pt
  preprocess/bbx.pt
  preprocess/vitpose.pt

../SplineHMR/outputs/<upload_sequence>/spline-opt/render_before.mp4
```

The frontend then automatically selects the new sequence. From that point onward, trajectory editing, 2D preview, Spline-Opt, rendering, and download buttons are exactly the same as for the bundled examples.

Requirements: `../third_party/GVHMR` must be present and runnable in the `gvhmr` conda environment, including its model/body assets.

## UI notes

- The main viewer has two synchronized windows: the left window is a non-interactive source-body preview. It uses `../SplineHMR/outputs/<seq>/spline-opt/render_before.mp4` when available and falls back to the input video with SMPL/SMPL-X reprojection skeleton. The right window is the editable target canvas.
- Both windows preserve the original video aspect ratio exactly. Portrait sequences stay portrait and are centered instead of being stretched or cropped.
- The right-side COCO17 T-pose is the primary joint selector. Clicking a joint updates the active joint without deleting edits already made on other joints.
- The selected joint trajectory is drawn over the whole selected frame range. Segment color indicates speed: blue is slow, red is fast. Joints that already have edits stay marked in the T-pose, but the editable canvas only visualizes the currently selected joint to keep the drawing area clean.
- After clicking Build, an Edited 2D Keypoints Preview panel appears. This preview is generated from the saved `keypoints_2d_edit.pt`, i.e. the exact tensor sent to Spline-Opt.
- The debug report is collapsed by default to keep the live demo compact; expand it only when inspecting request IDs or optimizer stats.
- Download buttons are shown for generated videos, including the edited-2D preview and rendered before/after/compare/annotated outputs.
- The dropdown under the T-pose is kept as a fallback/debug selector and stays synchronized with the clickable joints.

## Edit modes

The demo supports two ways to define the desired 2D trajectory for each selected joint. You can edit several joints in one request: finish a stroke or keyframe set for one joint, click another joint in the T-pose, edit it, and then Build once.

### Stroke mode

Draw a complete desired path directly on the video. The backend samples one target point per frame using the selected timing mode.

### Keyframe mode

Enable the Keyframes switch, scrub the video to a frame, and click the editable right-hand video window to set the selected joint target at that frame. Clicking the same frame again replaces the keyframe. The keyframe list is collapsed by default to keep the UI compact. At least two keyframes are required, but they do not need to include the selected frame range boundary. If the first/last keyframe lies inside the selected range, only that partial interval is strongly edited; frames outside the keyframe interval keep the original 2D reprojection with weak anchor confidence.

Available geometric interpolation modes are:

- `Linear`: piecewise straight segments between keyframes;
- `Bezier`: smooth cubic Bezier segments with automatically estimated tangents;
- `B-spline`: an interpolating cubic spline style curve that passes all keyframes.

Within each adjacent keyframe span, intermediate frames are sampled using that span's original SMPL-reprojection speed profile, normalized to hit the next keyframe exactly. This keeps the original fast/slow rhythm while guaranteeing keyframe constraints.

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

## Stopping long-running jobs and live progress

Both long-running backends are controlled as cancelable jobs.

- GVHMR upload initialization runs in a subprocess under the `gvhmr` conda environment. The UI shows coarse stages from the live GVHMR log and provides `Stop GVHMR`, which terminates the whole process group.
- Spline-Opt runs in a separate `splinehmr` Python subprocess instead of blocking the HTTP request. The UI shows live optimizer closure progress, loss values, render frame progress, and annotated-render frame progress. `Stop Spline-Opt` terminates the whole process group and returns the UI to an editable state.

The progress logs are written under:

```text
outputs/gvhmr_uploads/<job_id>/gvhmr.log
outputs/spline_opt_jobs/<job_id>/spline_opt.log
```

PyTorch LBFGS may evaluate the closure more times than the nominal `max_iter` because of line search. The UI therefore treats `optimizer closure=i/max_iter` as a real optimizer activity indicator rather than a mathematically exact percentage of all internal function evaluations.

## Optional B-spline optimization parameters

The Run panel contains a collapsed `B-spline optimization parameters` section. Leave fields empty to use the validated defaults. Filled fields are sent as a per-run override and recorded in `spline_opt_report.json` under `stats.bspline_overrides` and `stats.bspline_effective_config`.

Current UI defaults are:

- `degree`: `3`;
- `m_per_t`: `fps_div_2`;
- `confidence threshold`: `0.3`;
- `confidence power`: `4.0` for trajectory-edit runs;
- `body prior weight`: `0.02` for trajectory-edit runs;
- `smoothness weight`: `0.02`;
- `LBFGS lr`: `1.0`;
- `line_search_fn`: `strong_wolfe`;
- `learn_knots`: `false`;
- `knot_min_gap`: `0.001`;
- `knot_pos_w`: `1.0`;
- `knot_gap_w`: `0.2`.

For live demos, the safest knobs are usually `m_per_t`, `body prior weight`, and `smoothness weight`: smaller priors make edits more visible, while larger smoothness weights make motion steadier but can resist sharp intended changes.

## Trajectory-edit optimization defaults

The interactive demo intentionally uses a trajectory-edit-specific Spline-Opt setting rather than the public batch demo defaults:

- the edited joint is strongly supervised (`edit_conf=1`);
- unedited joints are kept as weak soft anchors (`base_conf=0.5`);
- neighboring joints receive a medium anchor (`neighbor_conf=0.75`);
- confidence weighting uses a sharper power (`conf_power=4`) so the edited joint still dominates while anchors discourage unrelated drift;
- global orientation and translation are fixed;
- static-motion regularization is disabled for the edit run;
- body-pose prior is mildly relaxed for visible local edits.

This avoids two failure modes at once: exact self-reprojection labels no longer pin the original pose in place, and unconstrained non-edited joints are less likely to drift. An optional deterministic `anchor_noise_px` parameter exists for debugging pseudo-label jitter, but the default is zero for repeatable live demos. The regular open-source `SplineHMR` demo configuration is not modified.

## Output package

Each edit writes:

```text
outputs/trajectory_edit/<edit_id>/
  edit_request.json
  keypoints_2d_edit.pt
  keypoints_2d_original.pt
  original_joint_trajectory.npy              # first edited joint, legacy compatibility
  sampled_target_trajectory.npy             # first edited joint, legacy compatibility
  original_joint_trajectories.npy           # all edited joints, shape (N,T,2)
  sampled_target_trajectories.npy           # all edited joints, shape (N,T,2)
  edited_joint_indices.npy
  <joint>_stroke_points_norm.npy            # when that joint uses stroke mode
  <joint>_stroke_points_px.npy
  <joint>_keyframes_norm.npy                # when that joint uses keyframe mode
  <joint>_keyframes_px.npy
  timing_s_t.npy                            # first edited joint, legacy compatibility
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

When `Render result after optimization` is enabled, the frontend displays `render_overlay_compare_annotated.mp4` when the `Show 2D trajectories` switch is checked, and the clean `render_overlay_compare.mp4` when it is unchecked. The `Download shown video` button always downloads the currently displayed version. The annotated render keeps the red/green before/after SMPL overlay and additionally draws every supervised joint's original 2D trajectory, edited target 2D trajectory, and current-frame highlighted target. Its legend is rendered in a separate right-side panel, so labels do not cover the person or the trajectory curves.

## API endpoints

```text
GET  /api/sequences
GET  /api/sequence/<name>/meta
POST /api/upload_video
GET  /api/gvhmr_job/<job_id>
POST /api/gvhmr_job/<job_id>/stop
POST /api/edit
POST /api/run_spline_opt
GET  /api/spline_opt_job/<job_id>
POST /api/spline_opt_job/<job_id>/stop
```

`/api/sequence/<name>/meta` computes and caches original COCO17 reprojection from the existing SMPL/SMPL-X sequence. If body-model reprojection fails, it falls back to `preprocess/vitpose.pt` and returns a warning.

## MVP caveats

- Multiple COCO17 joints can be supervised in the same Spline-Opt request; each joint independently stores either one continuous stroke or a keyframe sequence.
- The UI currently supports one stroke per joint. Press Reset edit to clear only the active joint.
- Keyframe interpolation preserves the original speed profile inside each adjacent keyframe span, while hitting all user keyframes exactly.
- The first version prioritizes making the interaction and edited-keypoint generation reliable.
- Live Spline-Opt can be slow; for oral presentation, cache polished examples.
- The Spline-Opt runner uses `../SplineHMR/configs/spline_opt.yaml` by default. Leave the max-iter field empty or set it to 60 to match the validated open-source demo configuration.

