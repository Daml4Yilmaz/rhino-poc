# Rhino PoC implementation plan

## 1. Product boundary

The PoC targets a metric external skin surface plus six provisional rhinoplasty measurements. It
does not infer cartilage or bone, predict surgical outcomes, or attempt to reproduce Kratos's
proprietary internal architecture.

The no-marker decision remains appropriate for the intended workflow. Metric scale comes from the
iPhone's ARKit trajectory and is cross-checked against LiDAR depth. This is an engineering choice,
not evidence that Kratos uses the same method.

## 2. Reference input

The current benchmark capture is a Stray Scanner export from an iPhone 14 Pro Max:

| Property | Observed value |
|---|---:|
| RGB/pose/depth/confidence records | 1,295 each |
| Actual timestamp duration | 32.003 seconds |
| Effective capture rate | 40.43 fps |
| RGB dimensions | 1,920 × 1,440 |
| ARKit trajectory length | 2.635 m |
| Trajectory span | 0.960 m |
| Orientation coverage | 151.25° |
| Tracking jumps over 15 cm | 0 |
| Focal-length variation | 0.65% |

The MP4 advertises 60 fps but has variable frame intervals. Exported frame IDs and timestamps are
therefore authoritative.

## 3. Pipeline

```text
Stray export
  -> synchronization and capture validation
  -> portrait normalization and sharp frame selection
  -> face masks and background-suppressed images
  -> COLMAP sparse visual refinement
  -> ARKit trajectory scale with LiDAR cross-check
  -> face-only photometric PatchMatch and fusion
  -> Poisson surface and largest-component cleanup
  -> versioned authoritative metric surface
  -> color-corrected UV texture atlas
  -> registered textured GLB
  -> multi-view landmark triangulation
  -> six provisional measurements
```

### 3.1 Ingest

- Require contiguous explicit frame IDs.
- Permit at most a one-frame difference between decoded video and odometry counts.
- Reject non-monotonic timestamps.
- Reject captures with insufficient translation, not merely insufficient camera rotation.
- Rotate portrait pixels and transform intrinsics through the same homogeneous pixel transform.
- Select sharp frames in equal-duration windows rather than equal frame-count windows.

### 3.2 Masking

- Detect one face in every selected image using MediaPipe Tasks with the CPU delegate.
- Dilate the facial convex hull to retain the full skin boundary.
- Write masks and black-background reconstruction images.
- Reject the stage if fewer than 70% or fewer than 60 images contain a face.

### 3.3 Sparse reconstruction

- Use measured per-frame intrinsics and keep them fixed during bundle adjustment.
- Start with sequential matching and ten-frame overlap.
- Select the largest connected sparse model and record its exact path.
- Require at least 60% and at least 40 registered images before dense reconstruction.

ARKit poses are not yet injected as fixed extrinsics. They remain metric references because rolling
shutter and VIO drift make them useful priors but imperfect photogrammetric ground truth.

### 3.4 Metric scale

- RANSAC similarity alignment between matched COLMAP and ARKit camera centers.
- Refit Umeyama similarity on trajectory inliers.
- Report inlier ratio, median residual, and 95th-percentile residual.
- Independently sample high-confidence LiDAR depth at masked sparse correspondences.
- Mark scale unverified when estimates disagree by more than 2%.

Agreement between two phone-derived estimates is not external accuracy proof. Before a clinical
study, reconstruct a calibrated rigid object once and compare recovered dimensions with traceable
physical measurements.

### 3.5 Dense reconstruction

Initial T4 benchmark configuration:

- 120 selected views, 1,400-pixel long dimension;
- at most 96 dense reference views;
- six source images per reference;
- photometric PatchMatch first;
- bounded 2–8 GB cache and cached fusion;
- Poisson depth 9 and largest connected component.

The benchmark fails if total runtime exceeds one hour. Increase quality settings only after a clean
face mesh is produced inside that limit.

### 3.6 Landmarks and measurements

For the surface-only milestone, FLAME is intentionally excluded.

- Detect provisional MediaPipe landmarks in registered views.
- Back-project rays using the measured intrinsics and refined camera poses.
- Robustly triangulate each point and project it to an authoritative triangle.
- Store triangle index and barycentric coordinates for exact surface correspondence.
- Calculate the six requested measurements from centralized definitions.
- Store quality metadata and label definitions provisional.

## 4. Runtime budget

| Stage | Target |
|---|---:|
| Install and capture copy | excluded from algorithm benchmark |
| Ingest and masks | 5 min |
| Sparse reconstruction | 10 min |
| Dense reconstruction and meshing | 35 min |
| Scale, export, landmarks, measurements | 10 min |
| Total | <60 min |

All observed stage times are stored in `case.json`. Estimates must not be reported as benchmark
results.

## 5. Milestones and acceptance gates

### M1 — Reproducible capture processing

- The reference export validates without relying on nominal FPS.
- Portrait images, intrinsics, masks, and LiDAR lookup share one pixel coordinate convention.
- A new Colab runtime can reach sparse reconstruction from the documented notebook.

### M2 — Clean metric face mesh

- Complete on a T4 in less than one hour.
- Face and nasal base are visually continuous.
- Room, torso, and cap do not dominate the retained mesh component.
- Scale JSON contains pose diagnostics and a LiDAR comparison.
- The authoritative PLY, textured GLB, landmarks, and measurements share one geometry identity.
- Every rendered face maps to the same-index authoritative triangle within one micrometre.

### M3 — Provisional measurements

- All ten required landmarks are produced without manual JSON entry.
- Every measurement includes units and definition version.
- No NaN, infinite, or anatomically impossible output is accepted silently.

### M4 — Engineering validation

- Capture at least three repeated scans of one subject.
- Measure a calibrated rigid object through the complete pipeline.
- Record runtime and mesh-quality changes across a small parameter grid.

### M5 — Clinical validation preparation

- Obtain surgeon-approved written landmark and reference-plane definitions.
- Collect blinded manual measurements from two observers.
- Predefine accuracy and repeatability metrics before collecting the study set.

## 6. Known limitations

- The current reference subject appears to follow the phone with their eyes and may move slightly.
- A convex-hull face mask may omit ears and parts of the forehead; it prioritizes nasal measurement.
- Black-background dense reconstruction can produce boundary artifacts that require later refinement.
- MediaPipe landmarks are not validated anatomical annotations.
- LiDAR has low spatial resolution and systematic error; it is a scale check, not the nasal surface.
- Photometric PatchMatch may oversmooth nostrils and alar creases.
- The one-hour target has not yet been demonstrated on a fresh T4 run.

## 7. Deferred work

- custom iOS capture application;
- ordinary Camera-app MOV input for metric output;
- Android support;
- FLAME or another morphable model;
- Gaussian or neural surface reconstruction;
- clinical viewer and material-map refinement;
- subcutaneous anatomy and surgical simulation.
