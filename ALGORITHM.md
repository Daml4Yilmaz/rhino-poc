# Facial reconstruction algorithm

## 1. Scope and central invariant

The current system reconstructs an external facial skin surface from a complete iPhone Stray
Scanner export. It produces:

- one metric, authoritative triangle mesh;
- one registered photorealistic rendering of that same surface;
- ten provisional surface landmarks;
- six provisional rhinoplasty measurements;
- input, stage, output, and repeatability quality reports.

It does not reconstruct cartilage, bone, or other subcutaneous anatomy. It does not predict a
surgical result. It is research software and is not validated for clinical use.

The most important architectural invariant is:

```text
                        one captured subject
                                 |
                                 v
                    authoritative metric mesh
                       face_geometry.ply
                         /             \
                        /               \
              measurement engine      renderer
                  landmarks +          same triangles
                  distances +          + UV albedo
                    angles             + PBR material
```

The renderer does not receive an independently generated face. Visual texture and material layers
enhance the authoritative surface without silently changing the coordinates used for measurement.

## 2. End-to-end pipeline

The CLI executes ten ordered stages:

```text
Stray Scanner export
  |
  +-- ingest ------ synchronization, video metrics, frame selection, pixel transforms
  +-- mask -------- MediaPipe face localization and binary reconstruction masks
  +-- sfm --------- SIFT matching, visual camera poses, sparse COLMAP model
  +-- scale ------- ARKit trajectory scale with an independent LiDAR check
  +-- mvs --------- PatchMatch depth, fusion, Poisson surface reconstruction
  +-- texture ----- UV atlas from original, unmasked, registered RGB views
  +-- export ------ authoritative metric PLY and registered PBR GLB
  +-- landmarks --- multi-view ray triangulation and surface consensus
  +-- measure ----- six provisional geometric measurements
  +-- quality ----- versioned PASS/WARN/FAIL report
```

Default Colab parameters select 120 frames, resize the long image dimension to 1,400 pixels, use
at most 96 PatchMatch reference views and six source views per reference, and use Poisson depth 9.
The target runtime is below one hour on an NVIDIA T4.

## 3. Input model and synchronization

### 3.1 Required data

The metric pipeline expects one directory containing:

```text
capture/
├── rgb.mp4
├── odometry.csv
├── camera_matrix.csv
├── depth/          # optional but strongly preferred
└── confidence/     # optional
```

For frame `i`, the normalized capture record contains:

- explicit integer frame identifier `i`;
- timestamp `t_i`;
- ARKit camera center `c_i` in metres;
- ARKit quaternion `q_i = (q_x, q_y, q_z, q_w)`;
- pinhole intrinsic matrix `K_i`.

The intrinsic matrix is:

```math
K_i = \begin{bmatrix}
f_{x,i} & 0 & c_{x,i} \\
0 & f_{y,i} & c_{y,i} \\
0 & 0 & 1
\end{bmatrix}.
```

Per-frame intrinsics in `odometry.csv` take precedence over the fallback matrix in
`camera_matrix.csv`.

### 3.2 Structural validation

Frame identifiers, rather than an assumed video frame rate, synchronize the streams. The loader
requires identifiers to be contiguous from zero, timestamps to increase strictly, and the decoded
video and odometry counts to differ by no more than one frame. This matters because a video can be
labelled 60 fps while its exported timestamps indicate a different effective rate.

Structurally invalid data raises an error immediately. Structurally valid but weak data receives a
quality decision under `poc_engineering_v1`.

### 3.3 Motion and coverage metrics

For adjacent ARKit camera centers, translation is:

```math
\Delta c_i = \lVert c_{i+1} - c_i \rVert_2.
```

The algorithm reports total path length, axis-aligned trajectory span, p95 linear speed, and the
number of jumps larger than 15 cm. It normalizes adjacent quaternions and computes angular change
from their absolute dot product:

```math
\Delta \theta_i = 2\arccos\left(\left|\hat q_i \cdot \hat q_{i+1}\right|\right).
```

Dividing translation and angular change by `t_(i+1) - t_i` gives linear and angular speeds.

ARKit camera forward vectors are recovered by rotating the camera `-Z` direction into world space.
View-direction coverage is the maximum pairwise angle between these forward vectors. Orientation
coverage is not treated as a substitute for camera translation: rotating from one stationary
location does not provide the parallax required for reconstruction.

## 4. Video quality analysis and frame selection

### 4.1 Image-quality series

Every video frame is decoded. The frame is converted to grayscale, reduced to 35% linear size, and
evaluated inside a fixed central region covering 10–90% of image height and 15–85% of image width.
This region reduces the influence of background texture.

The following values are calculated for each frame:

- sharpness: variance of the Laplacian;
- median luminance;
- percentage of pixels at or below intensity 5;
- percentage of pixels at or above intensity 250.

The summary reports sharpness p10 and median, median black/white clipping, luminance p05 and p95,
and the p05–p95 temporal luminance range. These values are meaningful only together with this
specific preprocessing definition; changing the region or scale requires a new quality profile.

### 4.2 Temporally distributed selection

The capture duration is divided into `N` equal-duration windows, with `N = 120` by default. The
sharpest frame in every window is retained if its Laplacian variance is at least 35. This strategy:

- preserves angular coverage across the full capture;
- avoids selecting a cluster of nearly identical sharp frames;
- handles variable frame intervals correctly;
- reduces a long 1,000–2,000-frame video to a tractable reconstruction set.

At least 60 selected frames are required by the hard pipeline gate. The quality profile expects at
least 100 for a full `PASS`.

### 4.3 Pixel geometry

The reference exports are commonly stored sideways. Each selected image is rotated and optionally
resized. A homogeneous `source_to_image` transform is stored for every frame so RGB, masks, camera
intrinsics, and LiDAR lookup use one explicit coordinate convention.

For clockwise rotation of a source image with width `W` and height `H`:

```text
(u, v) -> (H - 1 - v, u)
```

The intrinsics are transformed consistently. Clockwise rotation produces:

```math
K'_i = \begin{bmatrix}
f_{y,i} & 0 & H - 1 - c_{y,i} \\
0 & f_{x,i} & c_{x,i} \\
0 & 0 & 1
\end{bmatrix},
```

followed by multiplication of the first two rows by the image resize factor. The result is written
to `frames.json` for all later stages.

## 5. Facial masking

MediaPipe Face Landmarker detects a face in each selected frame. The first 468 landmarks are
converted to pixels, their two-dimensional convex hull is filled, and the hull is dilated with an
elliptical kernel whose diameter is approximately 2.5% of the longer image dimension.

Two outputs are produced per successful view:

- a binary mask used by COLMAP feature extraction;
- a black-background reconstruction image used by dense geometry.

Views without a detected face are excluded. The hard gate requires at least 60 accepted images and
at least 70% detection success; the final quality profile is stricter and gives `PASS` at 95%.

This is intentionally a face-first mask, not a complete semantic head segmentation model. It can
crop parts of the ears, scalp, and hair boundary. That limitation is relevant to cosmetic rendering
and is not hidden by the algorithm.

## 6. Sparse visual reconstruction

### 6.1 Features and measured calibration

COLMAP extracts SIFT features from the background-suppressed images while also applying the binary
masks. Every image is initially assigned a separate `PINHOLE` camera. The pipeline then updates
COLMAP's database with the measured, transformed `f_x`, `f_y`, `c_x`, and `c_y` for each image.

Bundle adjustment is configured not to refine focal length, principal point, or extra camera
parameters. This prevents an unconstrained visual optimizer from silently replacing the measured
calibration with a calibration that merely improves reprojection error.

### 6.2 Matching and mapping

The default matcher is sequential with ten-frame overlap and loop detection disabled. Exhaustive
matching is available for diagnostics but costs more. COLMAP then estimates visual camera poses and
sparse 3D points through incremental structure from motion and bundle adjustment.

ARKit extrinsics are not injected as fixed COLMAP poses. COLMAP solves its own visual coordinate
system because rolling shutter, VIO drift, and RGB/pose imperfections make ARKit useful metric
evidence but not exact photogrammetric ground truth.

If mapping produces multiple connected models, the model with the most registered images is
selected. Dense reconstruction is blocked unless at least 40 images and 60% of input images are
registered. Structured counts are saved in `sfm.json`.

## 7. Metric scale estimation

COLMAP geometry initially has arbitrary scale. ARKit camera centers provide the primary conversion
to metres, and LiDAR depth supplies an independent consistency check.

### 7.1 ARKit trajectory alignment

For every image registered by COLMAP, the algorithm pairs:

- COLMAP camera center `x_i` in arbitrary reconstruction units;
- ARKit camera center `y_i` in metres, matched by explicit frame identifier.

It estimates a similarity transform:

```math
y_i \approx s R x_i + t,
```

where `s` is metres per COLMAP unit, `R` is a proper rotation, and `t` is translation. The closed-form
Umeyama solution centers both trajectories, computes their covariance, applies singular value
decomposition, prevents reflection, and derives `s`, `R`, and `t`.

The solution is placed inside deterministic RANSAC:

- 1,000 iterations;
- eight trajectory pairs per hypothesis when available;
- 25 mm inlier threshold;
- final refit over the best inlier set;
- at least 20 inliers and at least 50% of pairs required.

The output includes inlier ratio and median/p95 metric residuals. The authoritative scale is the
ARKit trajectory estimate even when LiDAR is present.

### 7.2 LiDAR cross-check

For each registered image, sparse COLMAP points with valid 2D observations are projected back to
the original source pixel convention. Samples are retained only when:

- the image observation lies inside the face mask;
- LiDAR confidence is at least 2;
- LiDAR depth is between 0.25 and 1.5 m;
- the view provides at least 15 usable correspondences.

For a sparse point with COLMAP camera-space depth `z_colmap` and measured depth `z_lidar`, one scale
sample is:

```math
s_j = \frac{z_{lidar,j}}{z_{colmap,j}}.
```

At least 15 usable images are required. The algorithm takes a global median, rejects samples beyond
the larger of `3.5 × MAD` and 1% of the median, and takes the median again. Pose and depth scales are
considered verified when they differ by no more than 2%.

LiDAR agreement is not external ground truth because both signals originate from the same phone.
It detects gross inconsistencies; it does not prove submillimetre facial accuracy.

## 8. Dense geometry

### 8.1 PatchMatch stereo

COLMAP first undistorts the masked reconstruction images and the selected sparse model into a dense
workspace. The generated PatchMatch configuration is bounded for T4 runtime:

- reference views are uniformly sampled down to at most 96;
- six source images are requested per reference by default;
- photometric consistency is the default;
- geometric consistency is optional and substantially slower;
- cache size is 30% of system RAM, clamped to 2–8 GB unless set explicitly.

CUDA-enabled COLMAP is required. PatchMatch estimates a depth map for every configured reference
view. Stereo fusion merges mutually supported depth samples into the permanent case artifact
`face_dense_fused.ply`.

That case-root path is passed directly to COLMAP as `stereo_fusion --output_path`; it is not a copy,
mesh resampling, or reconstructed substitute. It is deliberately outside the disposable
`colmap/dense/` workspace. Immediately after fusion—and before normal estimation or Poisson—the
pipeline records in `mvs.json`:

- its explicit role and generating COLMAP stage;
- relative case path and generation-time absolute path;
- point and non-finite-point counts;
- whether the persisted artifact contains normals and colors;
- raw bounding-box minimum, maximum, and extent;
- metric bounding-box extent using the already estimated scale;
- file size and SHA-256 digest.

The file is then opened as Poisson input but never overwritten. If normals are absent, Open3D
estimates them only on the in-memory point-cloud object and records that fact separately.

### 8.2 Surface construction

The fused point cloud must contain at least 10,000 points. Normals are estimated if missing, then
Open3D screened Poisson reconstruction creates the triangle surface at depth 9 by default.

Low-support Poisson vertices are removed using the fourth percentile of returned density values.
The mesh then removes degenerate triangles, duplicate triangles, and unreferenced vertices. Only the
largest triangle-connected component is retained. The result, `face_mesh_raw.ply`, is still in
COLMAP's arbitrary units and is retained for diagnostics and texture projection.

This stage reconstructs geometry from RGB photometric correspondence. LiDAR is not fused as facial
surface geometry because its spatial resolution is insufficient for subtle nasal details; it is
used only for the scale cross-check.

## 9. Texture and appearance

Geometry and appearance use different pixel preparation but the same cameras and surface:

```text
masked RGB views   -> PatchMatch -> authoritative geometry
original RGB views -> projection -> UV albedo on that geometry
```

Using black-background images for texture projection created dark halos and false colors near the
face boundary. The texture stage now creates a separate COLMAP undistortion workspace from the
original selected RGB images and the already selected sparse model. COLMAP `mesh_texturer` projects
those registered images onto `face_mesh_raw.ply`, applies view-dependent color correction, creates
UV coordinates, and writes `texture.png` plus a textured PLY.

The texture can duplicate vertices along UV seams, but it is not allowed to change triangle order
or any triangle-corner position. The GLB uses a non-metallic PBR material with:

```text
metallic factor  = 0.00
roughness factor = 0.72
base color       = registered RGB texture atlas
```

These are conservative rendering defaults, not a measured biophysical skin model. Normal,
roughness, and validated displacement maps are not reconstructed yet.

## 10. Authoritative metric geometry

The raw mesh is multiplied by `scale_mm_per_unit / 1000` and persisted in metres as
`face_geometry.ply`. Scaling is the only geometric operation performed when creating this asset;
vertex order and triangle topology remain unchanged.

The persisted float32-compatible positions and ordered uint32 triangles are hashed separately and
together with SHA-256:

```text
vertex_positions_sha256 = hash(float32 vertex positions)
topology_sha256          = hash(uint32 ordered triangles)
geometry_id              = hash(positions + topology)
```

Downstream landmarks, measurements, and visual assets must carry the same `geometry_id`.

During GLB export, every rendered triangle corner is compared with the corresponding authoritative
triangle corner. Export fails if triangle count changes or maximum position deviation exceeds
`1e-6 m`—one micrometre. Therefore:

- render face index `i` maps to authoritative face index `i`;
- a UV seam may create multiple render copies of one authoritative vertex;
- picking must resolve the triangle and barycentric coordinate, not the nearest visual vertex;
- visual displacement cannot affect measurements unless validated and promoted to a new
  authoritative geometry identity.

## 11. Multi-view landmark localization

### 11.1 Current provisional landmark mapping

The current proof of concept maps ten MediaPipe indices to anatomical labels:

| Anatomical label | MediaPipe index |
|---|---:|
| glabella | 9 |
| nasion | 168 |
| pronasale | 1 |
| subnasale | 2 |
| columella | 164 |
| labiale superius | 0 |
| left alare | 98 |
| right alare | 327 |
| left endocanthion | 133 |
| right endocanthion | 362 |

The names are provisional approximations. MediaPipe was not designed as a surgeon-approved
rhinoplasty annotation system.

### 11.2 Back-projected rays

MediaPipe detects each landmark in every registered original RGB view. For pixel
`p = (u, v, 1)^T`, the camera-space ray is:

```math
d_c = K^{-1}p.
```

Using COLMAP's refined camera rotation, the normalized world ray is:

```math
d_w = \frac{R^T d_c}{\lVert R^T d_c \rVert_2}.
```

The ray origin is the COLMAP camera center obtained from the inverse camera-from-world transform.

### 11.3 Robust ray triangulation

For camera centers `c_i` and unit ray directions `d_i`, the initial 3D estimate minimizes squared
orthogonal distance to all rays. It solves:

```math
\left[\sum_i (I-d_i d_i^T)\right]x
= \sum_i (I-d_i d_i^T)c_i.
```

Up to four robust iterations reject rays using perpendicular residuals and a threshold based on the
median, MAD, twice the median, and a small numerical floor. At least six views must remain. The
median retained ray residual is recorded in millimetres.

### 11.4 Visible surface consensus

The initial triangulation is not silently moved several millimetres to the nearest arbitrary
surface. Instead, every view ray is intersected with the authoritative metric mesh. A hit is kept
only if it is finite and lies within 25 mm of the initial triangulated estimate; this rejects an
accidental hit on unrelated or back-side geometry.

Surface hits are then fused robustly for up to five iterations. Starting from their coordinate-wise
median, the algorithm calculates Euclidean distances and retains hits within:

```math
\tau = \min\left(8\text{ mm},
                  \max\left(2.5\text{ mm},
                            \operatorname{median}(r) + 3 \times 1.4826 \times MAD(r)
                       \right)
            \right).
```

At least six surface observations must remain. Their mean is projected to the closest authoritative
triangle. The result records:

- total 2D observations and triangulation inliers;
- median triangulation ray residual;
- number of visible ray–surface hits;
- number of surface-consensus inliers;
- median and p95 surface dispersion;
- final surface projection distance.

### 11.5 Barycentric binding

For authoritative triangle vertices `v_0`, `v_1`, and `v_2`, a landmark is stored as triangle index
plus barycentric weights:

```math
x = b_0v_0 + b_1v_1 + b_2v_2,
\qquad b_0+b_1+b_2=1.
```

This makes the landmark stable under future topology-preserving deformation. If a surgeon clicks
rendered face `i`, the same barycentric position can be evaluated directly on authoritative face
`i`.

## 12. Provisional measurements

All landmark coordinates in `landmarks.json` are millimetres. The current implementation computes
the following six values.

For three points `a`, `b`, and `c`, the generic 3D angle at `b` is:

```math
\theta(a,b,c) = \arccos\left(
\frac{(a-b)\cdot(c-b)}{\lVert a-b\rVert_2\lVert c-b\rVert_2}
\right).
```

| Output | Current formula |
|---|---|
| Nasofrontal angle | `angle(glabella, nasion, pronasale)` |
| Nasolabial angle | `angle(columella, subnasale, labiale_superius)` |
| Nose length | Euclidean distance from nasion to pronasale |
| Nose width | Euclidean distance from left alare to right alare |
| Goode ratio | distance from pronasale to alar midpoint, divided by nose length |
| Midline deviation | absolute projection of tip-to-eye-midpoint vector onto the inter-endocanthion axis |

For the last value, if:

```math
e = \frac{e_R-e_L}{\lVert e_R-e_L\rVert_2},
\qquad m = \frac{e_L+e_R}{2},
```

then:

```math
d_{midline} = |(p_{tip}-m)\cdot e|.
```

These are direct three-dimensional formulas, not final surgeon-approved clinical definitions. In
particular, the angle and projection definitions do not yet construct a validated facial sagittal
plane or fit anatomical tangent lines. The output is deliberately labelled
`provisional_surface_rhinoplasty_measurements_v1`, and its quality section cannot receive clinical
approval until surgeons define the landmarks, reference planes, and interpretation in writing.

## 13. Experimental dorsal hump reduction simulation

This optional operation is downstream of reconstruction, geometry identity, landmarks, and
measurements. It never participates in the authoritative pipeline and never overwrites its inputs.
The only parameter is `reduction_mm` in the closed interval 0.0–5.0 mm.

The simulator builds an orthogonal patient-specific frame from nasion, pronasale, subnasale, and
the two alar landmarks. The nasion-to-subnasale direction defines the longitudinal axis. The
pronasale component orthogonal to that axis defines anterior projection, and their cross product is
aligned with the observed left-to-right alar direction. No facial symmetry is imposed.

Within a narrow midline strip from nasion to an inferred supratip boundary, the existing dorsal
profile is the smoothed 90th-percentile anterior envelope. A straight chord through robust proximal
and distal anchors is used only to locate the maximum outward residual in the upper/mid-dorsal search
band (normalized 0.16–0.62). It is not the target profile and it never caps the requested operation.

`reduction_mm` is the desired posterior displacement amplitude at that detected hump apex. For every
accepted 0.0–5.0 mm request, the current solver applies the requested amplitude exactly; consequently
`applied_reduction_mm == requested_reduction_mm` and `cap_reason` is null. Any future anatomical cap
must change those fields explicitly rather than silently returning a smaller operation.

The sagittal target is the source profile minus a broad asymmetric Gaussian displacement kernel. Its
maximum is exactly `reduction_mm` at the hump apex. The upper side uses normalized sigma 0.34 and the
lower side 0.16. A C2 radix fade ends at 0.16, while a C2 distal fade begins at 0.64 and reaches zero
at the supratip anchor 0.88. This produces one smooth, continuous target that moves the upper and mid
dorsum coherently and returns to anchored anatomy without reference-line clipping, local steps, or
pointwise target truncation.

For each vertex of the connected dorsal vault, the simulator interpolates `delta(s)` and multiplies
it by a smooth transverse vault weight. The ridge receives full correction, adjacent slopes retain
more than 90%, and sidewall correction progressively approaches zero at the anatomical perimeter.
A mild medial component peaks on the slopes/sidewalls and is capped at 0.6 mm. Both posterior and
medial components are constraints in one vector-valued biharmonic solve; there is no center-strip
permission mask or second independent sidewall pass. The fixed perimeter includes the nasion,
supratip, cheeks, tip, alae, nostrils, columella, and upper lip. The solve never uses vertex normals
or imposes left/right symmetry. One unconstrained interior transition ring lets the biharmonic field
blend into the fixed perimeter without an abrupt mesh seam. A 0.0 mm request copies the authoritative
PLY exactly.

Outputs live under `simulations/dorsal_hump/`. `simulation.json` records the source geometry ID,
simulation geometry ID, requested reduction, maximum persisted displacement, affected vertex
count, median affected displacement, vertices moved over 0.1 mm, ROI definition and axes, source
hashes, output hashes, and PLY/GLB persistence checks. An ROI-only colored PLY exposes the selected
nasal surface. Profile JSON/SVG outputs contain original, target, and final curves plus
displacement and target error at radix/nasion, upper dorsum, hump apex, mid/lower dorsum, supratip,
and pronasale. Four transverse sections report original/target/final curves, bridge widths, central
heights, and left/right sidewall coordinates and medial displacement. Clay and textured front/profile
comparisons, normal visualization, separate front/profile displacement heatmaps, and ridge/slope/
sidewall coverage ratios are also persisted. The
notebook downloads a geometry-hash-qualified GLB so external viewers cannot reuse a stale file with
the same slider label. This is a visual aesthetic simulation, not a surgical-outcome prediction.
The manifest and profile JSON record requested/applied reduction, cap reason, apex/upper/mid/lower/
radix/supratip displacement, maximum vertex displacement, and moved-vertex count.
GLB transverse verification compares the exported maximum directly with the solved medial field to
within a 0.002 mm persistence tolerance; it does not require a fixed 0.1 mm movement when the applied
profile correction intentionally produces a smaller transverse request.

## 14. Quality decision algorithm

The system does not average unrelated checks into one reassuring score. Every section contains
individual `PASS`, `WARN`, or `FAIL` checks, and the section and case inherit the worst status:

```text
PASS < WARN < FAIL
```

The final report evaluates:

- capture synchronization, motion, coverage, sharpness, clipping, and illumination;
- face-mask success;
- sparse image registration and sparse point count;
- ARKit/COLMAP scale residuals and LiDAR agreement;
- mesh finiteness, degeneracy, connected components, winding, boundaries, and edge sampling;
- geometry identity and render-surface deviation;
- texture resolution and face-sampled black/white pixels;
- landmark ray residual, surface hit count, dispersion, and projection distance;
- presence and provisional status of the six measurements;
- recorded pipeline runtime and stage completion.

Texture clipping is sampled at every rendered face's UV centroid rather than across the full atlas,
because unused atlas background should not be mistaken for facial texture.

A failed capture blocks expensive reconstruction. A later-stage failure preserves diagnostic
outputs but leaves `clinical_use_authorized` false. Exact versioned thresholds are documented in
[QUALITY_ASSURANCE.md](QUALITY_ASSURANCE.md).

## 15. Repeatability algorithm

Repeatability analysis requires at least two independently captured reconstructions of the same
subject.

### 14.1 Measurement repeatability

For each measurement, the report calculates:

- all per-case values;
- arithmetic mean;
- sample standard deviation with `n - 1` denominator;
- coefficient of variation when the mean is nonzero;
- maximum pairwise difference, equal to `max(value) - min(value)`.

The maximum pairwise difference drives the candidate PASS/WARN/FAIL decision. Repeatability does
not determine which scan is correct.

### 14.2 Rigid surface repeatability

Each case loads its authoritative mesh and landmarks. Pairwise comparison performs:

1. rigid Kabsch initialization from common landmarks, with reflection prevention and no scale;
2. selection of central facial vertices within 75 mm of the landmark centroid;
3. 1.5 mm voxel downsampling;
4. point-to-point ICP with a 6 mm correspondence threshold and at most 100 iterations;
5. definition of the nasal region as vertices within 22 mm of nasion, pronasale, subnasale, or
   either alare;
6. exact point-to-triangle distance from the transformed source nasal vertices to the target mesh;
7. the reverse target-to-transformed-source distance;
8. concatenation into symmetric median, p95, and p99 nasal-surface distances.

Scale is never optimized during the reported surface alignment because doing so would hide metric
scale drift. A landmark-based similarity scale is calculated only as a diagnostic and is explicitly
reported as not applied.

Surface repeatability is still not absolute accuracy. Accuracy needs a trusted physical surface or
manual reference, not merely agreement between scans produced by the same algorithm.

## 16. Reproducibility, state, and logging

The capture fingerprint hashes metadata and file sizes for the RGB/pose/calibration streams and the
number and range of depth/confidence files. Each stage signature hashes:

```text
stage name
+ capture fingerprint
+ stage parameters
+ upstream stage signature
+ software version
```

`--resume` skips a stage only when the signature matches and every declared output still exists. A
stale record is rejected instead of combining results produced from different inputs or parameters.
`case.json` is written atomically and records stage start, completion or failure, metadata, and
elapsed time.

The MVS stage declares three required outputs: the direct fused point cloud, `mvs.json`, and the
Poisson mesh. Therefore a cached MVS stage cannot be considered complete when the pre-Poisson point
cloud is missing, even if the final mesh still exists. The manifest also embeds the fusion
diagnostics returned by the stage.

Long-running commands emit structured stage logs, periodic heartbeats, and progress probes. The
application log and complete raw COLMAP log are retained separately.

## 17. Output contract

The principal outputs are:

| File | Role |
|---|---|
| `case.json` | Reproducible stage manifest, signatures, status, and timings |
| `capture_quality.json` | Input acceptance decision after selected-frame analysis |
| `frames.json` | Frame IDs, timestamps, quality, transforms, intrinsics, and image sizes |
| `sfm.json` | Sparse registration metrics |
| `scale.json` | ARKit scale, pose residuals, and optional LiDAR comparison |
| `face_dense_fused.ply` | Exact persistent COLMAP stereo-fusion point cloud before Poisson |
| `mvs.json` | Pre-Poisson fusion role, path, point count, normals, bounds, size, and hash |
| `face_mesh_raw.ply` | Diagnostic unscaled Poisson mesh |
| `face_geometry.ply` | Authoritative metric mesh in metres |
| `geometry.json` | Geometry identity and visual correspondence policy |
| `texture/mesh.ply` | UV-mapped mesh in raw reconstruction units |
| `texture/texture.png` | Color-corrected source-image albedo atlas |
| `face_model.glb` | Registered PBR visualization of the authoritative surface |
| `landmarks.json` | Metric landmarks, confidence metrics, and barycentric bindings |
| `measurements.json` | Six provisional measurements sharing the geometry identity |
| `quality_report.json` | Machine-readable final acceptance report |
| `quality_report.html` | Human-readable final acceptance report |
| `simulations/dorsal_hump/reduction_Xmm.ply` | Separate metric simulated surface |
| `simulations/dorsal_hump/reduction_Xmm.glb` | Separate simulated visual asset |
| `simulations/dorsal_hump/reduction_Xmm_<hash>.glb` | Cache-safe copy for viewing/downloading |
| `simulations/dorsal_hump/reduction_Xmm_affected_roi.ply` | Colored nasal ROI diagnostic |
| `simulations/dorsal_hump/reduction_Xmm_moved_vertices.ply` | Point cloud containing only vertices actually moved |
| `simulations/dorsal_hump/reduction_Xmm_profile.svg` | Source/simulated sagittal profile overlay |
| `simulations/dorsal_hump/reduction_Xmm_profile.json` | Numerical source/target/simulated curves and apex |
| `simulations/dorsal_hump/reduction_Xmm_cross_sections.svg` | Upper/hump/mid/supratip transverse overlays |
| `simulations/dorsal_hump/reduction_Xmm_cross_sections.json` | Bridge width, height, and sidewall coordinate diagnostics |
| `simulations/dorsal_hump/reduction_Xmm_{front,profile}_{before,after}.png` | Orthographic diagnostic renders |
| `simulations/dorsal_hump/reduction_Xmm_{clay,profile_clay,normals}_{before,after}.png` | Geometry and normal comparisons |
| `simulations/dorsal_hump/reduction_Xmm_moved_vertices_heatmap.png` | Front/profile displacement heatmap |
| `simulations/dorsal_hump/reduction_Xmm_affected_roi.png` | Front/profile ROI-highlight render |
| `simulations/dorsal_hump/simulation.json` | Simulation provenance and displacement report |

## 18. What is not currently implemented

The following should not be inferred from the current algorithm:

- surgeon-approved landmark or measurement definitions;
- a validated facial midline or sagittal reference plane;
- automatic semantic segmentation of ears, scalp, hair, and neck;
- pore-level metric geometry or validated displacement reconstruction;
- normal or measured roughness maps;
- cartilage, bone, or internal nasal anatomy;
- predictive surgery, tissue biomechanics, or simulation operations other than dorsal hump reduction;
- absolute accuracy against traceable facial ground truth;
- clinical safety, efficacy, or regulatory validation.

Those are explicit later milestones. They must extend the authoritative-geometry architecture
rather than introducing a second visual face that can diverge from the measured surface.
