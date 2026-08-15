# Rhino PoC

Metric facial surface reconstruction and six provisional rhinoplasty measurements from an
iPhone 14 Pro Max Stray Scanner capture.

> **Research software:** this repository is not a medical device and its measurements are not
> validated for clinical use.

## Current objective

The first milestone is deliberately narrow:

- input: one complete Stray Scanner export;
- output: a clean, metric skin-surface mesh;
- measurements: nasofrontal angle, nasolabial angle, Goode ratio, nasal length, nasal width,
  and tip midline deviation;
- runtime target: less than one hour on a Google Colab T4 GPU.

FLAME fitting, subcutaneous anatomy, surgical morphing, Gaussian splatting, mobile application
development, and clinical validation are outside this milestone.

## Why Stray Scanner is required

A normal Camera-app MOV contains RGB pixels but no exported ARKit trajectory or LiDAR depth.
Monocular photogrammetry determines shape only up to an arbitrary scale. A complete Stray export
provides:

```text
capture/
├── rgb.mp4
├── odometry.csv
├── camera_matrix.csv
├── depth/
└── confidence/
```

The pipeline uses explicit exported frame IDs and timestamps. It never assumes that the nominal
MP4 frame rate is the true capture rate.

## Recommended execution: Google Colab

Open [`colab_reconstruction.ipynb`](colab_reconstruction.ipynb) in Colab and select a **T4 GPU**.
The notebook is linear and has three phases:

1. install and verify CUDA-enabled COLMAP;
2. download the official MediaPipe model, run the versioned capture preflight, and build a metric
   sparse checkpoint;
3. run dense reconstruction, registered texture mapping, metric export, landmarks, measurements,
   and the final acceptance report.

The notebook runs compute-intensive work on `/content`, not through Google Drive's FUSE layer.
It copies checkpoints and final artifacts back to Drive.

Default Colab settings are intentionally bounded:

| Setting | Default | Rationale |
|---|---:|---|
| Selected frames | 120 | Removes temporal redundancy from the 1,295-frame reference capture. |
| Long image dimension | 1,400 px | Preserves facial detail while reducing PatchMatch cost. |
| MVS references | 96 | Uniform coverage without processing every selected view densely. |
| Source images/reference | 6 | Appropriate initial budget for a small, highly overlapping subject. |
| Geometric consistency | Off | Photometric-first benchmark for the sub-hour target. |

These settings are a benchmark configuration, not a validated optimum. Quality and runtime must be
recorded before increasing them.

## Command-line use

Install Python 3.10 or newer, the package, and a COLMAP binary:

```bash
uv venv --python 3.11
uv pip install -e '.[dev]'
```

Inspect a capture without reconstructing it. A failed input returns exit code 2:

```bash
poc inspect /path/to/stray_capture --output input_quality.json --json
```

Download the official MediaPipe Tasks model once:

```bash
poc download-models --output-dir models
```

Run through sparse reconstruction locally on macOS:

```bash
poc reconstruct /path/to/stray_capture \
  --output case_001 \
  --face-landmarker-model models/face_landmarker.task \
  --no-sift-gpu \
  --until sfm
```

Run the complete pipeline on a CUDA machine:

```bash
poc reconstruct /path/to/stray_capture \
  --output case_001 \
  --face-landmarker-model models/face_landmarker.task \
  --frame-count 120 \
  --max-dimension 1400 \
  --mvs-references 96 \
  --mvs-source-images 6
```

Use `--resume` only with unchanged inputs and parameters. `case.json` stores the capture
fingerprint, software version, stage parameters, upstream signatures, status, and timing. A stale
stage is rejected instead of being silently reused.

## Live progress and logs

Every stage prints timestamped progress to the notebook or terminal. Long subprocesses emit a
heartbeat every 15 seconds even when COLMAP itself is silent. PatchMatch progress is read from
completed depth maps, which is more stable across COLMAP releases than parsing its console format.

Files:

- `run.log`: structured application log;
- `colmap/colmap.log`: complete raw COLMAP output and commands;
- `case.json`: stage state, parameters, signatures, and elapsed times.
- `capture_quality.json`: pre-reconstruction capture acceptance report;
- `quality_report.json` and `quality_report.html`: final stage-by-stage acceptance report.

Use `--verbose` to mirror raw COLMAP lines to the terminal.

## Output layout

```text
case_001/
├── case.json
├── run.log
├── images/
├── masks/
├── reconstruction_images/
├── colmap/
├── face_dense_fused.ply
├── mvs.json
├── face_mesh_raw.ply
├── capture_quality.json
├── sfm.json
├── scale.json
├── face_geometry.ply
├── geometry.json
├── texture/
│   ├── mesh.ply
│   └── texture.png
├── face_model.glb
├── landmarks.json
├── measurements.json
├── quality_report.json
└── quality_report.html
```

`face_geometry.ply` is the authoritative surface in metres. `geometry.json` records its identity,
topology, visual correspondence, and displacement policy. The GLB contains the same registered
surface with a source-image UV texture atlas and non-metallic skin material; UV seams may duplicate render vertices, but render
triangle indices remain one-to-one with authoritative triangles. Landmark and measurement JSON
files use millimetres and carry the same geometry identity. See [ARCHITECTURE.md](ARCHITECTURE.md).

Geometry reconstruction uses masked images. Texture mapping deliberately rebuilds its workspace
from the original registered RGB frames, preventing black reconstruction masks from contaminating
skin color and facial boundaries.

`face_dense_fused.ply` is the exact output path passed to COLMAP `stereo_fusion`. It is the raw
dense point cloud after PatchMatch and fusion but before normal estimation, Poisson reconstruction,
or mesh cleanup. It is stored outside the disposable COLMAP workspace and is a required MVS-stage
output. `mvs.json` records its role, point count, original normals/colors, bounding box, file size,
SHA-256 digest, and generation stage.

## Capture protocol

- Seat the subject with a supported, still head and neutral closed-mouth expression.
- Ask the subject to look at a fixed mark rather than following the phone with their eyes.
- Use constant diffuse lighting and a static environment.
- Keep the main rear camera approximately 60–70 cm from the face.
- Record one continuous Stray session with two passes:
  - eye-level ear-to-ear arc;
  - lower arc tilted upward to expose the nasal base.
- Do not stop recording between passes; one ARKit session provides one metric coordinate system.

The complete filming instructions and numeric PASS/WARN/FAIL thresholds are in
[QUALITY_ASSURANCE.md](QUALITY_ASSURANCE.md).

The reference capture is stored sideways without rotation metadata. The default
`--rotation clockwise` normalizes it to portrait and transforms camera intrinsics consistently.

## Measurement status

Landmarks are detected with MediaPipe Tasks' explicit CPU delegate in multiple registered views.
Triangulated rays initialize a robust consensus of visible ray–surface intersections. Every landmark stores its
triangle index and barycentric coordinates so it remains registered during clicking and future
deformation. Measurement definitions are centralized and explicitly labeled provisional. They are
suitable for engineering evaluation only and must be reviewed by a surgeon before any clinical
study.

For same-subject repeats, `poc repeatability-report` computes measurement variation and rigid-only
nasal surface distances. Repeatability is not accuracy: no accuracy claim is currently possible
because the project has no manual reference measurements or traceable facial ground truth.

## Development

```bash
uv run pytest
uv run ruff check .
```

See [ALGORITHM.md](ALGORITHM.md) for the implemented algorithm, [QUALITY_ASSURANCE.md](QUALITY_ASSURANCE.md)
for acceptance criteria, and [PLAN.md](PLAN.md) for milestones and known limitations.
