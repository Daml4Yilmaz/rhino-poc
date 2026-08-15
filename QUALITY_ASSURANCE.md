# Quality assurance and capture standard

This project uses stage-specific acceptance gates. It does not combine unrelated metrics into one
opaque score: a critical failure remains visible even when other stages are strong. Every check is
reported as `PASS`, `WARN`, or `FAIL` under a versioned profile.

`poc_engineering_v1` is an engineering profile for the proof of concept. Its limits are not a
medical-device specification and do not establish clinical accuracy.

## Standard capture protocol

Use the iPhone 14 Pro Max rear 1× camera through Stray Scanner. Use one uninterrupted session so
RGB, ARKit poses, intrinsics, and LiDAR remain in one coordinate system.

1. Seat the subject with the back and head supported. Hair must be held away from the forehead,
   temples, and ears. Remove glasses and reflective jewelry.
2. Keep a neutral, relaxed, closed-mouth expression. The subject looks at one fixed mark and does
   not follow the phone with the eyes. Avoid talking, swallowing, blinking during close nasal
   views, or changing expression.
3. Use broad, diffuse, constant lighting from both front sides. Avoid direct sunlight, moving
   shadows, specular hotspots, mixed color temperatures, and automatic brightness changes.
4. Keep the phone approximately 60–70 cm from the skin. Use the same lens throughout; do not zoom
   or allow automatic macro/lens switching.
5. Record for 25–45 seconds and move slowly. First make an eye-level arc from near the right ear,
   through frontal, to near the left ear. Then make a lower arc tilted upward enough to see the
   columella, alar rims, and nostril openings. Neighboring views should overlap by at least 70%.
6. Aim for at least 140° of view-direction coverage and at least 1.2 m of accumulated camera path.
   Do not make a full orbit behind the head; the clinical target is the face and visible ears.
7. Do not insert isolated close-ups, still photographs, pauses, or a second recording into the
   session. A separate high-resolution texture capture can be added only after the pipeline has an
   explicit registered-photo ingestion stage.

Lock focus, exposure, white balance, lens selection, and torch state when the capture application
supports those controls. Stray Scanner may not expose every lock. The preflight therefore measures
focal drift, clipping, sharpness, and temporal luminance rather than assuming the controls stayed
fixed.

Run preflight before reconstruction:

```bash
poc inspect /path/to/stray_capture --output input_quality.json --json
```

A `FAIL` returns exit code 2 and must be recaptured. A `WARN` can proceed for engineering analysis,
but the warning remains in the case record.

## Input gates: `poc_engineering_v1`

| Check | PASS | WARN | FAIL |
|---|---:|---:|---:|
| Synchronized frames | ≥750 | 540–749 | <540 |
| Duration | 25–45 s | 18–24.99 or 45.01–60 s | outside 18–60 s |
| Effective rate | ≥29 fps | 24–28.99 fps | <24 fps |
| Long RGB dimension | ≥1,920 px | 1,280–1,919 px | <1,280 px |
| Camera path | ≥1.2 m | 0.75–1.19 m | <0.75 m |
| Trajectory span | ≥0.4 m | 0.25–0.39 m | <0.25 m |
| View-direction coverage | ≥140° | 120–139.99° | <120° |
| Tracking jumps over 15 cm | 0 | 1 | >1 |
| Focal-length drift | ≤1% | >1–2% | >2% |
| Linear speed p95 | ≤0.15 m/s | >0.15–0.30 m/s | >0.30 m/s |
| Angular speed p95 | ≤30°/s | >30–60°/s | >60°/s |
| Selected-frame sharpness p10 | ≥80 | 40–79.99 | <40 |
| Selected-frame median sharpness | ≥150 | 80–149.99 | <80 |
| Clipped black or white pixels | ≤1% | >1–5% | >5% |
| Temporal luminance p05–p95 | ≤40 levels | >40–70 | >70 |
| LiDAR frame coverage, when present | ≥95% | 80–94.99% | <80% |

Laplacian sharpness is resolution- and implementation-dependent. The profile fixes the central
region, downsampling, and thresholds together; changing that implementation requires a new profile
version.

## Output and stage gates

The final `quality` stage writes `quality_report.json` and `quality_report.html`.

| Stage | Key acceptance conditions |
|---|---|
| Masking | PASS at ≥95% accepted views; WARN at 85–94.99%; FAIL below 85%. |
| Sparse reconstruction | PASS at ≥80 images, ≥85% registration, and ≥10,000 sparse points; hard minimums are 40 images and 60%. |
| Scale | ARKit/COLMAP pose inlier ratio ≥0.8, median residual ≤10 mm, p95 ≤20 mm; LiDAR agreement must not explicitly fail. |
| Authoritative mesh | Finite vertices, consistent winding, largest component ≥99.9%, median edge length 0.2–1.0 mm. Open crop boundaries are reported as `WARN`, not hidden. |
| Geometry/visual registration | Geometry IDs match, the rendered surface stays within 1 micrometre of the authoritative surface, the atlas is at least 4K for PASS, and face-sampled near-black texture remains ≤2% for PASS (FAIL >8%). |
| Landmarks | Minimum inlier ratio ≥0.7, ray residual ≤1.5 mm, surface snap ≤1 mm, and new surface-consensus dispersion p95 ≤2 mm. |
| Measurements | Six values must exist, but the section remains `WARN` until surgeons approve definitions and reference planes. |
| Runtime | PASS at ≤3,600 s; WARN through 5,400 s; FAIL above 5,400 s. |

Mesh integrity and repeatability are not absolute accuracy. Absolute accuracy requires a traceable
physical reference and manual anatomical reference measurements.

## Repeatability

Use independently captured cases of the same subject:

```bash
poc repeatability-report \
  /path/to/case_001 /path/to/case_002 /path/to/case_003 \
  --subject-id pseudonymous_subject_001 \
  --output repeatability_report.json \
  --html-output repeatability_report.html
```

The report contains:

- mean, sample standard deviation, coefficient of variation, and maximum pairwise difference for
  every measurement;
- rigid-only central-face alignment with no scale optimization;
- symmetric point-to-surface median, p95, and p99 distances in an automatically defined nasal
  region;
- a diagnostic landmark similarity scale that is reported but never applied to hide scale drift.

Candidate repeatability limits are 3°/5° for angle PASS/FAIL boundaries, 1/2 mm for distances,
0.03/0.05 for the Goode ratio, and 1/2 mm for nasal-surface p95. These must be approved before a
formal study. With one subject, population reliability and ICC are not estimable.

## Required validation sequence

1. Freeze the capture protocol and software/profile versions.
2. Have surgeons approve written landmark, reference-plane, and measurement definitions.
3. Build a click-based manual landmark reference on the authoritative mesh with at least two
   blinded observers.
4. Use a calibrated rigid object to test metric scale independently of a face.
5. Predefine repeatability and accuracy endpoints, sample sizes, exclusion rules, and acceptable
   limits before collecting the validation set.
6. Report repeatability, inter-observer variability, and agreement against reference separately.

Until those steps are complete, `clinical_use_authorized` remains `false` in every quality report.
