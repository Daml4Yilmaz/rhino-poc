# rhino-poc — Execution Plan (v2, Aug 2026)

Supersedes the sequencing in `README.md`. Decisions from `HANDOVER_claude_code.md`
are kept; what changes here is **where each stage runs** (local vs. Colab GPU)
and the order of work.

---

## 0. What Kratos actually does — and what it means for us

Findings from their site, App Store listing and marketing copy (Aug 2026):

| Claim | Source | What it implies |
|---|---|---|
| "move around the person as if you were shooting a video" | site / press | **Videogrammetry** — a monocular RGB video orbit. Same input as ours. |
| "using only your smartphone", "standard mobile cameras", "no expensive in-clinic hardware" | site | The baseline path does **not** depend on LiDAR or TrueDepth. |
| App requires iOS 16+, **no Pro / LiDAR / TrueDepth restriction** listed | App Store | Confirms the above — it runs on non-Pro iPhones. |
| "Advanced Scanning" on **iPhone Pro** adds a "4th Stage" nose capture in "ultra-high resolution" | App Store | Pro-only extra pass. Likely LiDAR-assisted **scale/pose priors**, not the geometry itself. |
| "cloud-based algorithm trained on CNN data" | site | Server-side reconstruction. Processing is not on-device. |
| Resections at Radix / Rhinion / Supratip / Infratip, "subcutaneous anatomy (under the skin)" | site | A **statistical model fitted to the scan** (soft-tissue → bone/cartilage prior), not measured anatomy. Exactly the FLAME-fit role in our pipeline. |
| Validation vs. "legacy hardware scanners" at Ege, Cerrahpaşa, Cairo | site | Their accuracy bar is a comparison study, same class of evidence as our caliper protocol. |

**Conclusion: the user's hypothesis holds.** Photogrammetry/videogrammetry on the
rear camera, cloud processing, a template model for under-the-skin inference.
Nothing here requires a capture modality we don't have.

**The one thing they have and we must engineer around: metric scale.** They ship a
native app, so they can read ARKit/LiDAR pose and depth at capture time and get
mm for free. We ingest an uploaded video, so scale must come from a printed
marker (PoC) — and from ARKit pose recording once we go native (post-G1). This is
the single largest accuracy risk in the PoC and is why WP2 exists.

**Not visible anywhere in their public material:** the reconstruction backend, the
mesh resolution, the accuracy number in mm. Treat their marketing precision
("millimetric") as unverified.

---

## 1. Architecture change: local-first, GPU only when unavoidable

The Colab free tier was exhausted mid-run once already, wiping `/content`. So we
stop treating Colab as the machine and start treating it as an **accelerator we
visit for two stages only**.

Audit of what genuinely needs CUDA:

| Stage | Needs CUDA? | Where it runs |
|---|---|---|
| (a) frames — ffmpeg + Laplacian | no | **local (M4 Pro)** |
| (b) SfM — COLMAP feature/match/mapper | no (SIFT on 12 CPU cores is fine at 200 frames) | **local** |
| (c1) MVS — `patch_match_stereo` | **yes, hard CUDA requirement** | **Colab T4** |
| (c2) Gaussian splatting — gsplat | **yes** | **Colab T4** |
| (d) masking — MediaPipe face parsing | no | **local** |
| (e) scale — ChArUco detect + triangulate | no | **local** |
| (f) FLAME fit | no, *if* written without PyTorch3D (see below) | **local (MPS/CPU)** |
| (g) measure | no | **local** |
| (h) export GLB | no | **local** |
| report/compare | no | **local** |

Two consequences worth stating plainly:

1. **Drop PyTorch3D.** Its value here is chamfer distance + a mesh container, both
   of which are ~40 lines of plain PyTorch plus an Open3D/scipy KD-tree. Keeping
   PyTorch3D forces the riskiest, most iteration-heavy stage (FLAME fit) onto a
   CUDA box, where every debug cycle costs Colab quota. Written in plain PyTorch it
   runs on the M4 Pro via MPS in seconds per iteration, and stays portable to a
   rented GPU later without change. This also deletes the "install a wheel matching
   the CUDA version" failure mode from the handover.
2. **Colab's job shrinks to one call:** `mvs` (and later `gsplat`). A session that
   only runs MVS burns a fraction of the quota that a full end-to-end run does.

Hardware on hand: Apple M4 Pro, 12 cores, 24 GB unified memory. Adequate for every
local stage; the 24 GB is shared with the GPU so keep Poisson depth ≤10 and
downsample the scan before the FLAME fit.

---

## 2. Handoff protocol (local ⇄ Colab)

The **case directory is the unit of exchange**. Google Drive is the transport (the
user already has `MyDrive/rhino-poc-data/`). Nothing binary goes into git.

```
vaka_001/
  case.json          <- NEW: manifest; which stages ran, when, with what params
  frames/            <- local output, needed by Colab       (~200 jpg, ~60 MB)
  colmap/sparse/0/   <- local output, needed by Colab       (~5 MB)
  colmap/dense/      <- Colab output (fused.ply)            (large, do not sync back whole)
  mesh_raw.ply       <- Colab output, needed by local       (~40 MB)
  scale.json         <- local
  landmarks.json     <- local
  model.glb          <- local
  measurements.json  <- local
```

Round trip:

```bash
# LOCAL
poc run frames vaka_001.mp4 --out vaka_001
poc run sfm    --out vaka_001
poc pack       vaka_001 --for gpu      # -> vaka_001_gpu.zip  (frames + sparse only)
# upload zip to Drive

# COLAB (GPU runtime; setup cell + this)
poc unpack /content/drive/.../vaka_001_gpu.zip --out /content/vaka_001
poc run mvs --out /content/vaka_001
poc pack    /content/vaka_001 --for local   # -> mesh_raw.ply + logs only
# writes back to Drive

# LOCAL
poc unpack vaka_001_local.zip --out vaka_001
poc run scale --out vaka_001 --marker-mm 49.5
poc run flame --out vaka_001
poc run measure --out vaka_001
poc run export  --out vaka_001
```

This requires two code changes: a per-stage entry point (`poc run <stage>`) and
`case.json` so a stage can verify its inputs exist and were produced with the
parameters it expects. `poc process` stays as a convenience wrapper for a
single-machine run.

---

## 3. Work packages

Ordered by risk-adjusted value. WP1–WP3 are the critical path to a first number.

### WP0 — Local environment (half a day, local)
- `brew install ffmpeg colmap` (COLMAP builds without CUDA on macOS — expected and fine; MVS is Colab's job).
- Python 3.11 via `uv` or `brew` — the system 3.9 is too old for the stack.
- `uv venv && uv pip install -e .` plus `mediapipe`, `torch`, `pymeshlab`, `scipy`.
- **Acceptance:** `poc --help` runs; `colmap -h` runs; `python -c "import cv2,open3d,trimesh,mediapipe,torch"` clean; `torch.backends.mps.is_available()` is True.
- Open a FLAME account now (`flame.is.tue.mpg.de`) — approval is not instant and WP4 blocks on it.

### WP1 — Stage runner + manifest + pack/unpack (1 day, local)
- Refactor `cli.py`: `poc run <stage>` with explicit inputs, `poc process` calling it in sequence.
- `case.json` manifest: stage name, timestamp, parameters, output hashes.
- `poc pack --for gpu|local`, `poc unpack`.
- Decouple `--no-gpu`: currently one flag gates both SfM SIFT and implies MVS is dead. Split into `--sift-gpu/--no-sift-gpu`; MVS just fails loudly with a clear message if CUDA is absent.
- **Acceptance:** `vaka_001` reruns end-to-end from an existing case folder with zero recomputation; a packed gpu zip is < 100 MB.

### WP2 — Scale: ArUco → ChArUco (1–2 days, local) — *highest-risk-per-hour*
The handover's diagnosis stands: a paper marker bends on a curved forehead, default
corner refinement is off, and 4 corners are too few. Fix in this order:
1. **Rigid mount.** Marker on rigid card, card on a headband. Not on skin. This alone may be most of the error.
2. **`cornerRefinementMethod = CORNER_REFINE_SUBPIX`** — one line, currently missing in `scale.py` (it passes a default `DetectorParameters()`).
3. **ChArUco board** (`scripts/make_charuco.py`, e.g. 5×7, 12 mm squares): chessboard corners are subpixel-accurate and there are dozens of them. Triangulate every corner, fit a plane, and derive scale by **least-squares fit of all inter-corner distances** against the known board geometry — not from a single edge.
4. Report `scale_residual_mm` (RMS of fitted vs. known distances) in `scale.json`. This is the honest per-case scale-quality number; `side_spread_pct` is the weaker single-marker version of it.
- **Cross-check (do it once, early):** if an iPhone Pro is available, capture the same subject with Record3D / Stray Scanner, align COLMAP poses to the metric ARKit poses with Umeyama, and compare the two scale factors. Agreement within ~1% means the marker path is trustworthy; disagreement means stop and fix scale before touching FLAME. This also de-risks the intended post-G1 production path.
- IPD stays a **sanity check only** (55–70 mm), never the scale source.
- **Acceptance:** the same subject captured 3× gives scale factors within 1% of each other, and `scale_residual_mm` < 0.5 mm.

### WP3 — Masking + mesh cleanup (1 day, local)
- MediaPipe face parsing / selfie segmentation per frame → per-frame mask.
- Apply as a COLMAP `--mask_path` before MVS (better: kills hair/background at the source, cheaper than post-filtering), plus a fused.ply back-projection filter as fallback.
- Open3D statistical outlier removal + largest connected component; PyMeshLab island removal, hole fill, non-manifold repair.
- **Acceptance:** `mesh_raw.ply` contains face + neck only, no hair strands, no background sheet, watertight enough for the fit.

### WP4 — FLAME registration (3–5 days, local, MPS) — *riskiest stage*
Blocked on FLAME approval (WP0).
1. MediaPipe Face Landmarker → 2D landmarks per frame.
2. Triangulate to 3D using COLMAP poses (reuse `scale.py`'s DLT), median across frames, reject outlier views.
3. Umeyama alignment **without scale** — the scan is already in mm from WP2; letting Umeyama solve scale would silently absorb our metric error and make the measurements look better than they are. This matters: it is the difference between measuring the face and measuring the template.
4. Non-rigid fit, plain PyTorch on MPS: chamfer (KD-tree nearest-neighbour, recomputed every N iters) + landmark L2 + shape/expression regularization. Staged unlocking — rigid → shape → expression → per-vertex offsets. Adam, ~500–1000 iters.
5. Read the 11 anatomical points from **fixed FLAME vertex indices** → `landmarks.json`.
- **Acceptance:** median point-to-surface distance between the fitted FLAME and the masked scan, over the nasal region only, < 1.0 mm. Report it per case — it is the internal quality gate that predicts the caliper result before any caliper is picked up.
- **Fallback if the fit fights us:** measure directly off the cleaned scan using landmarks triangulated in step 2. Loses the under-the-skin capability and repeatability, but produces a G1 number. Decide by end of week 3, not later.

### WP5 — Measurement + error report (1 day, local)
- `measure.py` is written and correct; it just needs `landmarks.json`.
- Fill `data/calipers_template.csv` from the surgeon's blind caliper session.
- `report/compare.py` already computes per-measurement median/MAE and checks the 2 mm bar. Extend it with: per-case pass/fail, a Bland–Altman plot, and a repeatability column (same subject, 3 captures) — repeatability is what a surgeon will actually challenge.
- **Acceptance:** the G1 table for 10 subjects, produced by one command.

### WP6 — Gaussian splatting track (2–3 days, Colab) — *only after WP4 has a number*
- COLMAP poses → nerfstudio format, train splatfacto (15–20k steps on T4), surface extraction (2DGS/SuGaR-class).
- Compare against the MVS mesh on the **same** WP4 acceptance metric.
- **Decision point:** pick one track. Do not carry both past this.
- Deliberately last: it costs GPU quota and only matters if MVS is the accuracy bottleneck, which WP4's per-case surface-distance number will tell us.

### WP7 — Textured GLB + viewer (2 days)
- Texture bake in trimesh (Open3D's GLB export is weak — handover decision stands).
- Minimal three.js page, measurement overlay. **No UI work beyond this** — explicitly out of scope per the user.

### WP8 — Escape from Colab (when numbers get serious)
- Same code, Docker image, RunPod/Lambda 24 GB. Only `mvs` and `gsplat` ever run there. No code change expected — that is the point of WP1's stage split.

---

## 4. Timeline

| Week | Local | Colab |
|---|---|---|
| 1 | WP0, WP1, WP2 | one `mvs` run to unblock WP2 |
| 2 | WP2 finish, WP3 | `mvs` per new capture |
| 3 | WP4 | — |
| 4 | WP4 finish, WP5 | WP6 if WP4 says MVS is the bottleneck |
| 5 | 10-subject capture + caliper session | `mvs` ×10 |
| 6 | WP5 report → **G1 gate** | — |

---

## 5. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Marker scale error > 1 mm | **high** (already observed) | kills every mm measurement | WP2: rigid mount + ChArUco + ARKit cross-check |
| FLAME fit doesn't converge on the nose | medium | no repeatable landmarks | WP4 fallback: measure off the raw scan |
| Subject moves / talks during capture | medium | SfM fails outright | protocol is already written; enforce it, reshoot without argument |
| Colab quota exhausted mid-run | **high** (already happened) | lost session | local-first architecture; Colab touches one stage |
| MVS mesh too smooth at the alar crease | medium | nasal width error | WP6 gsplat track exists for exactly this |
| FLAME license blocks commercialization | low for PoC, **certain later** | product-level | flagged to the solution partner now, not at G1 |

---

## 6. Open questions for the surgeon (before the caliper session)

1. Exact landmark definitions for all 6 measurements, in writing, signed. `measure.py` already encodes one interpretation (e.g. Goode = alar-crease-to-tip over nasion-to-tip); it must match theirs or every error number is meaningless.
2. Caliper session must be blind and, ideally, two observers — so we can report inter-observer variability alongside our error. If the caliper's own repeatability is ±1.5 mm, a 2 mm target needs restating.
