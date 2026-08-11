# rhino-poc — Execution Plan (v4, Aug 2026)

**This is a rebuild, not a patch.** v2/v3 are superseded. What changes:

1. **No fiducial markers.** ArUco and ChArUco are dropped entirely.
2. **Two-person capture.** Subject sits still; a second person films two passes.
3. **Scale comes from ARKit/LiDAR**, cross-checked against LiDAR depth — not from a printed target.
4. Retained from v2: local-first execution, Colab GPU for `mvs`/`gsplat` only, PyTorch3D dropped.

---

## 0. Kratos — what the evidence supports

| Claim | Source | What it implies |
|---|---|---|
| "move around the person as if you were shooting a video" | site / press | A guided orbit around a still subject. Reconstruction is SfM/MVS **photogrammetry**. |
| "using only your smartphone", "standard mobile cameras" | site | Baseline path does not require LiDAR. |
| iOS 16+, **no Pro / LiDAR restriction** in the App Store listing | App Store | Confirms it runs on non-Pro iPhones. |
| "Advanced Scanning" on **iPhone Pro** adds a "4th Stage" nose capture in "ultra-high resolution" | App Store | A Pro-only extra pass. Pro is exactly the LiDAR tier, and iOS 16 is exactly the ARKit 6 4K tier — most plausibly LiDAR- and/or 4K-assisted. |
| "cloud-based algorithm trained on CNN data" | site | Server-side reconstruction. |
| Resections at Radix / Rhinion / Supratip / Infratip, "subcutaneous anatomy" | site | A statistical model fitted to the scan — the FLAME-fit role in our pipeline. |

**No printed marker appears anywhere in their material** — not in the app screenshots, not in the workflow description, not in the patient instructions. A consumer product that shipped a print-and-wear fiducial would have to say so prominently, and it doesn't. Dropping ArUco/ChArUco moves us toward their architecture, not away from it.

Since they ship a native app, their scale almost certainly comes from ARKit at capture time. That is the path this plan now takes.

---

## 1. Capture protocol — two-person, two passes

**Roles.** The subject is stationary. A second person operates the phone and walks the arc. This removes the arm's-length constraint, the awkward self-orbit, and the long static hold that v3 required.

### Setup
- Subject **seated**, upright, back against the chair, feet flat. A chair with a headrest is better than one without.
- Hair fully under a cap. Ears exposed. No glasses, no earrings.
- Eyes fixed on a **marked point** at eye level, ~3 m away. Not on the phone — tracking the phone rotates the head.
- **Neutral face, mouth closed, no talking, no swallowing on cue.** SfM cannot survive a non-rigid scene.
- Matte skin (translucent powder if shiny). Specular highlights move with the camera and are reconstructed as geometry.

### Environment — changed from earlier versions
- **Background must be static and visually textured.** This reverses the old "plain untextured background" instruction, for two reasons: ARKit's visual-inertial odometry needs environment features to hold metric tracking, and with a stationary subject the room and the face form **one rigid scene**, so background features add well-conditioned constraints to the SfM problem rather than corrupting it. Background geometry is removed later by masking (WP3), which is cheap; poor tracking is not recoverable.
- Nothing in frame may move — no other people, no screens, no windows with traffic behind them.
- Diffuse, even, **constant** lighting. No flicker; if shooting under mains-frequency lighting, verify no banding at the chosen frame rate.

### Pass 1 — eye level, ear to ear
- Filmer starts at the subject's **right ear**, ends at the **left ear** (or the reverse — consistently, and recorded in `case.json`).
- Camera at the subject's **eye level**, lens pointed at the nose.
- Distance **60–70 cm**, held constant. The filmer's forearm length is a usable gauge.
- **25–35 seconds** for the full 180°. Heel-toe walk, both hands on the phone, elbows braced against the ribs.
- Steady pace. Pauses are harmless; sudden accelerations are not — they blur frames and shake VIO.

### Pass 2 — low angle, tilted up (nasal base)
- Filmer drops to roughly **chest/chin height of the subject**, camera tilted **upward ~30°** so both nostrils, the columella and the alar creases are clearly visible.
- Same ear-to-ear arc, **15–25 seconds**, same distance.
- This pass carries the geometry behind four of the six measurements. If one pass is worth reshooting, it is this one.

### Between passes
- Let the subject relax, then re-settle and re-fix on the mark. Two short holds beat one long one.
- **Do not stop the recording between passes** if using an ARKit recorder — a single continuous session keeps one tracking origin and one metric frame, which is what makes the scale estimate global. If the app forces separate files, record them as two takes and process them as two cases that are later rigidly aligned (worse; avoid).

### Phone settings
- **Rear main lens. Zoom exactly 1.0×, never touched.** A lens switch mid-take changes the effective focal length and silently invalidates the single-camera assumption.
- **AE/AF locked before recording starts.** A take without the lock is discarded, not rescued.
- Highest resolution the recording path allows (see §2.3).

---

## 2. Scale without a fiducial

COLMAP output is unitless. The marker was the external ruler; removing it means the ruler now comes from the phone's own metric tracking. Three independent estimates, one gate.

### 2.0 First, the good news about *these six measurements*

| Measurement | Scale-dependent? | Error at 1% scale error |
|---|---|---|
| Nasofrontal angle | **no** — dimensionless | 0 |
| Nasolabial angle | **no** — dimensionless | 0 |
| Goode ratio | **no** — a ratio of two lengths | 0 |
| Nasal length (~50 mm) | yes | 0.5 mm |
| Nasal width (~35 mm) | yes | 0.35 mm |
| Midline deviation (~2 mm) | yes | 0.02 mm |

Half the measurement set is scale-invariant, and the metric half consists of **short** distances. Even a 2% scale error keeps every one of them under 1.1 mm — comfortably inside the 2 mm G1 bar and still inside the 1.5 mm G2 bar.

This reframes the whole risk picture. The marker was buying precision on the axis where this particular measurement set is most forgiving. **With scale handled to ~1%, the dominant error term becomes landmark localization (WP4), not scale.** That is where the effort should go.

This is an argument for marker-free being *sufficient*, not for being careless — a 5% scale failure still breaks G1, so the estimate must be verified per case, which is what §2.2 is for.

### 2.1 Primary: metric camera trajectory (Umeyama)

ARKit's VIO reports camera pose in **metres**, fusing IMU with camera. Reconstruct with COLMAP as usual, then estimate the similarity transform between the two camera-centre trajectories:

- Take COLMAP camera centres `C_colmap[i]` and ARKit camera centres `C_arkit[i]` for the same frames.
- Umeyama with scale → `s`, `R`, `t`. The scalar `s` is mm-per-COLMAP-unit.
- **RANSAC over frame subsets**, because VIO occasionally glitches (a relocalization jump corrupts a handful of poses and would drag a least-squares fit).
- Report inlier ratio and RMS residual after alignment.

Why this is well-conditioned here: the two-person arc translates the camera **over a metre**, in a plane, with strong parallax — close to the ideal case for both VIO and for trajectory alignment. The old arm's-length self-orbit had a fraction of that baseline. **The two-person protocol is what makes marker-free scale viable**; these two changes belong together.

### 2.2 Cross-check: LiDAR depth vs. reconstructed depth

Fully independent of the trajectory, using the same capture:

- For each frame, render/lookup COLMAP depth at pixels where the LiDAR **confidence map is high** and the **face mask** is true.
- Robustly regress `depth_lidar ≈ s · depth_colmap` (Huber or median-of-ratios) across all frames.
- iPhone LiDAR is 256×192 and roughly ±1 cm per sample, so no single reading is useful at our tolerance — but aggregated over ~10⁵ high-confidence correspondences the random component collapses. What survives is systematic bias, which is exactly what a second, differently-biased estimator is for.

**Agreement gate:** if `|s_pose − s_depth| / s_pose > 1.5%`, mark the case `scale_unverified` and do not include it in the G1 table without inspection. This is the honest replacement for the marker's `side_spread_pct` — a per-case, self-reported quality number rather than a claim of trust.

### 2.3 The capture-path decision

The tension: LiDAR/pose recording apps cap RGB resolution, while the best photogrammetry input is the highest resolution the sensor offers.

| Option | RGB | Poses | Verdict |
|---|---|---|---|
| **A. Stray Scanner** (LiDAR required) | 1920×1440 | ARKit poses + LiDAR depth + confidence + intrinsics, 30 fps | **Start here.** Everything §2.1 and §2.2 need, in one file, today. |
| **B. Record3D** | ~720p–1440p depending on mode; works without LiDAR | ARKit poses, optional depth | Fallback if the handset has no LiDAR. |
| **C. Minimal custom ARKit app** | **3840×2160 @ 30** via `recommendedVideoFormatFor4KResolution` (iPhone 11+, iOS 16+) | ARKit poses + LiDAR, same session | **The real answer.** ~250 lines of Swift. Also *is* the post-G1 production capture path. |
| D. Plain 4K video, no pose data | 3840×2160 | none | **Not viable** — no scale source at all. |

Is 1920×1440 enough for the PoC? At 65 cm the face spans roughly 60% of the 1440 axis, giving **≈4 px/mm**. That is ample for 2 mm geometry. Option A is not a compromise on accuracy for G1; it is a compromise on texture quality, which matters for WP7 and not for the measurements.

**Sequencing:** Option A now to unblock WP1–WP4. Option C once the pipeline produces numbers — it lifts texture to 4K, removes the third-party-app dependency, and is work that has to happen for the product regardless. The handover excludes "iOS/Android apps" from PoC scope; a 250-line capture utility is not that app, and this plan treats it as tooling.

### 2.4 Sanity checks (never scale sources)
- IPD in the 55–70 mm band. Population variance is ±3–4 mm, so it can flag a gross failure and nothing finer.
- Head bounding box 200–250 mm.
- Both reported in `scale.json`; neither ever sets `s`.

### 2.5 Rejected
- **ArUco / ChArUco** — dropped per the new direction.
- **Any known-size object in frame** — a fiducial by another name.
- **Metric monocular depth networks** (Depth Pro, Metric3D v2) — 2–5% error, worse than both estimators above. Reconsider only if the handset has neither LiDAR nor usable VIO.

---

## 3. Reconstruction

Unchanged in substance — COLMAP SfM → MVS → Poisson, with a gsplat track evaluated later — plus two upgrades that the ARKit data makes possible:

- **Known intrinsics.** ARKit reports per-frame intrinsics. Feed them to COLMAP as a fixed camera rather than solving for them. Fewer free parameters, less chance of focal/distortion error being absorbed into geometry.
- **Pose priors, if the mapper struggles.** ARKit poses can seed or constrain the mapper (or drive `point_triangulator` with fixed extrinsics). Hold this in reserve: VIO drift and rolling shutter make ARKit poses good priors and poor ground truth, so free-solve-then-align (§2.1) stays the default.

Frame selection returns to the v1 approach — ffmpeg/decode at a fixed rate, Laplacian-variance blur rejection, best-frame-per-time-window so angular coverage is preserved. Target ~250–350 frames across both passes, weighted toward pass 2.

---

## 4. Execution split: local-first

Only two stages genuinely need CUDA. Everything else runs on the M4 Pro (12 cores, 24 GB).

| Stage | CUDA? | Runs on |
|---|---|---|
| (a) ingest — ARKit dataset parse, frame select, blur reject | no | **local** |
| (b) SfM — COLMAP feature/match/mapper | no | **local** |
| (c1) MVS — `patch_match_stereo` | **yes** | **Colab T4** |
| (c2) Gaussian splatting | **yes** | **Colab T4** |
| (d) masking — MediaPipe face parsing | no | **local** |
| (e) scale — Umeyama + depth regression | no | **local** |
| (f) FLAME fit | no, written in plain PyTorch | **local (MPS)** |
| (g) measure / (h) export / report | no | **local** |

**PyTorch3D stays dropped.** Chamfer distance plus a mesh container is ~40 lines of plain PyTorch and a KD-tree; keeping PyTorch3D would pin the most iteration-heavy stage to a CUDA box and burn Colab quota on every debug cycle. Colab's job is one command per case.

---

## 5. Handoff (local ⇄ Colab)

Case directory is the exchange unit; Drive is the transport; nothing binary goes into git.

```
vaka_001/
  case.json          # manifest: stages, params, hashes, pass direction
  arkit/             # raw Stray Scanner export (rgb, depth, confidence, odometry, intrinsics)
  images/            # selected frames for COLMAP           -> Colab
  masks/             # per-frame face masks                 -> Colab
  colmap/sparse/0/   # local SfM output                     -> Colab
  colmap/dense/      # Colab output (fused.ply), not synced back whole
  mesh_raw.ply       # Colab output                         -> local
  scale.json         # s_pose, s_depth, agreement, residuals, IPD, bbox
  landmarks.json / model.glb / measurements.json
```

```bash
# LOCAL
poc run ingest  vaka_001.r3d --out vaka_001      # or --stray <dir>
poc run mask    --out vaka_001
poc run sfm     --out vaka_001
poc pack        vaka_001 --for gpu               # images + masks + sparse

# COLAB (GPU)
poc unpack vaka_001_gpu.zip --out /content/vaka_001
poc run mvs --out /content/vaka_001
poc pack    /content/vaka_001 --for local        # mesh_raw.ply + logs

# LOCAL
poc unpack vaka_001_local.zip --out vaka_001
poc run scale   --out vaka_001                   # no --marker-mm any more
poc run flame   --out vaka_001
poc run measure --out vaka_001
poc run export  --out vaka_001
```

---

## 6. Work packages

### WP0 — Local environment (half a day)
- `brew install colmap ffmpeg` (no CUDA on macOS — expected; MVS is Colab's job).
- Python 3.11 (system 3.9 is too old). `uv venv && uv pip install -e .` plus `mediapipe`, `torch`, `pymeshlab`, `scipy`.
- **Acceptance:** `colmap -h` runs; `import cv2, open3d, trimesh, mediapipe, torch` clean; `torch.backends.mps.is_available()` is True.
- Open the FLAME account now (`flame.is.tue.mpg.de`) — approval is not instant and WP4 blocks on it.
- Install Stray Scanner (or Record3D) on the handset; confirm export reaches the Mac.

### WP1 — ARKit ingest + stage runner + pack/unpack (2 days)
- **New `pipeline/arkit.py`:** parse Stray Scanner (`rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`) and Record3D `.r3d`. Emit a normalized `ArkitCapture`: per-frame metric pose (4×4), intrinsics, depth, confidence.
- **`pipeline/frames.py`:** decode + blur-reject + window-best selection, carrying the ARKit frame index through so every selected image keeps its pose.
- Refactor `cli.py` to `poc run <stage>`; `case.json` manifest; `poc pack/unpack`.
- Split `--no-gpu` into `--sift-gpu/--no-sift-gpu`; `mvs` fails loudly with a readable message when CUDA is absent.
- **Acceptance:** a case reruns with zero recomputation; every selected frame resolves to an ARKit pose; gpu zip < 400 MB.

### WP2 — Marker-free scale (2 days) — *rebuilt from scratch*
- **Delete `scripts/make_aruco.py` and the ArUco path in `scale.py`.**
- `scale_from_poses()` — RANSAC Umeyama over camera centres → `s_pose`, inlier ratio, RMS residual (§2.1).
- `scale_from_depth()` — robust `depth_lidar ≈ s·depth_colmap` regression over high-confidence, in-mask pixels → `s_depth` (§2.2).
- `compute_scale()` — combine, apply the 1.5% agreement gate, write IPD and bbox sanity, set `scale_verified`.
- **Acceptance:** same subject captured 3× yields `s` within **1%** across takes, and `|s_pose − s_depth|` within 1.5% on every take.
- **If the handset has no LiDAR:** `s_depth` is unavailable and the gate degrades to repeatability across takes. Say so in the report rather than quietly dropping the check.

### WP3 — Masking + mesh cleanup (1 day)
- MediaPipe face parsing per selected frame → masks, fed to COLMAP as `--mask_path` (kills hair and the now-deliberately-textured background at the source).
- Open3D statistical outlier removal + largest connected component; PyMeshLab island removal, hole fill, non-manifold repair.
- Masks are also required by §2.2, so this lands **before** WP2 completes.
- **Acceptance:** `mesh_raw.ply` is face + neck only, no hair, no room.

### WP4 — FLAME registration (4–6 days, local/MPS) — *now the dominant error term*
1. MediaPipe Face Landmarker → 2D landmarks per frame.
2. Triangulate to 3D with COLMAP poses, median across frames, reject outlier views.
3. Umeyama **without scale** — the scan is already metric from WP2. Letting the fit solve scale would absorb our metric error and flatter the result.
4. Non-rigid fit, plain PyTorch on MPS: chamfer (KD-tree NN, refreshed every N iters) + landmark L2 + shape/expression regularization. Staged unlocking: rigid → shape → expression → per-vertex offsets. Adam, 500–1000 iters.
5. Read the 11 anatomical points from fixed FLAME vertex indices → `landmarks.json`.
- **Acceptance:** median point-to-surface distance, fitted FLAME vs. masked scan, **nasal region only**, < 1.0 mm, reported per case. This is the internal gate that predicts the caliper result.
- **Fallback:** measure directly off the cleaned scan using the step-2 landmarks. Loses under-the-skin capability and some repeatability, but produces a G1 number. Decide by end of week 3.
- Budget increased over v3 by design — per §2.0, this is where accuracy is now won or lost.

### WP5 — Measurement + error report (1 day)
- `measure.py` is written and correct; it needs `landmarks.json`.
- Extend `report/compare.py`: per-case pass/fail, Bland–Altman, and a **repeatability column** (same subject × 3 takes). With scale now self-reported rather than externally certified, repeatability is the strongest evidence we can offer a surgeon.
- **Acceptance:** the 10-subject G1 table from one command.

### WP6 — Gaussian splatting track (2–3 days, Colab) — only after WP4 has a number
- COLMAP poses → nerfstudio, splatfacto 15–20k steps on T4, surface extraction; compare on the **same** WP4 metric. Pick one track; do not carry both.

### WP7 — Texture + viewer (2 days)
- Texture bake in trimesh. Revisit after Option C (4K) — texture is the one thing 1920×1440 genuinely limits.
- Minimal three.js page with measurement overlay. **No UI work beyond this.**

### WP8 — Custom ARKit capture app (Option C, 3–4 days, Swift)
- `ARWorldTrackingConfiguration` + `recommendedVideoFormatFor4KResolution`, LiDAR depth, per-frame pose/intrinsics, single continuous session across both passes, on-device pass-quality checks (AE/AF lock, pace, coverage).
- Schedule after G1 unless texture or third-party export becomes the blocker sooner. This is the production capture path.

---

## 7. Timeline

| Week | Local | Colab |
|---|---|---|
| 1 | WP0, WP1 | one `mvs` to unblock WP2/WP3 |
| 2 | WP3, WP2 | `mvs` per take |
| 3 | WP4 | — |
| 4 | WP4 finish, WP5 | WP6 if WP4 says MVS is the bottleneck |
| 5 | 10-subject capture + blind caliper session | `mvs` ×10 |
| 6 | WP5 report → **G1 gate** | — |

---

## 8. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **ARKit VIO scale error > 2%** | medium | metric measurements drift | §2.2 depth cross-check + 3-take repeatability; §2.0 shows even 2% stays under the bar |
| **Handset has no LiDAR** | *unknown — see §10* | loses the independent cross-check | Record3D VIO-only path; gate degrades to repeatability, stated openly in the report |
| **Subject moves during a pass** | medium | non-rigid scene → SfM fails or warps | two short passes with a reset; headrest; eyes on a fixed mark; reshoot without argument |
| Landmark localization error | **high** | now the dominant error term | WP4 budget increased; nasal-region surface-distance gate per case |
| Lens/zoom change mid-take | medium | metrically wrong yet plausible reconstruction | ingest validates constant intrinsics across the ARKit stream |
| Featureless background breaks VIO | medium | no metric poses at all | protocol reversed to require a static **textured** background |
| Colab quota exhausted mid-run | **high** (happened once) | lost session | local-first; Colab touches one stage |
| MVS too smooth at the alar crease | medium | nasal width error | dense pass-2 coverage; WP6 exists for this |
| FLAME licence blocks commercialization | certain, later | product-level | flagged to the solution partner now, not at G1 |

---

## 9. What was removed, and why it is recoverable

The marker gave one thing this plan no longer has: an **externally certified** ruler, independent of the phone. Everything in §2 is the phone measuring itself, and two estimators derived from the same hardware can share a bias that neither will reveal.

The mitigation is deliberately not another marker. It is **one calibration exercise, once**: capture a rigid object of precisely known dimensions (a machined block, or a printed checkerboard measured with calipers — used as a *validation target*, never in a patient capture) through the full pipeline, and confirm the recovered scale against its true size. That converts "two estimators agree" into "the system is known accurate to X%", and it never touches the clinical protocol. Half a day, worth doing in week 2.

---

## 10. Open questions

1. **Which iPhone is available for capture?** LiDAR requires a Pro model (12 Pro or later) or an iPad Pro. With LiDAR: §2.2 cross-check works, plus a depth prior for low-texture cheeks and forehead. Without it: ARKit VIO still gives metric poses via Record3D and §2.1 stands alone. This changes WP2's acceptance criteria, so it is the one answer needed before WP2 starts.
2. **Exact landmark definitions for the six measurements, in writing, signed by the surgeon.** `measure.py` encodes one interpretation (e.g. Goode = alar-crease-to-tip ÷ nasion-to-tip); if it differs from theirs, every error number is meaningless.
3. **Blind caliper session with two observers**, so inter-observer variability is reported alongside our error. If the caliper's own repeatability is ±1.5 mm, a 2 mm target needs restating.
