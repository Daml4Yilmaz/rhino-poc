# Authoritative facial geometry architecture

## Core invariant

Each reconstruction has exactly one versioned metric surface. `face_geometry.ply` is that surface;
its coordinates are metres in the COLMAP world frame. `geometry.json` identifies its float32
positions and ordered triangles with SHA-256 hashes. Measurements, landmarks, clicks, simulations,
and visual assets must declare the same `geometry_id`.

`face_dense_fused.ply` is the permanent, direct output of COLMAP stereo fusion before normal
estimation and Poisson reconstruction. `face_mesh_raw.ply` is not authoritative; it is the unitless
Poisson output retained for diagnostics and texture projection. Changing scale, trimming,
smoothing, remeshing, or displacement after the authoritative asset is created produces a
different geometry and therefore requires a new `geometry_id`.

## Assets and correspondence

```text
face_dense_fused.ply
        |
        +---- Poisson ----> face_mesh_raw.ply + verified scale
                 |
                 v
        face_geometry.ply  <---- geometry.json
          (metres)                  |
             |                      +---- landmarks.json
             |                      |       triangle + barycentric binding
             |                      +---- measurements.json
             |
             +---- COLMAP UV atlas -----> face_model.glb
                    same ordered faces      textured visual surface
```

The initial untextured GLB has identical vertices and triangle indices. UV atlases require a
rendering vertex to be duplicated where it belongs to two texture seams. In the textured GLB,
render face `i` is therefore guaranteed to represent authoritative triangle `i`, and all three
rendered corner positions are verified against that triangle before export. This is a registered
representation, not an independently reconstructed mesh.

The albedo atlas is projected from the original registered RGB frames, not from the
black-background images used for dense geometry. The exported material is non-metallic PBR with a
conservative skin roughness. These appearance choices do not change vertex positions or
measurement coordinates.

`geometry.json` records which of these relationships applies. Export fails if triangle count,
triangle order, or surface position changes beyond one micrometre.

## Picking and landmarks

Application raycasting must use `face_geometry.ply`, not a shader-displaced visual surface. A hit
is stored as:

```json
{
  "geometry_id": "sha256:...",
  "triangle_index": 12345,
  "barycentric": [0.2, 0.3, 0.5]
}
```

The exact metric point is the barycentric combination of that triangle's three vertices. Existing
automatic landmarks use this representation in `landmarks.json`. A renderer may use the GLB face
index to select the same authoritative triangle, or raycast the companion PLY directly. Never map a
click by nearest visual vertex.

## Appearance maps

- Albedo, roughness, and tangent-space normal maps are appearance only.
- Normal maps never alter raycasting, landmarks, distances, or angles.
- Shader displacement is visual only by default; picking still targets the undisplaced metric
  surface.
- Displacement reconstructed from capture data may affect measurements only after accuracy
  validation and promotion to a new authoritative mesh and `geometry_id`.

The current pipeline creates a color-corrected COLMAP UV atlas and embeds it in the GLB. Normal and
roughness maps are future appearance assets; they must follow the same policies.

## Surgical deformation

The experimental dorsal-hump operation reads authoritative vertices once and writes a separate
geometry state. It derives superior–inferior, left–right, and anterior axes from existing nasal
landmarks, estimates the observed midline dorsal envelope, fits a smooth target through the outer
dorsal bands, and uses only positive convex excess to localize the hump. The normalized hump weight
is searched only in the upper/mid-dorsum for one explicit apex. The requested correction is capped
at the detected convexity so it cannot create a sagittal scoop. The complete connected upper/mid
convexity receives the apex correction fraction, with C2 fades at the nasion and supratip; this keeps
the superior shoulder from being attenuated more strongly than the peak. A separate C2 ceiling is
inactive for ordinary targets but limits an excessive distal target to 35% of the applied peak at the
lower-dorsum boundary and zero at the supratip. This keeps a larger distal convexity from exceeding
the lower-to-peak safety limit without sharpening an already-safe anatomical fade.

Deformation uses one vector-valued biharmonic solve over one connected dorsal surface. At every
longitudinal position, `delta(s)` drives a transverse target whose ridge weight is strongest, whose
slopes retain more than 90% of that target, and whose sidewalls taper continuously to zero only at
the anatomical vault perimeter. Posterior and mild medial components share the same constraint
system, so the centerline cannot move independently of the vault. Triangle adjacency, fixed
nasion/supratip/perimeter boundaries, and the smooth target field preserve continuous transitions.
The operation does not displace along vertex normals or symmetrize the source face.

The original PLY, GLB, `geometry.json`, `landmarks.json`, and measurements remain untouched. UV-seam
render vertices follow their corresponding canonical corners in the simulated GLB. A saved
simulation records its parent `geometry_id`, its own deterministic output identity, deformation
parameters, affected vertex count, and explicit non-clinical status. At 0.0 mm, the simulated PLY
is a byte-for-byte copy of the authoritative PLY. Non-zero export is rejected unless the persisted
PLY hash differs, displacement is meaningfully close to the applied correction, the sagittal
profiles separate, the transverse GLB coordinates change, and the reloaded GLB corners match the
simulated authoritative surface. Profile curves, named profile-point displacements, four transverse
sections, clay/normal renders, and a moved-vertex heatmap are persisted for audit. Viewer downloads
use the simulation geometry hash in the filename to avoid stale-file ambiguity.

Topology-changing simulation requires an explicit old-to-new surface correspondence. It must not
reuse old landmark bindings or measurements without remapping and validation.

## Acceptance state

`quality_report.json` is the machine-readable decision record for a case. It evaluates capture,
masking, sparse registration, scale, authoritative mesh integrity, landmark localization,
geometry/visual identity, measurement definition status, and runtime independently. A successful
file export does not imply an accepted case, and `clinical_use_authorized` remains false until a
separate clinical validation process is completed.
