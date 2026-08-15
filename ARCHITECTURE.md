# Authoritative facial geometry architecture

## Core invariant

Each reconstruction has exactly one versioned metric surface. `face_geometry.ply` is that surface;
its coordinates are metres in the COLMAP world frame. `geometry.json` identifies its float32
positions and ordered triangles with SHA-256 hashes. Measurements, landmarks, clicks, simulations,
and visual assets must declare the same `geometry_id`.

`face_mesh_raw.ply` is not authoritative. It is the unitless Poisson output retained for diagnostics
and texture projection. Changing scale, trimming, smoothing, remeshing, or displacement after the
authoritative asset is created produces a different geometry and therefore requires a new
`geometry_id`.

## Assets and correspondence

```text
face_mesh_raw.ply + verified scale
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

A simulation must deform authoritative vertices once. Landmarks are recomputed from their stored
triangle and barycentric coordinates. UV-seam render vertices follow their corresponding canonical
corners, then normals and tangents are recomputed. Measurements are evaluated on the deformed
authoritative surface. A saved simulation is a new geometry state with its parent `geometry_id`,
deformation parameters, and deterministic output identity recorded.

Topology-changing simulation requires an explicit old-to-new surface correspondence. It must not
reuse old landmark bindings or measurements without remapping and validation.

## Acceptance state

`quality_report.json` is the machine-readable decision record for a case. It evaluates capture,
masking, sparse registration, scale, authoritative mesh integrity, landmark localization,
geometry/visual identity, measurement definition status, and runtime independently. A successful
file export does not imply an accepted case, and `clinical_use_authorized` remains false until a
separate clinical validation process is completed.
