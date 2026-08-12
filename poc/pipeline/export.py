"""Adim (h): olcekli, RENKLI GLB disa aktarimi.

Renk COLMAP'ten bedava gelir: `stereo_fusion` her noktaya karelerden okunan
RGB'yi yazar (`fused.ply`), Poisson da bunu mesh'e tasir. Burada tek isimiz
o rengi GLB'ye kadar kaybetmeden goturmek.

Neden trimesh'e dogrudan okutmuyoruz: `trimesh.load(..., process=True)`
(varsayilan) yakin vertexleri birlestirir ve vertex renkleri sessizce
bozulabilir/dusebilir. Bu yuzden geometriyi ve renkleri Open3D ile okuyup
trimesh'e acikca veriyoruz — GLB yazimi yine trimesh'te, cunku Open3D'nin
GLB destegi zayif.

NOT: bu VERTEX RENGI, doku (UV texture) degil. Cozunurlugu mesh yogunluguyla
sinirlidir; ben/kirisiklik gibi ince detay icin karelerden UV doku pisirmek
gerekir — o WP7 isi ve olcumleri etkilemez.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def export_glb(mesh_ply: Path, scale_mm_per_unit: float, out_glb: Path) -> Path:
    m = o3d.io.read_triangle_mesh(str(mesh_ply))
    if len(m.vertices) == 0:
        raise RuntimeError(f"Mesh bos veya okunamadi: {mesh_ply}")

    verts = np.asarray(m.vertices, dtype=np.float64) * scale_mm_per_unit
    faces = np.asarray(m.triangles, dtype=np.int64)

    colors = None
    if m.has_vertex_colors():
        rgb = np.asarray(m.vertex_colors)
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        alpha = np.full((len(rgb), 1), 255, np.uint8)
        colors = np.concatenate([rgb, alpha], axis=1)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_colors=colors, process=False)
    mesh.export(str(out_glb))

    ext = mesh.bounding_box.extents
    unit = "mm" if scale_mm_per_unit != 1.0 else "birim (OLCEKSIZ)"
    print(f"[export] {out_glb}  (bbox {unit}: "
          f"{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f})")
    if colors is None:
        print("[export] UYARI: mesh'te vertex rengi yok — GLB gri cikacak. "
              "fused.ply renk tasiyor mu diye bak (stereo_fusion ciktisi).")
    else:
        print(f"[export] {len(colors)} vertex rengi tasindi (RGBA)")
    return out_glb
