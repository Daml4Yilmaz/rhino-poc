"""Adim (h): olcekli GLB disa aktarimi (simdilik dokusuz; doku hafta 2-3)."""
from __future__ import annotations

from pathlib import Path

import trimesh


def export_glb(mesh_ply: Path, scale_mm_per_unit: float, out_glb: Path) -> Path:
    mesh = trimesh.load(str(mesh_ply), force="mesh")
    mesh.apply_scale(scale_mm_per_unit)  # artik mm cinsinden
    mesh.export(str(out_glb))
    ext = mesh.bounding_box.extents
    print(f"[export] {out_glb}  (bbox mm: "
          f"{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f})")
    return out_glb
