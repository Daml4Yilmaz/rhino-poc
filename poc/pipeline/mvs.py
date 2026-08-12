"""Adim (c1): Klasik hat — COLMAP MVS (patch match) + Poisson mesh.

DIKKAT: patch_match_stereo CUDA'li COLMAP gerektirir. CPU-only COLMAP'ta bu
adim calismaz (notebook'taki kurulum notuna bak).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import open3d as o3d


def _run(cmd: list[str], log_file: Path) -> None:
    with open(log_file, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def run_mvs(frames_dir: Path, sparse_model: Path, dense_dir: Path,
            colmap_bin: str = "colmap", poisson_depth: int = 10,
            poisson_trim: float = 7.0) -> Path:
    """fused.ply + mesh_raw.ply uretir; mesh yolunu dondurur."""
    dense_dir.mkdir(parents=True, exist_ok=True)
    log = dense_dir.parent / "colmap.log"

    # image_undistorter var olan dosyanin uzerine YAZMAZ; onceki yarim kalmis
    # kosudan kalan dense/images yuzunden "filesystem error: File exists" ile
    # SIGABRT verir. Kopan bir Colab oturumundan sonra tam olarak bu olur.
    #
    # dense/ klasorunu topluca SILMIYORUZ: pahali olan stereo/depth_maps orada
    # ve COLMAP hazir olanlari kendisi atlayip devam edebiliyor. Sadece images
    # klasorunu ele aliyoruz.
    import pycolmap
    n_expected = pycolmap.Reconstruction(str(sparse_model)).num_reg_images()
    images_dst = dense_dir / "images"
    n_have = len(list(images_dst.glob("*"))) if images_dst.is_dir() else 0

    if n_have == n_expected and (dense_dir / "sparse").exists():
        print(f"[mvs] undistort atlandi: {n_have} goruntu zaten hazir")
    else:
        if images_dst.exists():
            print(f"[mvs] yarim undistort temizleniyor ({n_have}/{n_expected} "
                  "goruntu) — derinlik haritalari korunuyor")
            shutil.rmtree(images_dst)
        _run([colmap_bin, "image_undistorter",
              "--image_path", str(frames_dir),
              "--input_path", str(sparse_model),
              "--output_path", str(dense_dir),
              "--output_type", "COLMAP"], log)

    _run([colmap_bin, "patch_match_stereo",
          "--workspace_path", str(dense_dir),
          "--PatchMatchStereo.geom_consistency", "true"], log)

    fused = dense_dir / "fused.ply"
    _run([colmap_bin, "stereo_fusion",
          "--workspace_path", str(dense_dir),
          "--output_path", str(fused)], log)

    # Poisson: Open3D ile (yogunluk dusuk vertexleri kirparak)
    pcd = o3d.io.read_point_cloud(str(fused))
    # fused.ply karelerden okunan RGB'yi tasir; Poisson bunu mesh'e aktarir.
    # Renk buradan kaybolursa GLB gri cikar — erken haber ver.
    if not pcd.has_colors():
        print("[mvs] UYARI: fused.ply renk tasimiyor — mesh renksiz olacak.")
    if not pcd.has_normals():
        pcd.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth)
    import numpy as np
    dens = np.asarray(densities)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, poisson_trim / 100.0))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    out = dense_dir.parent.parent / "mesh_raw.ply"
    o3d.io.write_triangle_mesh(str(out), mesh)
    print(f"[mvs] mesh: {len(mesh.vertices)} vertex, "
          f"renk {'var' if mesh.has_vertex_colors() else 'YOK'} -> {out}")
    return out
