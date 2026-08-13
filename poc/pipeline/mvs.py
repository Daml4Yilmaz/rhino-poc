"""Adim (c1): Klasik hat — COLMAP MVS (patch match) + Poisson mesh.

DIKKAT: patch_match_stereo CUDA'li COLMAP gerektirir. CPU-only COLMAP'ta bu
adim calismaz (notebook'taki kurulum notuna bak).
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d

from ._proc import run_logged


def _auto_cache_gb() -> int:
    """Toplam RAM'in ~%30'u, 2-8 GB araliginda.

    Colab (12.7 GB) -> 3, buyuk bir makine (64 GB) -> 8. COLMAP'in 32 GB'lik
    varsayilani neredeyse her paylasimli ortamda OOM demek.
    """
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return max(2, min(8, int(total / 1e9 * 0.30)))
    except (ValueError, OSError, AttributeError):
        return 4


def _tune_patchmatch(cfg: Path, src_images: int, max_refs: int) -> int:
    """patch-match.cfg'yi is miktarina gore ayarla; referans sayisini dondur.

    Dosya bicimi ikili satirlardir:  <goruntu adi>\\n__auto__, <komsu>\\n

    Iki kaldirac da maliyeti DOGRUSAL etkiler ve ikisi de fazla ayarlanmis
    gelir:

    REFERANS SAYISI — image_undistorter kaydedilen her goruntuyu referans
    yapar. 151 dereceyi 300 kareyle taramak goruntu basina 0.5 derece demek;
    yuzey rekonstruksiyonu icin bu ciddi bir fazlalik. ~1.5 derece aralik
    (≈100-150 referans) fuzyon icin fazlasiyla yeterli. Alt ornekleme
    ZAMANDA duzgun dagilir, yoksa yayin bir ucu bos kalir.

    KOMSU SAYISI — varsayilan 20. Yuz gibi kucuk ve yogun ortusmeli bir
    sahnede 10 komsu ayni yuzeyi verir.

    Cozunurluge dokunulmaz: dogrulugun bagli oldugu tek parametre odur.
    """
    if not cfg.exists():
        return 0
    lines = [l for l in cfg.read_text().splitlines() if l.strip()]
    blocks = [(lines[i], lines[i + 1]) for i in range(0, len(lines) - 1, 2)]
    n0 = len(blocks)
    if n0 == 0:
        return 0

    if max_refs and n0 > max_refs:
        keep = np.linspace(0, n0 - 1, max_refs).round().astype(int)
        blocks = [blocks[i] for i in sorted(set(keep.tolist()))]

    out = []
    for name, nb in blocks:
        out.append(name)
        out.append(re.sub(r"__auto__,\s*\d+", f"__auto__, {src_images}", nb)
                   if src_images else nb)
    cfg.write_text("\n".join(out) + "\n")

    # fusion.cfg AYNI listeye indirilmeli. stereo_fusion kendi listesini
    # oradan okur; patch-match'i 150'ye indirip fusion'a 300 birakirsak
    # fuzyon var olmayan derinlik haritalarini arar ve coker.
    fus = cfg.parent / "fusion.cfg"
    if fus.exists() and len(blocks) != n0:
        fus.write_text("\n".join(name for name, _ in blocks) + "\n")

    if len(blocks) != n0 or src_images:
        print(f"[mvs] patch-match: {n0} -> {len(blocks)} referans, "
              f"komsu {src_images or 'varsayilan'}")
    return len(blocks)


def run_mvs(frames_dir: Path, sparse_model: Path, dense_dir: Path,
            colmap_bin: str = "colmap", poisson_depth: int = 10,
            poisson_trim: float = 7.0, cache_size_gb: int | None = None,
            max_image_size: int | None = None,
            src_images: int = 10, max_refs: int = 150,
            geom_consistency: bool = True) -> Path:
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
        run_logged([colmap_bin, "image_undistorter",
                    "--image_path", str(frames_dir),
                    "--input_path", str(sparse_model),
                    "--output_path", str(dense_dir),
                    "--output_type", "COLMAP"], log, "undistort")

    n_ref = _tune_patchmatch(dense_dir / "stereo" / "patch-match.cfg",
                             src_images, max_refs)

    # cache_size VARSAYILANI 32 GB'dir; Colab'in ~12.7 GB RAM'inde surec
    # OOM killer tarafindan SIGKILL ile oldurulur (abort degil, sessiz olum).
    # Onbellek boyutu KALITEYI etkilemez, sadece disk trafigini artirir —
    # mevcut bellege gore olcekle.
    cache_gb = cache_size_gb or _auto_cache_gb()
    cmd = [colmap_bin, "patch_match_stereo",
           "--workspace_path", str(dense_dir),
           "--PatchMatchStereo.geom_consistency",
           "true" if geom_consistency else "false",
           "--PatchMatchStereo.cache_size", str(cache_gb)]
    if max_image_size:
        cmd += ["--PatchMatchStereo.max_image_size", str(max_image_size)]
    # Kaba sure tahmini: T4'te 1600px'te referans basina ~20 sn fotometrik,
    # geometrik gecis bunun ~3 kati. Kullaniciya ne bekleyecegini soyle.
    est = n_ref * 20 * (4 if geom_consistency else 1) / 60
    print(f"[mvs] patch_match: {n_ref} referans, onbellek {cache_gb} GB "
          f"— T4'te kabaca {est:.0f} dk (hazir derinlik haritalari atlanir)"
          + ("" if geom_consistency else "  [geometrik gecis KAPALI]"))
    suf = ".geometric.bin" if geom_consistency else ".photometric.bin"
    run_logged(cmd, log, "patch_match",
               watch=(dense_dir / "stereo" / "depth_maps", suf, n_ref))

    # stereo_fusion da ayni tuzagi tasir, hatta daha kotusu: use_cache
    # VARSAYILANI 0'dir, yani butun derinlik/normal haritalarini bellege
    # yukler. Acinca sinirli bir LRU onbellege gecer — 300 goruntude fark
    # OOM ile tamamlanmis fuzyon arasindaki fark oluyor.
    # geom kapaliysa fuzyon FOTOMETRIK haritalari okumali; varsayilani
    # "geometric" oldugu icin aksi halde var olmayan dosyalari arar.
    fused = dense_dir / "fused.ply"
    run_logged([colmap_bin, "stereo_fusion",
                "--workspace_path", str(dense_dir),
                "--output_path", str(fused),
                "--input_type",
                "geometric" if geom_consistency else "photometric",
                "--StereoFusion.use_cache", "1",
                "--StereoFusion.cache_size", str(cache_gb)], log, "fusion")

    # Poisson: Open3D ile (yogunluk dusuk vertexleri kirparak).
    # Tek uzun cagri, ara ilerleme vermez — en azindan giris boyutunu ve
    # sureyi bildir ki takilmis mi anlasilsin.
    import time as _t
    _t0 = _t.time()
    print("[poisson] fused.ply okunuyor…", flush=True)
    pcd = o3d.io.read_point_cloud(str(fused))
    print(f"[poisson] {len(pcd.points)} nokta -> yuzey cikariliyor "
          f"(depth={poisson_depth}, birkac dakika surebilir)", flush=True)
    # fused.ply karelerden okunan RGB'yi tasir; Poisson bunu mesh'e aktarir.
    # Renk buradan kaybolursa GLB gri cikar — erken haber ver.
    if not pcd.has_colors():
        print("[mvs] UYARI: fused.ply renk tasimiyor — mesh renksiz olacak.")
    if not pcd.has_normals():
        pcd.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth)
    dens = np.asarray(densities)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, poisson_trim / 100.0))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    print(f"[poisson] bitti — {_t.time()-_t0:.0f} sn", flush=True)
    out = dense_dir.parent.parent / "mesh_raw.ply"
    o3d.io.write_triangle_mesh(str(out), mesh)
    print(f"[mvs] mesh: {len(mesh.vertices)} vertex, "
          f"renk {'var' if mesh.has_vertex_colors() else 'YOK'} -> {out}")
    return out
