"""Adim (e): markersiz mutlak olcek.

Fotogrametri (COLMAP) birimsiz bir model verir — sekil dogru, boyut yok.
Marker kullanmadigimiz icin "1 COLMAP birimi kac mm" carpani telefonun
metrik ARKit takibinden gelir. Model hala tamamen fotogrametriyle uretilir;
burada uretilen tek sey bir sayidir.

Iki BAGIMSIZ tahmin uretilir ve birbirine karsi kontrol edilir:

  s_pose   Kamera yorungeleri. COLMAP kamera merkezleri ile ARKit kamera
           merkezleri arasindaki benzerlik donusumunun olcegi. Iki kisilik
           protokolde kamera >1.5 m yol aldigi icin bu iyi kosullu.
  s_depth  LiDAR derinligi. Seyrek COLMAP noktalarinin kare icindeki
           derinligi ile ayni pikseldeki LiDAR derinliginin orani. Tek bir
           LiDAR okumasi +-1 cm gurultuludur ama 10^5 esleşmede rastgele
           bilesen sonumlenir; kalan sistematik sapma zaten ikinci bir
           tahminciyle yakalamak istedigimiz sey.

Ikisi %1.5'ten fazla ayrisirsa vaka `scale_verified=false` isaretlenir.
Marker'in `side_spread_pct` kontrolunun yerini bu tutar.

NOT: s_pose olcegi sadece kamera MERKEZLERINDEN cikar; ARKit (OpenGL, -Z
ileri) ile COLMAP (OpenCV, +Z ileri) eksen farki bu yuzden onemsizdir.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .arkit import ArkitCapture, load_capture

RNG = np.random.default_rng(0)


def _cam_from_world(img):
    """pycolmap surum farki: 3.x'te ozellik, 4.x'te metot."""
    cfw = img.cam_from_world
    return cfw() if callable(cfw) else cfw


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """dst ~ s*R@src + t  (Umeyama 1991). src,dst: (N,3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = (D.T @ S) / len(src)
    U, d, Vt = np.linalg.svd(cov)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1.0
    R = U @ W @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = float(np.trace(np.diag(d) @ W) / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def _frame_map(case_dir: Path) -> dict[str, int]:
    """COLMAP goruntu adi -> ARKit kare indeksi (frames.py yazar)."""
    p = case_dir / "frames_index.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} yok. Olcek, her karenin hangi ARKit karesi oldugunu bilmek "
            "zorunda — 'frames' adimini bu surumle tekrar kos.")
    return json.loads(p.read_text())["frame_of_image"]


def scale_from_poses(rec: pycolmap.Reconstruction, cap: ArkitCapture,
                     fmap: dict[str, int], n_pairs: int = 20000) -> dict:
    """Yorunge tabanli olcek (metre / COLMAP birimi).

    Olcegi ikili mesafe oranlarinin MEDYANINDAN alir: bu, VIO'nun donme
    kaymasindan ve tek tuk relokalizasyon sicramasindan etkilenmez. Sonra
    ayni ic noktalarla tam Umeyama kosup artigi raporlar.
    """
    C_col, C_ark = [], []
    for img in rec.images.values():
        fi = fmap.get(img.name)
        if fi is None or fi >= cap.n_frames:
            continue
        C_col.append(_cam_from_world(img).inverse().translation)
        C_ark.append(cap.centers[fi])
    if len(C_col) < 30:
        raise RuntimeError(
            f"Sadece {len(C_col)} karede hem COLMAP pozu hem ARKit pozu var "
            "(<30). SfM cok az kare kaydetmis ya da kare eslesmesi bozuk.")
    C_col = np.asarray(C_col, np.float64)
    C_ark = np.asarray(C_ark, np.float64)

    n = len(C_col)
    i = RNG.integers(0, n, n_pairs)
    j = RNG.integers(0, n, n_pairs)
    d_col = np.linalg.norm(C_col[i] - C_col[j], axis=1)
    d_ark = np.linalg.norm(C_ark[i] - C_ark[j], axis=1)

    # Kisa taban cizgileri oranı patlatir: yorunge capinin %10'undan kisa
    # ciftleri at.
    span = float(np.linalg.norm(C_col.max(0) - C_col.min(0)))
    ok = d_col > 0.10 * span
    if ok.sum() < 100:
        raise RuntimeError("Yeterli uzun taban cizgisi yok — kamera neredeyse "
                           "sabit kalmis, olcek bu veriden cikarilamaz.")
    ratios = d_ark[ok] / d_col[ok]
    s = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - s)))
    inlier = float((np.abs(ratios - s) / s < 0.05).mean())

    _, R, t = _umeyama(C_col, C_ark)
    resid = np.linalg.norm(C_ark - (s * (C_col @ R.T) + t), axis=1)

    return {
        "s_pose_m_per_unit": s,
        "pose_spread_pct": round(100.0 * mad / s, 3),
        "pose_inlier_ratio": round(inlier, 4),
        "pose_resid_median_mm": round(float(np.median(resid)) * 1000.0, 2),
        "n_frames_paired": n,
    }


def scale_from_depth(rec: pycolmap.Reconstruction, cap: ArkitCapture,
                     fmap: dict[str, int], mask_dir: Path | None = None,
                     conf_min: int = 2, d_min: float = 0.25,
                     d_max: float = 1.50) -> dict | None:
    """LiDAR tabanli olcek (metre / COLMAP birimi). LiDAR yoksa None."""
    if not cap.has_depth:
        return None

    W_rgb, H_rgb = cap.rgb_wh
    ratios: list[np.ndarray] = []
    n_img = 0

    for img in rec.images.values():
        fi = fmap.get(img.name)
        if fi is None:
            continue
        dc = cap.depth_m(fi)
        if dc is None:
            continue
        depth, conf = dc
        dh, dw = depth.shape

        cam = rec.cameras[img.camera_id]
        # COLMAP goruntusu yeniden boyutlanmis olabilir -> LiDAR izgarasina oran
        fx_img, fy_img = dw / float(cam.width), dh / float(cam.height)

        mask = None
        if mask_dir is not None:
            mp = mask_dir / (Path(img.name).stem + ".png")
            if mp.exists():
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    mask = cv2.resize(m, (dw, dh), interpolation=cv2.INTER_NEAREST) > 127

        cfw = _cam_from_world(img)
        uv, dcol = [], []
        for p2 in img.points2D:
            if not p2.has_point3D():
                continue
            X = rec.points3D[p2.point3D_id].xyz
            z = float((cfw.rotation.matrix() @ X + cfw.translation)[2])
            if z <= 0:
                continue
            uv.append(p2.xy)
            dcol.append(z)
        if len(dcol) < 20:
            continue

        uv = np.asarray(uv, np.float64)
        dcol = np.asarray(dcol, np.float64)
        px = np.clip((uv[:, 0] * fx_img).astype(int), 0, dw - 1)
        py = np.clip((uv[:, 1] * fy_img).astype(int), 0, dh - 1)

        dlid = depth[py, px]
        good = (conf[py, px] >= conf_min) & (dlid > d_min) & (dlid < d_max)
        if mask is not None:
            good &= mask[py, px]
        if good.sum() < 20:
            continue

        ratios.append(dlid[good] / dcol[good])
        n_img += 1

    if n_img < 20:
        print(f"[scale] LiDAR capraz kontrolu atlandi — sadece {n_img} karede "
              "yeterli yuksek guvenli derinlik eslesmesi bulundu.")
        return None

    r = np.concatenate(ratios)
    s = float(np.median(r))
    mad = float(np.median(np.abs(r - s)))
    return {
        "s_depth_m_per_unit": s,
        "depth_spread_pct": round(100.0 * mad / s, 3),
        "n_depth_samples": int(r.size),
        "n_depth_frames": n_img,
    }


def compute_scale(case_dir: Path, sparse_model: Path, capture: Path,
                  out_json: Path, mask_dir: Path | None = None,
                  agreement_pct: float = 1.5) -> float:
    """mm / COLMAP birimi dondurur ve scale.json yazar."""
    rec = pycolmap.Reconstruction(str(sparse_model))
    cap = load_capture(capture)
    fmap = _frame_map(case_dir)

    pose = scale_from_poses(rec, cap, fmap)
    depth = scale_from_depth(rec, cap, fmap, mask_dir)

    s_m = pose["s_pose_m_per_unit"]
    result: dict = dict(pose)
    result["scale_source"] = "pose"

    if depth is not None:
        result.update(depth)
        disagree = abs(depth["s_depth_m_per_unit"] - s_m) / s_m * 100.0
        result["agreement_pct"] = round(disagree, 3)
        result["scale_verified"] = bool(disagree <= agreement_pct)
        if not result["scale_verified"]:
            print(f"[scale] UYARI: poz ve LiDAR olcekleri %{disagree:.2f} "
                  f"ayrisiyor (esik %{agreement_pct}). Vaka "
                  "'scale_verified=false' — G1 tablosuna incelemeden girmesin.")
    else:
        result["scale_verified"] = None
        result["agreement_pct"] = None
        print("[scale] LiDAR yok/yetersiz — capraz kontrol yapilamadi. "
              "Guvence sadece tekrarlanabilirlikten (3 cekim) gelecek.")

    scale_mm = s_m * 1000.0
    result["scale_mm_per_unit"] = scale_mm
    out_json.write_text(json.dumps(result, indent=2))

    print(f"[scale] {scale_mm:.6f} mm/birim  "
          f"(poz: {pose['n_frames_paired']} kare, sapma "
          f"%{pose['pose_spread_pct']}, ic nokta "
          f"%{pose['pose_inlier_ratio']*100:.0f})")
    if pose["pose_spread_pct"] > 2.0:
        print("[scale] UYARI: yorunge olcegi %2'den fazla sapiyor — VIO takibi "
              "zayif. Arka planin dokulu ve sabit oldugundan emin ol.")
    return scale_mm


def sanity_check(mesh_mm, landmarks: dict | None = None) -> dict:
    """Olcegi ASLA belirlemez; sadece kaba hatayi yakalar (rapora yazilir)."""
    ext = np.asarray(mesh_mm.bounding_box.extents, dtype=float)
    out = {"bbox_mm": [round(float(v), 1) for v in ext],
           "bbox_plausible": bool(150.0 <= float(ext.max()) <= 320.0)}
    if landmarks and "en_l" in landmarks and "en_r" in landmarks:
        ipd = float(np.linalg.norm(np.asarray(landmarks["en_l"]) -
                                   np.asarray(landmarks["en_r"])))
        out["ipd_mm"] = round(ipd, 1)
        out["ipd_plausible"] = bool(55.0 <= ipd <= 70.0)
    return out
