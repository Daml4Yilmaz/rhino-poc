"""Adim (e): ArUco kose ucgenlemesiyle mutlak olcek.

COLMAP rekonstruksiyonu keyfi birimdedir. Alindaki bilinen boyutlu ArUco
marker'inin 4 kosesini karelerde tespit edip COLMAP pozlariyla 3D'ye
ucgenler, kenar uzunlugunun kac "COLMAP birimi" ettigini olcer ve
scale_mm = marker_mm / kenar_birim faktorunu uretiriz.

Not: Kose pikselleri once kamera distorsiyonundan arindirilir (OPENCV
modeli), ucgenleme normalize koordinatlarda DLT ile yapilir.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap


def _detect_corners(frames_dir: Path, dict_name: str, marker_id: int
                    ) -> dict[str, np.ndarray]:
    """kare adi -> (4,2) kose pikselleri (tl,tr,br,bl sirali)."""
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    detector = cv2.aruco.ArucoDetector(adict, cv2.aruco.DetectorParameters())
    out: dict[str, np.ndarray] = {}
    for p in sorted(frames_dir.glob("*.jpg")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        corners, ids, _ = detector.detectMarkers(img)
        if ids is None:
            continue
        for c, i in zip(corners, ids.flatten()):
            if int(i) == marker_id:
                out[p.name] = c.reshape(4, 2).astype(np.float64)
    return out


def _camera_K_dist(cam) -> tuple[np.ndarray, np.ndarray]:
    """pycolmap OPENCV kamerasindan K ve distorsiyon katsayilari."""
    fx, fy, cx, cy, k1, k2, p1, p2 = list(cam.params)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array([k1, k2, p1, p2])
    return K, dist


def _triangulate(obs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """obs: [(P_3x4, xy_normalized), ...] -> 3D nokta (DLT, en kucuk kareler)."""
    A = []
    for P, xy in obs:
        x, y = xy
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.stack(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def compute_scale(frames_dir: Path, sparse_model: Path, out_json: Path,
                  marker_mm: float = 50.0, dict_name: str = "DICT_4X4_50",
                  marker_id: int = 0, min_views: int = 5) -> float:
    rec = pycolmap.Reconstruction(str(sparse_model))
    detections = _detect_corners(frames_dir, dict_name, marker_id)

    name_to_img = {img.name: img for img in rec.images.values()}
    corners3d = []
    n_views_used = 0
    for ci in range(4):
        obs = []
        for name, corners in detections.items():
            img = name_to_img.get(name)
            if img is None:
                continue
            cam = rec.cameras[img.camera_id]
            K, dist = _camera_K_dist(cam)
            und = cv2.undistortPoints(
                corners[ci].reshape(1, 1, 2), K, dist).reshape(2)
            # cam_from_world -> P = [R|t] (normalize koordinatlarda K=I)
            cfw = img.cam_from_world.matrix()  # 3x4
            obs.append((cfw, und))
        if len(obs) < min_views:
            raise RuntimeError(
                f"ArUco kose {ci} sadece {len(obs)} karede kayitli goruldu "
                f"(<{min_views}). Marker gorunurlugunu/isigi kontrol et.")
        n_views_used = max(n_views_used, len(obs))
        corners3d.append(_triangulate(obs))

    c = np.stack(corners3d)
    sides = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
    diags = [np.linalg.norm(c[0] - c[2]), np.linalg.norm(c[1] - c[3])]
    side_units = float(np.mean(sides))
    scale = marker_mm / side_units

    # Tutarlilik kontrolleri
    side_spread = float((max(sides) - min(sides)) / side_units)
    diag_ratio = float(diags[0] / diags[1])
    result = {
        "marker_mm": marker_mm,
        "side_units_mean": side_units,
        "scale_mm_per_unit": scale,
        "side_spread_pct": round(side_spread * 100, 2),
        "diag_ratio": round(diag_ratio, 4),
        "n_frames_with_marker": len(detections),
        "n_views_best_corner": n_views_used,
    }
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[scale] {scale:.6f} mm/birim  (marker {len(detections)} karede; "
          f"kenar sapmasi %{side_spread*100:.1f})")
    if side_spread > 0.05:
        print("[scale] UYARI: kenar uzunluklari %5'ten fazla sapiyor — "
              "ucgenleme zayif olabilir, cekimi/marker yapistirmayi kontrol et.")
    return scale
