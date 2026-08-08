"""Adim (f): FLAME sablonunun taramaya kaydi.  [HAFTA 2 — STUB]

Plan:
1. MediaPipe Face Landmarker ile karelerden 2D landmark -> COLMAP pozlariyla
   ucgenleyerek seyrek 3D landmark seti.
2. FLAME'i rigid + olcek hizala (Umeyama), sonra shape/expression optimize,
   son olarak non-rigid ICP ile yuzeye otur.
3. Anatomik noktalar (measure.py'nin bekledigi g, n, prn, sn, cm, ls, al_*,
   ac, en_*) FLAME'in sabit vertex indekslerinden okunur -> landmarks.json.

FLAME indirimi kayit gerektirir: https://flame.is.tue.mpg.de (hesabi erken ac).
"""
from __future__ import annotations

from pathlib import Path


def run_flame_fit(mesh_ply: Path, frames_dir: Path, sparse_model: Path,
                  out_landmarks_json: Path) -> None:
    raise NotImplementedError("Hafta 2: FLAME kaydi henuz yazilmadi.")
