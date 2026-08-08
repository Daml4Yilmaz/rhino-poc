"""Adim (c2): Gaussian splatting hatti (nerfstudio/gsplat) + yuzey cikarimi.
[HAFTA 2 — STUB]

Plan: COLMAP pozlarini nerfstudio formatina cevir (ns-process-data skip),
splatfacto egit, yuzey cikarimi (orn. 2DGS/SuGaR benzeri) -> mesh.
Hafta 3'te dogruluga gore klasik MVS ile yarisir, biri secilir.

Not: Colab T4 (16 GB) splatfacto icin yeterli ama yavas; egitim adim sayisini
dusuk tut (15-20k) ve Drive'a checkpoint yaz.
"""
from __future__ import annotations

from pathlib import Path


def run_gsplat(frames_dir: Path, sparse_model: Path, out_dir: Path) -> None:
    raise NotImplementedError("Hafta 2: GS hatti henuz yazilmadi.")
