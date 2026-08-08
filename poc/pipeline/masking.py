"""Adim (d): saç/fon maskeleme + Open3D temizlik.  [HAFTA 2 — STUB]

Plan: yuz ayristirma (face parsing, orn. BiSeNet/MediaPipe selfie seg.) ile
kare bazinda maske; maskeler MVS'e girmeden once uygulanir veya fused.ply
uzerinde geri-projeksiyon ile filtrelenir. Ardindan Open3D statistical
outlier removal + en buyuk bagli bilesen.
"""
from __future__ import annotations

from pathlib import Path


def run_masking(frames_dir: Path, out_dir: Path) -> None:
    raise NotImplementedError("Hafta 2: face parsing maskeleme henuz yazilmadi.")
