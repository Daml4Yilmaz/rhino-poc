"""Adim (a): ffmpeg ile kare cikarma + Laplacian varyansiyla bulanik kare eleme.

Strateji: videoyu sabit fps ile ham karelere ac, her karenin Laplacian
varyansini hesapla, zaman eksenini pencerelere bolup her pencereden en keskin
kareyi sec. Boylece hem bulanik kareler elenir hem de kafa etrafindaki acisal
kapsama korunur (sadece "en keskin 200" alinsa hepsi ayni acidan gelebilirdi).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def extract_raw_frames(video: Path, raw_dir: Path, fps: int) -> list[Path]:
    if not video.exists():
        raise FileNotFoundError(
            f"Video bulunamadi: {video} — yolu ve uzantiyi (mp4/MOV) kontrol et.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"fps={fps}",
        "-qscale:v", "2",
        str(raw_dir / "f%05d.jpg"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.splitlines()[-10:])
        raise RuntimeError(f"ffmpeg hata verdi (kod {r.returncode}):\n{tail}")
    return sorted(raw_dir.glob("f*.jpg"))


def laplacian_var(img_path: Path) -> float:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def select_frames(video: Path, frames_dir: Path, fps: int = 15,
                  n_frames: int = 200, blur_min_var: float = 40.0) -> list[Path]:
    raw_dir = frames_dir.parent / "_raw_frames"
    raw = extract_raw_frames(video, raw_dir, fps)
    if not raw:
        raise RuntimeError(f"Video'dan kare cikarilamadi: {video}")

    scores = np.array([laplacian_var(p) for p in raw])

    frames_dir.mkdir(parents=True, exist_ok=True)
    selected: list[Path] = []
    n_windows = min(n_frames, len(raw))
    bounds = np.linspace(0, len(raw), n_windows + 1, dtype=int)
    for i in range(n_windows):
        lo, hi = bounds[i], bounds[i + 1]
        if lo >= hi:
            continue
        best = lo + int(np.argmax(scores[lo:hi]))
        if scores[best] < blur_min_var:
            continue  # pencerenin tamami bulanik -> atla
        dst = frames_dir / raw[best].name
        shutil.copy2(raw[best], dst)
        selected.append(dst)

    shutil.rmtree(raw_dir, ignore_errors=True)
    if len(selected) < 30:
        raise RuntimeError(
            f"Sadece {len(selected)} keskin kare bulundu (<30). "
            "Cekim cok bulanik — AE/AF kilidi ve daha yavas tur ile tekrar cek.")
    print(f"[frames] {len(raw)} ham kare -> {len(selected)} secildi "
          f"(medyan keskinlik {np.median(scores):.0f})")
    return selected
