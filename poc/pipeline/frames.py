"""Adim (a): videodan kare cikarma ("ekran goruntusu") + bulanik kare eleme.

Fotogrametrinin girdisi bu karelerdir: video -> kareler -> COLMAP SfM + MVS.

Strateji: videoyu bastan sona bir kez cozup her karenin Laplacian varyansini
olcer, zaman eksenini pencerelere bolup her pencereden EN KESKIN kareyi
secer. Boylece hem bulanik kareler elenir hem de kafa etrafindaki acisal
kapsama korunur (sadece "en keskin 300" alinsa hepsi ayni acidan gelebilirdi).

ffmpeg yerine OpenCV ile cozuyoruz — cunku her secilen karenin KAYNAK KARE
INDEKSI'ni bilmek zorundayiz: olcek adimi o indeksle ARKit pozunu bulur
(frames_index.json). ffmpeg'in `fps=` filtresi kareleri yeniden ornekler ve
bu esleşmeyi sessizce bozar.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _decode_scores(video: Path) -> np.ndarray:
    """Her karenin keskinlik skoru (Laplacian varyansi). Tek gecis."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {video}")
    scores = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Skoru kucuk kopyada olc: sonuc siralamasi ayni, cozme suresi degil.
        g = cv2.resize(g, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        scores.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    cap.release()
    if not scores:
        raise RuntimeError(f"Video'dan hic kare okunamadi: {video}")
    return np.asarray(scores)


def _pick(scores: np.ndarray, n_frames: int, blur_min_var: float) -> list[int]:
    idx: list[int] = []
    n_windows = min(n_frames, len(scores))
    bounds = np.linspace(0, len(scores), n_windows + 1, dtype=int)
    for i in range(n_windows):
        lo, hi = bounds[i], bounds[i + 1]
        if lo >= hi:
            continue
        best = lo + int(np.argmax(scores[lo:hi]))
        if scores[best] < blur_min_var:
            continue  # pencerenin tamami bulanik -> atla
        idx.append(best)
    return idx


def _write(video: Path, frames_dir: Path, wanted: list[int],
           max_dim: int | None) -> list[str]:
    """Secilen kareleri diske yazar; ikinci sirali gecis (seek guvenilmez)."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    want = set(wanted)
    cap = cv2.VideoCapture(str(video))
    names: list[str] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            if max_dim:
                h, w = frame.shape[:2]
                if max(h, w) > max_dim:
                    s = max_dim / float(max(h, w))
                    frame = cv2.resize(frame, (round(w * s), round(h * s)),
                                       interpolation=cv2.INTER_AREA)
            name = f"f{i:06d}.jpg"
            cv2.imwrite(str(frames_dir / name), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            names.append(name)
        i += 1
    cap.release()
    return names


def select_frames(video: Path, frames_dir: Path, n_frames: int = 300,
                  blur_min_var: float = 40.0, max_dim: int | None = 1600,
                  index_json: Path | None = None) -> list[str]:
    if not Path(video).exists():
        raise FileNotFoundError(
            f"Video bulunamadi: {video} — yolu ve uzantiyi (mp4/MOV) kontrol et.")

    scores = _decode_scores(Path(video))
    wanted = _pick(scores, n_frames, blur_min_var)
    if len(wanted) < 60:
        raise RuntimeError(
            f"Sadece {len(wanted)} keskin kare bulundu (<60). Cekim cok "
            "bulanik — AE/AF kilidi, daha yavas ve duzgun bir yay, daha iyi "
            "isik ile tekrar cek.")

    names = _write(Path(video), frames_dir, wanted, max_dim)

    if index_json is not None:
        index_json.parent.mkdir(parents=True, exist_ok=True)
        index_json.write_text(json.dumps({
            "video": str(video),
            "n_source_frames": int(len(scores)),
            "max_dim": max_dim,
            # Olcek adimi bunu kullanir: goruntu adi -> kaynak kare indeksi
            "frame_of_image": {n: int(n[1:-4]) for n in names},
        }, indent=2))

    print(f"[frames] {len(scores)} kaynak kare -> {len(names)} secildi "
          f"(medyan keskinlik {np.median(scores):.0f}, "
          f"esik {blur_min_var:.0f})")
    return names
