"""Adim (g): 6 rinoplasti olcumu.

Girdi: mm cinsinden 3D landmark sozlugu (hafta 2'de FLAME kaydindan sabit
vertex indeksleriyle uretilecek; simdilik landmarks.json elle/harici de
verilebilir).

Beklenen landmark isimleri (hepsi [x,y,z] mm):
  g    : glabella
  n    : nasion (sellion)
  prn  : pronasale (burun ucu)
  sn   : subnasale
  cm   : columella orta noktasi
  ls   : labiale superius
  al_l : sol alare (burun kanadi en dis nokta)
  al_r : sag alare
  ac   : alar crease (alar-fasiyal oluk, orta)
  en_l : sol endocanthion (ic goz pinari)
  en_r : sag endocanthion
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REQUIRED = ["g", "n", "prn", "sn", "cm", "ls", "al_l", "al_r", "ac", "en_l", "en_r"]


def _angle(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    v1, v2 = a - vertex, b - vertex
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def compute_measurements(landmarks: dict[str, list[float]]) -> dict:
    missing = [k for k in REQUIRED if k not in landmarks]
    if missing:
        raise ValueError(f"Eksik landmark(lar): {missing}")
    L = {k: np.asarray(v, dtype=np.float64) for k, v in landmarks.items()}

    nose_length = float(np.linalg.norm(L["n"] - L["prn"]))
    projection = float(np.linalg.norm(L["ac"] - L["prn"]))

    # Orta hat duzlemi: endocanthion orta noktasi + normal = en_r - en_l
    mid = (L["en_l"] + L["en_r"]) / 2.0
    normal = L["en_r"] - L["en_l"]
    normal /= np.linalg.norm(normal)
    midline_dev = float(abs(np.dot(L["prn"] - mid, normal)))

    return {
        "nasofrontal_angle_deg": round(_angle(L["g"], L["n"], L["prn"]), 1),
        "nasolabial_angle_deg": round(_angle(L["cm"], L["sn"], L["ls"]), 1),
        "goode_ratio": round(projection / nose_length, 3),
        "nose_length_mm": round(nose_length, 2),
        "nose_width_mm": round(float(np.linalg.norm(L["al_l"] - L["al_r"])), 2),
        "midline_deviation_mm": round(midline_dev, 2),
    }


def run_measure(landmarks_json: Path, out_json: Path) -> dict:
    landmarks = json.loads(landmarks_json.read_text())
    m = compute_measurements(landmarks)
    out_json.write_text(json.dumps(m, indent=2))
    print(f"[measure] {out_json}:")
    for k, v in m.items():
        print(f"  {k}: {v}")
    return m
