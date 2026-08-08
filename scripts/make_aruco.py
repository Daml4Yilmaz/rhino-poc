"""Bilinen boyutta basilabilir ArUco marker uretir.

Kullanim:
    python scripts/make_aruco.py --mm 50 --out aruco_50mm.png

Cikti 300 DPI PNG'dir; "Gercek boyut / %100 olcek" ile MAT kagida bas.
BASTIKTAN SONRA kenari cetvelle olc — pipeline'a o degeri ver
(--marker-mm), nominali degil. Yazicilar %1-3 olcekle oynayabilir ve bu
dogrudan olcum hatasina girer.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
from PIL import Image

DPI = 300


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", type=float, default=50.0, help="kenar uzunlugu (mm)")
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--out", default="aruco_50mm.png")
    args = ap.parse_args()

    px = int(round(args.mm / 25.4 * DPI))
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    marker = cv2.aruco.generateImageMarker(adict, args.id, px)

    # Beyaz kenarlik (quiet zone) + altina 100 mm'lik kontrol cetveli
    border = px // 5
    ruler_h = int(10 / 25.4 * DPI)
    W = max(px + 2 * border, int(100 / 25.4 * DPI) + 2 * border)
    H = px + 2 * border + ruler_h * 2
    canvas = np.full((H, W), 255, np.uint8)
    x0 = (W - px) // 2
    canvas[border:border + px, x0:x0 + px] = marker

    # 100 mm cizgi (baski olceginin dogrulanmasi icin)
    y = px + 2 * border + ruler_h // 2
    lx = (W - int(100 / 25.4 * DPI)) // 2
    canvas[y:y + 3, lx:lx + int(100 / 25.4 * DPI)] = 0
    cv2.putText(canvas, "bu cizgi tam 100 mm olmali", (lx, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    Image.fromarray(canvas).save(args.out, dpi=(DPI, DPI))
    print(f"{args.out} yazildi ({args.mm} mm, id={args.id}, {args.dict}, {DPI} DPI)")
    print("Yazdirirken 'Gercek boyut/%100' sec, MAT kagit kullan, "
          "sonra kenari cetvelle dogrula.")


if __name__ == "__main__":
    main()
