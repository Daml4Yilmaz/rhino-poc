"""ARKit yakalama verisi (Stray Scanner / Record3D) okuma.

BU MODUL 3D MODELI URETMEZ. Model tamamen fotogrametriyle uretilir:
video -> kare cikarma (frames.py) -> COLMAP SfM + MVS (sfm.py, mvs.py).

Buradaki verinin tek isi CETVEL saglamak. Fotogrametri birimsiz bir model
verir; ayni kareler 50 mm'lik burunla da 500 mm'lik burunla da tutarlidir.
Mutlak boyut goruntulerin icinde yoktur, disaridan gelmek zorundadir.
Marker kullanmadigimiz icin bu bilgi telefonun kendi metrik takibinden
(ARKit VIO + LiDAR) gelir; sadece "1 COLMAP birimi kac mm" carpanini
uretmek icin kullanilir (bkz. scale.py).

Stray Scanner disa aktarimi (LiDAR sart):
  rgb.mp4              1920x1440 H.264, 30 fps
  depth/000000.png     256x192, 16-bit, MILIMETRE
  confidence/*.png     256x192, 8-bit, 0/1/2 (2 = yuksek guven)
  odometry.csv         timestamp, frame, x, y, z, qx, qy, qz, qw
  camera_matrix.csv    3x3 K (RGB cozunurlugunde)

Record3D (.r3d): zip; icindeki `metadata` JSON'da poses
[qx,qy,qz,qw,tx,ty,tz] ve K bulunur. LiDAR yoksa da poz verir.

ONEMLI — koordinat sistemi: olcek tahmini icin sadece kamera MERKEZLERI
kullanilir. ARKit kamera ekseni OpenGL (-Z ileri), COLMAP OpenCV (+Z ileri);
merkezler uzerinden calisinca bu fark onemsizlesir, Umeyama iki dunya
cercevesi arasindaki donusumu zaten cozer.
"""
from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ArkitCapture:
    """Normalize edilmis yakalama. Kare indeksleri rgb videosuyla 1:1."""
    centers: np.ndarray          # (N,3) metre, dunya cercevesinde kamera merkezi
    quats: np.ndarray            # (N,4) xyzw, dunya<-kamera
    K: np.ndarray                # (3,3) RGB cozunurlugunde
    rgb_wh: tuple[int, int]      # (W,H)
    rgb_path: Path
    depth_dir: Path | None       # None -> LiDAR yok
    conf_dir: Path | None
    source: str                  # "stray" | "record3d"

    @property
    def n_frames(self) -> int:
        return len(self.centers)

    @property
    def has_depth(self) -> bool:
        return self.depth_dir is not None

    def depth_m(self, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
        """(derinlik_metre, guven) — LiDAR yoksa veya kare yoksa None."""
        if self.depth_dir is None:
            return None
        dp = self.depth_dir / f"{frame:06d}.png"
        if not dp.exists():
            return None
        d = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        depth = d.astype(np.float32) / 1000.0          # mm -> m
        if self.conf_dir is not None:
            cp = self.conf_dir / f"{frame:06d}.png"
            conf = cv2.imread(str(cp), cv2.IMREAD_UNCHANGED)
            conf = np.zeros(d.shape, np.uint8) if conf is None else conf
        else:
            conf = np.full(d.shape, 2, np.uint8)
        return depth, conf


def _rgb_size(video: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def load_stray(root: Path) -> ArkitCapture:
    odo, kmat, rgb = root / "odometry.csv", root / "camera_matrix.csv", root / "rgb.mp4"
    for p in (odo, kmat, rgb):
        if not p.exists():
            raise FileNotFoundError(
                f"Stray Scanner disa aktariminda {p.name} yok: {root}\n"
                "Uygulamadan 'Export' ile TUM klasoru aktardigindan emin ol.")

    rows = []
    with open(odo) as f:
        for r in csv.DictReader(f):
            rows.append([float(r["x"]), float(r["y"]), float(r["z"]),
                         float(r["qx"]), float(r["qy"]), float(r["qz"]),
                         float(r["qw"])])
    if not rows:
        raise RuntimeError(f"odometry.csv bos: {odo}")
    a = np.asarray(rows, dtype=np.float64)

    K = np.loadtxt(kmat, delimiter=",").reshape(3, 3)
    depth_dir = root / "depth" if (root / "depth").is_dir() else None
    conf_dir = root / "confidence" if (root / "confidence").is_dir() else None

    return ArkitCapture(centers=a[:, 0:3], quats=a[:, 3:7], K=K,
                        rgb_wh=_rgb_size(rgb), rgb_path=rgb,
                        depth_dir=depth_dir, conf_dir=conf_dir, source="stray")


def load_record3d(path: Path) -> ArkitCapture:
    """.r3d (zip) veya acilmis klasor. Poz + K okur; LiDAR derinligi okunmaz."""
    if path.is_file():
        work = path.parent / (path.stem + "_r3d")
        if not work.exists():
            with zipfile.ZipFile(path) as z:
                z.extractall(work)
        root = work
    else:
        root = path

    meta_p = next((p for p in (root / "metadata", root / "metadata.json")
                   if p.exists()), None)
    if meta_p is None:
        raise FileNotFoundError(f"Record3D metadata bulunamadi: {root}")
    meta = json.loads(meta_p.read_text())

    poses = np.asarray(meta["poses"], dtype=np.float64)   # (N,7) qx qy qz qw tx ty tz
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise RuntimeError(f"Beklenmeyen Record3D poz bicimi: {poses.shape}")
    K = np.asarray(meta["K"], dtype=np.float64).reshape(3, 3).T  # column-major

    rgb = next((p for p in (root / "rgbd.mp4", root / "rgb.mp4") if p.exists()), None)
    if rgb is None:
        raise FileNotFoundError(f"Record3D video bulunamadi: {root}")

    return ArkitCapture(centers=poses[:, 4:7], quats=poses[:, 0:4], K=K,
                        rgb_wh=_rgb_size(rgb), rgb_path=rgb,
                        depth_dir=None, conf_dir=None, source="record3d")


def load_capture(path: Path) -> ArkitCapture:
    """Stray Scanner klasoru veya Record3D .r3d — bicimi kendi anlar."""
    path = Path(path)
    if path.is_dir() and (path / "odometry.csv").exists():
        cap = load_stray(path)
    elif path.suffix.lower() == ".r3d" or (path / "metadata").exists():
        cap = load_record3d(path)
    else:
        raise ValueError(
            f"Taninmayan yakalama bicimi: {path}\n"
            "Beklenen: Stray Scanner klasoru (odometry.csv) veya Record3D .r3d")
    validate(cap)
    return cap


def validate(cap: ArkitCapture) -> None:
    """Yakalamayi ise baslamadan ele: sessiz hatalar burada yakalanir."""
    if cap.n_frames < 300:
        raise RuntimeError(
            f"Sadece {cap.n_frames} kare poz var (<300). 30 fps'te bu ~10 sn "
            "eder — iki gecis icin cok kisa. Cekimi tekrarla.")

    # VIO sicramasi: ardisik kareler arasi mesafe 30 fps'te ~1-3 cm olmali.
    step = np.linalg.norm(np.diff(cap.centers, axis=0), axis=1)
    jumps = int((step > 0.15).sum())
    if jumps > cap.n_frames * 0.02:
        raise RuntimeError(
            f"{jumps} karede ani poz sicramasi (>15 cm) — ARKit takibi kopmus. "
            "Dokulu ve sabit bir arka plan onunde tekrar cek.")

    # Yol uzunlugu: kulaktan kulaga iki gecis en az ~1.5 m yol demek. Kisa
    # yol = zayif paralaks = Umeyama olcegi kotu kosullu.
    path_m = float(step.sum())
    if path_m < 1.0:
        raise RuntimeError(
            f"Kamera toplam {path_m:.2f} m yol almis (<1.0 m). Iki kisilik "
            "protokolde beklenen >1.5 m — filmci yeterince genis yay cizmemis.")

    print(f"[arkit] {cap.source}: {cap.n_frames} kare, yol {path_m:.2f} m, "
          f"RGB {cap.rgb_wh[0]}x{cap.rgb_wh[1]}, "
          f"LiDAR {'var' if cap.has_depth else 'YOK'}")
    if not cap.has_depth:
        print("[arkit] UYARI: derinlik yok — olcek capraz kontrolu (s_depth) "
              "yapilamayacak, sadece poz tabanli olcek kalir.")
