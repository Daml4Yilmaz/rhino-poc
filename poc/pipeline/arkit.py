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
    K: np.ndarray                # (3,3) ortalama, RGB cozunurlugunde
    rgb_wh: tuple[int, int]      # (W,H)
    rgb_path: Path
    depth_dir: Path | None       # None -> LiDAR yok
    conf_dir: Path | None
    source: str                  # "stray" | "record3d"
    # (N,4) fx,fy,cx,cy — kare basina. Otofokus kilitlenemedigi zaman
    # kritik: COLMAP'e tek bir odak uzakligi tahmin ettirmek yerine her
    # karenin OLCULMUS degerini veriyoruz (bkz. sfm.write_arkit_intrinsics).
    intrinsics: np.ndarray | None = None

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

    rows, intr = [], []
    with open(odo) as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames:
            raise RuntimeError(f"odometry.csv basligi okunamadi: {odo}")
        # Stray Scanner basligi virgulden SONRA bosluk birakir
        # ("timestamp, frame, x, ..."), yani ham anahtarlar " x" olur.
        rd.fieldnames = [c.strip() for c in rd.fieldnames]
        for r in rd:
            rows.append([float(r["x"]), float(r["y"]), float(r["z"]),
                         float(r["qx"]), float(r["qy"]), float(r["qz"]),
                         float(r["qw"])])
            if r.get("fx"):
                intr.append([float(r["fx"]), float(r["fy"]),
                             float(r["cx"]), float(r["cy"])])
    if not rows:
        raise RuntimeError(f"odometry.csv bos: {odo}")
    a = np.asarray(rows, dtype=np.float64)

    # Ic parametreler: odometry.csv kare basina verir ve bu camera_matrix.csv'den
    # daha guvenilirdir. Ayrica fx'in kare boyunca SABIT olup olmadigini
    # gorebiliriz — oynuyorsa otofokus kilitli degildir ve COLMAP'in tek-kamera
    # varsayimi bozulur.
    P = None
    if intr:
        P = np.asarray(intr, dtype=np.float64)
        fx, fy, cx, cy = P.mean(axis=0)
        drift = float((P[:, 0].max() - P[:, 0].min()) / P[:, 0].mean() * 100.0)
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        if drift > 1.0:
            print(f"[arkit] Odak uzakligi cekim boyunca %{drift:.1f} oynamis "
                  "(AF kilitli degil) — kare basina olculmus ic parametreler "
                  "COLMAP'e verilecek, tek-kamera varsayimi kullanilmayacak.")
    else:
        K = np.loadtxt(kmat, delimiter=",").reshape(3, 3)

    depth_dir = root / "depth" if (root / "depth").is_dir() else None
    conf_dir = root / "confidence" if (root / "confidence").is_dir() else None
    if depth_dir is not None:
        n_depth = len(list(depth_dir.glob("*.png")))
        if n_depth < len(a) * 0.5:
            print(f"[arkit] UYARI: {len(a)} poz var ama sadece {n_depth} "
                  "derinlik karesi — LiDAR capraz kontrolu zayif olacak.")

    return ArkitCapture(centers=a[:, 0:3], quats=a[:, 3:7], K=K,
                        rgb_wh=_rgb_size(rgb), rgb_path=rgb,
                        depth_dir=depth_dir, conf_dir=conf_dir, source="stray",
                        intrinsics=P)


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


def _view_dirs(quats: np.ndarray) -> np.ndarray:
    """Kare basina bakis yonu (ARKit kamera ekseninde -Z ileri)."""
    x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    n = x * x + y * y + z * z + w * w
    s = 2.0 / np.where(n < 1e-12, 1.0, n)
    # Donme matrisinin 3. sutunu; ileri yon onun negatifi.
    f = np.stack([-(s * (x * z + y * w)),
                  -(s * (y * z - x * w)),
                  -(1 - s * (x * x + y * y))], axis=1)
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def angular_coverage_deg(cap: ArkitCapture) -> float:
    """Bakis yonlerinin tarandigi en genis aci. SfM icin ASIL olcut budur."""
    f = _view_dirs(cap.quats)
    G = np.clip(f @ f.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(G)).max())


def validate(cap: ArkitCapture) -> None:
    """Yakalamayi ise baslamadan ele.

    Ayrim onemli: cozulemez olan seyler HATA, protokolun altinda kalan
    seyler UYARI. Yol uzunlugu tek basina olcut degil — bir yuzun etrafinda
    72 derecelik kisa bir yay, iki metrelik duz bir kaydirmadan cok daha
    fazla bilgi tasir. Belirleyici olan ACISAL KAPSAMA.
    """
    n = cap.n_frames
    if n < 120:
        raise RuntimeError(
            f"Sadece {n} kare poz var (<120). Cekim cok kisa, tekrarla.")

    step = np.linalg.norm(np.diff(cap.centers, axis=0), axis=1)
    jumps = int((step > 0.15).sum())
    if jumps > n * 0.02:
        raise RuntimeError(
            f"{jumps} karede ani poz sicramasi (>15 cm) — ARKit takibi kopmus. "
            "Dokulu ve sabit bir arka plan onunde tekrar cek.")

    path_m = float(step.sum())
    cover = angular_coverage_deg(cap)

    # HATA: paralaks yoksa problem matematiksel olarak cozumsuz.
    if cover < 20.0:
        raise RuntimeError(
            f"Bakis acisi cekim boyunca sadece {cover:.0f} derece degismis "
            f"(yol {path_m:.2f} m). Kamera denegin ETRAFINDA donmemis — "
            "paralaks yok, SfM bunu cozemez. Telefonu cevirmek yetmez, "
            "konumu degismeli.")

    print(f"[arkit] {cap.source}: {n} kare, acisal kapsama {cover:.0f} derece, "
          f"yol {path_m:.2f} m, RGB {cap.rgb_wh[0]}x{cap.rgb_wh[1]}, "
          f"LiDAR {'var' if cap.has_depth else 'YOK'}")

    # UYARI: kosar ama protokolun altinda; sonuc G1 kalitesinde olmaz.
    if cover < 120.0:
        print(f"[arkit] UYARI: {cover:.0f} derece kapsama — protokol kulaktan "
              "kulaga ~180 derece ister. Yuzun yan bolgeleri eksik kalacak.")
    if n < 700:
        print(f"[arkit] UYARI: {n} kare (~{n/30:.0f} sn) — protokol iki gecis "
              "icin ~40-60 sn ister. Burun tabani gecisi yapilmamis olabilir.")
    if not cap.has_depth:
        print("[arkit] UYARI: derinlik yok — olcek capraz kontrolu (s_depth) "
              "yapilamayacak, sadece poz tabanli olcek kalir.")
