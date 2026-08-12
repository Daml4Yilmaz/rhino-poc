"""Adim (b): COLMAP ile kamera pozlari (SfM).

COLMAP binary'sini subprocess ile cagiririz (pycolmap sadece model okumak
icin). Video karelerinde sirali eslestirme (sequential matching) hem hizli
hem yeterli; loop detection kulaktan kulaga turun iki ucunu baglar.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

PINHOLE = 1   # COLMAP kamera modeli: params = [fx, fy, cx, cy]


def _count_by(it) -> dict:
    d: dict = {}
    for v in it:
        d[v] = d.get(v, 0) + 1
    return d


def write_arkit_intrinsics(db_path: Path, frames_dir: Path, cap,
                           frame_of_image: dict[str, int]) -> int:
    """Her goruntuye ARKit'in O KARE icin bildirdigi ic parametreleri ata.

    Neden: Stray Scanner otofokusu kilitlemeye izin vermiyor, yani odak
    uzakligi cekim boyunca oynuyor (test kaydinda %3.3). COLMAP'e tek bir
    ortak odak uzakligi tahmin ettirmek bu degisimi geometriye yayar. Ama
    ARKit her karenin fx/fy/cx/cy'sini ZATEN olcup odometry.csv'ye yaziyor —
    tahmin ettirmek yerine olculmus degeri veriyoruz. Bu, AF kilidinden daha
    iyidir: kilit sabitligi varsayar, bu ise gercek degeri kullanir.

    Kareler kucultulmusse (max_dim) ic parametreler ayni oranda olceklenir.
    """
    if cap.intrinsics is None:
        return 0

    import sqlite3
    import cv2

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    rows = cur.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    if not rows:
        con.close()
        return 0

    # Kameralari YERINDE guncelliyoruz, yenisini ekleyip eskisini SILMIYORUZ:
    # COLMAP 4.x'te `rigs` tablosu kamera id'lerine referans verir, kamera
    # degistirilince rig'ler bosa duser ve mapper "Camera N from rig N not
    # found" ile cokerdi. Yerinde guncelleme rigs/frames tablolarina hic
    # dokunmadigi icin surumden bagimsiz calisir.
    shared = [c for c, k in _count_by(r[2] for r in rows).items() if k > 1]
    if shared:
        con.close()
        raise RuntimeError(
            "Goruntuler kamera paylasiyor — kare basina ic parametre "
            "yazilamaz. feature_extractor '--ImageReader.single_camera 0' "
            "ile kosmali.")

    probe = cv2.imread(str(frames_dir / rows[0][1]))
    if probe is None:
        con.close()
        raise RuntimeError(f"Kare okunamadi: {frames_dir / rows[0][1]}")
    H_img, W_img = probe.shape[:2]
    sx = W_img / float(cap.rgb_wh[0])
    sy = H_img / float(cap.rgb_wh[1])

    n = 0
    for _image_id, name, camera_id in rows:
        fi = frame_of_image.get(name)
        if fi is None or fi >= len(cap.intrinsics):
            continue
        fx, fy, cx, cy = cap.intrinsics[fi]
        params = np.array([fx * sx, fy * sy, cx * sx, cy * sy], dtype=np.float64)
        cur.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, "
            "prior_focal_length=1 WHERE camera_id=?",
            (PINHOLE, W_img, H_img, params.tobytes(), camera_id))
        n += 1

    con.commit()
    con.close()
    print(f"[sfm] {n} goruntuye ARKit ic parametreleri yazildi "
          f"(olcek {sx:.3f}x); bundle adjustment odak uzakligini SABIT tutacak")
    return n


_GPU_OPT_CACHE: dict[str, tuple[str, str]] = {}


def _gpu_opts(colmap_bin: str) -> tuple[str, str]:
    """(cikarim, eslestirme) GPU secenek adlari — COLMAP surumune gore.

    COLMAP 4.x bunlari yeniden adlandirdi:
      3.x: --SiftExtraction.use_gpu   / --SiftMatching.use_gpu
      4.x: --FeatureExtraction.use_gpu / --FeatureMatching.use_gpu
    Yanlisini vermek 'unrecognised option' ile calismayi bastan kirar; hangi
    surumun kurulu oldugunu bilemeyecegimiz icin (yerelde brew, Colab'da
    conda-forge) yardim ciktisindan tespit ediyoruz.
    """
    if colmap_bin in _GPU_OPT_CACHE:
        return _GPU_OPT_CACHE[colmap_bin]
    try:
        h = subprocess.run([colmap_bin, "feature_extractor", "-h"],
                           capture_output=True, text=True, timeout=60)
        new = "FeatureExtraction.use_gpu" in (h.stdout + h.stderr)
    except (OSError, subprocess.SubprocessError):
        new = False
    opts = (("--FeatureExtraction.use_gpu", "--FeatureMatching.use_gpu") if new
            else ("--SiftExtraction.use_gpu", "--SiftMatching.use_gpu"))
    _GPU_OPT_CACHE[colmap_bin] = opts
    return opts


def _run(cmd: list[str], log_file: Path) -> None:
    with open(log_file, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def run_sfm(frames_dir: Path, colmap_dir: Path, colmap_bin: str = "colmap",
            camera_model: str = "OPENCV", seq_overlap: int = 15,
            use_gpu: bool = True, matcher: str = "sequential",
            capture=None, frame_of_image: dict[str, int] | None = None) -> Path:
    """Sparse rekonstruksiyon uretir; en buyuk alt-modelin yolunu dondurur.

    `capture` (ArkitCapture) ve `frame_of_image` verilirse odak uzakligi
    tahmin ETTIRILMEZ: her goruntuye ARKit'in o kare icin olctugu ic
    parametreler yazilir ve bundle adjustment onlari sabit tutar.
    """
    colmap_dir.mkdir(parents=True, exist_ok=True)
    db = colmap_dir / "database.db"
    sparse = colmap_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    log = colmap_dir / "colmap.log"
    gpu = "1" if use_gpu else "0"
    ext_gpu, match_gpu = _gpu_opts(colmap_bin)

    use_arkit_K = (capture is not None and frame_of_image
                   and getattr(capture, "intrinsics", None) is not None)

    _run([colmap_bin, "feature_extractor",
          "--database_path", str(db),
          "--image_path", str(frames_dir),
          # ARKit ic parametreleri gelecekse tek-kamera varsaymanin anlami yok:
          # her goruntu kendi olculmus kamerasini alacak.
          "--ImageReader.single_camera", "0" if use_arkit_K else "1",
          "--ImageReader.camera_model",
          "PINHOLE" if use_arkit_K else camera_model,
          ext_gpu, gpu], log)

    if use_arkit_K:
        write_arkit_intrinsics(db, frames_dir, capture, frame_of_image)

    if matcher == "exhaustive":
        # Birlestirilmis (concat) videolar icin: klip sinirindaki siçrama
        # sequential eslestirmeyi bolebilir; exhaustive herkesle eslestirir.
        # ~200 karede T4'te ~10-20 dk.
        _run([colmap_bin, "exhaustive_matcher",
              "--database_path", str(db),
              match_gpu, gpu], log)
    else:
        _run([colmap_bin, "sequential_matcher",
              "--database_path", str(db),
              "--SequentialMatching.overlap", str(seq_overlap),
              "--SequentialMatching.loop_detection", "0",
              match_gpu, gpu], log)

    mapper = [colmap_bin, "mapper",
              "--database_path", str(db),
              "--image_path", str(frames_dir),
              "--output_path", str(sparse)]
    if use_arkit_K:
        # Olculmus degerleri yeniden optimize etme — yoksa BA onlari
        # tekrar serbest birakir ve ARKit'i vermenin anlami kalmaz.
        mapper += ["--Mapper.ba_refine_focal_length", "0",
                   "--Mapper.ba_refine_principal_point", "0",
                   "--Mapper.ba_refine_extra_params", "0"]
    _run(mapper, log)

    import pycolmap

    # COLMAP birden fazla ALT-MODEL uretebilir (sparse/0, sparse/1, ...).
    # Sahne parcalanirsa 0 numarali model kucuk bir parca olabilir; her zaman
    # "0"i almak sessizce birkac karelik bir enkazla devam etmek demektir.
    # En cok kare kaydedileni sec.
    cand = sorted(p for p in sparse.iterdir()
                  if (p / "cameras.bin").exists() or (p / "cameras.txt").exists())
    if not cand:
        raise RuntimeError(
            "COLMAP mapper hic model uretemedi. colmap.log'a bak; muhtemel "
            "neden: yetersiz ortusme/doku ya da kamera hic hareket etmemis.")

    recs = [(pycolmap.Reconstruction(str(p)), p) for p in cand]
    rec, model = max(recs, key=lambda rm: rm[0].num_reg_images())

    n_in = len(list(frames_dir.glob("*.jpg")))
    n_reg = rec.num_reg_images()
    rate = n_reg / max(n_in, 1)

    if len(cand) > 1:
        sizes = ", ".join(f"{p.name}:{r.num_reg_images()}" for r, p in recs)
        print(f"[sfm] UYARI: rekonstruksiyon {len(cand)} parcaya bolundu "
              f"({sizes}) — en buyugu secildi. Parcalanma genelde tur "
              "ortasinda kopan ortusmeden olur.")

    print(f"[sfm] {n_reg}/{n_in} kare kaydedildi (%{rate*100:.0f}), "
          f"{rec.num_points3D()} nokta")

    # KALITE KAPISI: dusuk kayit oraninda MVS'e gecmek bosuna 20-40 dk GPU
    # yakar ve sonunda anlamsiz bir mesh cikar. Burada dur.
    if n_reg < 20 or rate < 0.5:
        raise RuntimeError(
            f"SfM basarisiz: {n_in} kareden sadece {n_reg} tanesi kaydedildi "
            f"(%{rate*100:.0f}). MVS'e gecilmiyor — cikacak mesh anlamsiz olur.\n"
            "\nSik nedenler, en olasidan baslayarak:\n"
            "  1) KAMERA HAREKET ETMEMIS. Fotogrametri paralaks ister; sabit\n"
            "     tripoddan cekim ya da yerinde donme (pan) matematiksel olarak\n"
            "     cozulemez. Kamera denegin ETRAFINDA yer degistirmeli.\n"
            "  2) Sahne hareketli: denek kimildamis/konusmus, ya da arka planda\n"
            "     hareket var. SfM katı sahne varsayar.\n"
            "  3) Ortusme yetersiz: tur cok hizli ya da secilen kare sayisi az.\n"
            "     --n-frames degerini artir (orn. 300).\n"
            "  4) Birlestirilmis/kesilmis klipler: --matcher exhaustive dene.\n"
            "  5) Doku yok: duz duvar, parlama, asiri bulanik kareler.\n"
            "\nAyrinti: " + str(log))
    return model
