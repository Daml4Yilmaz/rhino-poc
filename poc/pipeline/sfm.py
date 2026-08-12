"""Adim (b): COLMAP ile kamera pozlari (SfM).

COLMAP binary'sini subprocess ile cagiririz (pycolmap sadece model okumak
icin). Video karelerinde sirali eslestirme (sequential matching) hem hizli
hem yeterli; loop detection kulaktan kulaga turun iki ucunu baglar.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str], log_file: Path) -> None:
    with open(log_file, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def run_sfm(frames_dir: Path, colmap_dir: Path, colmap_bin: str = "colmap",
            camera_model: str = "OPENCV", seq_overlap: int = 15,
            use_gpu: bool = True, matcher: str = "sequential") -> Path:
    """Sparse rekonstruksiyon uretir; sparse/0 yolunu dondurur."""
    colmap_dir.mkdir(parents=True, exist_ok=True)
    db = colmap_dir / "database.db"
    sparse = colmap_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    log = colmap_dir / "colmap.log"
    gpu = "1" if use_gpu else "0"

    _run([colmap_bin, "feature_extractor",
          "--database_path", str(db),
          "--image_path", str(frames_dir),
          "--ImageReader.single_camera", "1",       # tek telefon, tek kamera
          "--ImageReader.camera_model", camera_model,
          "--SiftExtraction.use_gpu", gpu], log)

    if matcher == "exhaustive":
        # Birlestirilmis (concat) videolar icin: klip sinirindaki siçrama
        # sequential eslestirmeyi bolebilir; exhaustive herkesle eslestirir.
        # ~200 karede T4'te ~10-20 dk.
        _run([colmap_bin, "exhaustive_matcher",
              "--database_path", str(db),
              "--SiftMatching.use_gpu", gpu], log)
    else:
        _run([colmap_bin, "sequential_matcher",
              "--database_path", str(db),
              "--SequentialMatching.overlap", str(seq_overlap),
              "--SequentialMatching.loop_detection", "0",
              "--SiftMatching.use_gpu", gpu], log)

    _run([colmap_bin, "mapper",
          "--database_path", str(db),
          "--image_path", str(frames_dir),
          "--output_path", str(sparse)], log)

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
