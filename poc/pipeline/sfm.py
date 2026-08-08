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

    model = sparse / "0"
    if not model.exists():
        raise RuntimeError(
            "COLMAP mapper model uretemedi. colmap.log'a bak; muhtemel neden: "
            "yetersiz ortusme/doku. Cekimi daha yavas ve yakin tekrarla.")

    # Kac kare kayit oldu? (kalite gostergesi)
    import pycolmap
    rec = pycolmap.Reconstruction(str(model))
    print(f"[sfm] {rec.num_reg_images()} kare kaydedildi, "
          f"{rec.num_points3D()} nokta")
    return model
