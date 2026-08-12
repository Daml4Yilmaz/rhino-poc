"""Pipeline parametreleri — tek yerden ayarlanir."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Kare secimi (fotogrametrinin girdisi)
    n_frames: int = 300            # iki gecis boyunca secilecek keskin kare
    blur_min_var: float = 40.0     # Laplacian varyans alt esigi (mutlak ret)
    max_dim: int | None = 1600     # COLMAP'e verilen karenin uzun kenari

    # Markersiz olcek (ARKit poz + LiDAR; bkz. pipeline/scale.py)
    scale_agreement_pct: float = 1.5   # poz/LiDAR ayrisma esigi
    depth_conf_min: int = 2            # LiDAR guven haritasi: 2 = yuksek

    # COLMAP
    colmap_bin: str = "colmap"
    camera_model: str = "OPENCV"
    seq_overlap: int = 20          # video karesi -> sirali eslestirme penceresi
    use_gpu: bool = True

    # MVS bellek: COLMAP varsayilani 32 GB'lik onbellek ister ve paylasimli
    # ortamlarda (Colab ~12.7 GB) surec SIGKILL yer. None -> RAM'e gore otomatik.
    mvs_cache_gb: int | None = None

    # Poisson mesh
    poisson_depth: int = 10
    poisson_trim: float = 7.0      # dusuk yogunluklu vertexleri kirp

    # Cikti
    out_dir: Path = field(default_factory=lambda: Path("vaka_out"))

    def frames_dir(self) -> Path: return self.out_dir / "frames"
    def frames_index(self) -> Path: return self.out_dir / "frames_index.json"
    def masks_dir(self) -> Path: return self.out_dir / "masks"
    def colmap_dir(self) -> Path: return self.out_dir / "colmap"
    def sparse_dir(self) -> Path: return self.colmap_dir() / "sparse"
    def dense_dir(self) -> Path: return self.colmap_dir() / "dense"
