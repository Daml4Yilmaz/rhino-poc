"""Pipeline parametreleri — tek yerden ayarlanir."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Kare cikarma / secim
    extract_fps: int = 15          # 20-30 sn video @15fps -> 300-450 ham kare
    n_frames: int = 200            # secilecek iyi kare sayisi
    blur_min_var: float = 40.0     # Laplacian varyans alt esigi (mutlak ret)

    # ArUco olcek
    marker_dict: str = "DICT_4X4_50"
    marker_id: int = 0
    marker_mm: float = 50.0        # BASILI marker'in CETVELLE OLCULMUS kenar uzunlugu!

    # COLMAP
    colmap_bin: str = "colmap"
    camera_model: str = "OPENCV"
    seq_overlap: int = 15          # video icin sequential matching penceresi
    use_gpu: bool = True

    # Poisson mesh
    poisson_depth: int = 10
    poisson_trim: float = 7.0      # dusuk yogunluklu vertexleri kirp

    # Cikti
    out_dir: Path = field(default_factory=lambda: Path("vaka_out"))

    def frames_dir(self) -> Path: return self.out_dir / "frames"
    def colmap_dir(self) -> Path: return self.out_dir / "colmap"
    def sparse_dir(self) -> Path: return self.colmap_dir() / "sparse"
    def dense_dir(self) -> Path: return self.colmap_dir() / "dense"
