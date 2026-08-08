"""Tek komut: poc process video.mp4 --out vaka_001

Adimlar sirayla kosar; --until ile erken durabilir, ara ciktilar vaka
klasorunde birikir. Hafta-2 adimlari (mask/gs/flame) stub — atlanir.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from .config import Config

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

STEPS = ["frames", "sfm", "mvs", "scale", "export", "measure"]


@app.command()
def process(
    video: Path = typer.Argument(..., help="Telefon videosu (mp4/mov)"),
    out: Path = typer.Option(..., "--out", help="Vaka cikti klasoru"),
    marker_mm: float = typer.Option(50.0, help="Basili ArUco kenar uzunlugu (CETVELLE OLC!)"),
    n_frames: int = typer.Option(200, help="Secilecek kare sayisi"),
    until: str = typer.Option("measure", help=f"Bu adimdan sonra dur: {STEPS}"),
    no_gpu: bool = typer.Option(False, help="COLMAP'i GPU'suz kos (yavas; MVS calismaz)"),
    matcher: str = typer.Option("sequential", help="sequential | exhaustive (birlestirilmis videolarda exhaustive kullan)"),
    resume: bool = typer.Option(False, "--resume", help="Var olan ara ciktilari atla (kesintiden devam)"),
):
    t0 = time.time()
    cfg = Config(out_dir=out, marker_mm=marker_mm, n_frames=n_frames,
                 use_gpu=not no_gpu)
    out.mkdir(parents=True, exist_ok=True)
    stop = STEPS.index(until)

    # 1. frames
    if resume and len(list(cfg.frames_dir().glob("*.jpg"))) >= 30:
        print(f"[frames] atlandi (resume): {cfg.frames_dir()} mevcut")
    else:
        from .pipeline.frames import select_frames
        select_frames(video, cfg.frames_dir(), cfg.extract_fps, cfg.n_frames,
                      cfg.blur_min_var)
    if stop < 1:
        return _done(t0)

    # 2. sfm
    model = cfg.sparse_dir() / "0"
    if resume and (model / "cameras.bin").exists():
        print(f"[sfm] atlandi (resume): {model} mevcut")
    else:
        from .pipeline.sfm import run_sfm
        model = run_sfm(cfg.frames_dir(), cfg.colmap_dir(), cfg.colmap_bin,
                        cfg.camera_model, cfg.seq_overlap, cfg.use_gpu, matcher)
    if stop < 2:
        return _done(t0)

    # 3. mvs  (not: yarim kalmis MVS'te COLMAP hazir derinlik haritalarini
    # kendisi atlar, yani resume olmasa da kismi ilerleme bosa gitmez)
    mesh_raw = out / "mesh_raw.ply"
    if resume and mesh_raw.exists():
        print(f"[mvs] atlandi (resume): {mesh_raw} mevcut")
    else:
        from .pipeline.mvs import run_mvs
        mesh_raw = run_mvs(cfg.frames_dir(), model, cfg.dense_dir(),
                           cfg.colmap_bin, cfg.poisson_depth, cfg.poisson_trim)
    if stop < 3:
        return _done(t0)

    # 4. scale
    scale_json = out / "scale.json"
    if resume and scale_json.exists():
        scale = json.loads(scale_json.read_text())["scale_mm_per_unit"]
        print(f"[scale] atlandi (resume): {scale:.6f} mm/birim")
    else:
        from .pipeline.scale import compute_scale
        scale = compute_scale(cfg.frames_dir(), model, scale_json,
                              cfg.marker_mm, cfg.marker_dict, cfg.marker_id)
    if stop < 4:
        return _done(t0)

    # 5. export (olcekli GLB)
    from .pipeline.export import export_glb
    export_glb(mesh_raw, scale, out / "model.glb")
    if stop < 5:
        return _done(t0)

    # 6. measure — landmarks.json varsa (hafta 2'de FLAME uretir)
    lm = out / "landmarks.json"
    if lm.exists():
        from .pipeline.measure import run_measure
        run_measure(lm, out / "measurements.json")
    else:
        print("[measure] landmarks.json yok — atlandi "
              "(hafta 2: FLAME kaydi uretecek).")
    _done(t0)


def _done(t0: float) -> None:
    print(f"[poc] bitti — {time.time() - t0:.0f} sn")


@app.command()
def measure(
    landmarks: Path = typer.Argument(..., help="landmarks.json (mm)"),
    out: Path = typer.Option(Path("measurements.json"), "--out"),
):
    """Sadece olcum: landmarks.json -> measurements.json"""
    from .pipeline.measure import run_measure
    run_measure(landmarks, out)


if __name__ == "__main__":
    app()
