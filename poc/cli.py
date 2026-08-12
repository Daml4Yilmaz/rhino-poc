"""Tek komut: poc process CAPTURE --out vaka_001

CAPTURE = Stray Scanner klasoru (odometry.csv iceren) veya Record3D .r3d.
Icindeki video kareye ayrilir, fotogrametri (COLMAP SfM + MVS) 3D modeli
uretir; ARKit pozu/LiDAR'i SADECE mm carpanini bulmak icin okunur.

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


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi"}


def _require_exists(capture: Path) -> None:
    """Yol yoksa komsu dosyalari listeleyerek dur.

    Colab'da Drive mount'u Linux tarafinda BUYUK-KUCUK HARF DUYARLI; iPhone
    videolari .MOV yazar, yolu .mov diye verince dosya 'yok' sayilir. Bu en
    sik dusulen tuzak oldugu icin ayrica isaret ediyoruz.
    """
    if capture.exists():
        return
    parent = capture.parent
    lines = [f"Yol bulunamadi: {capture}"]
    if parent.is_dir():
        names = sorted(p.name + ("/" if p.is_dir() else "")
                       for p in parent.iterdir())
        near = [n for n in names
                if n.lower().rstrip("/") == capture.name.lower()]
        if near:
            lines.append(f"AMA su var: {near[0]} — sadece BUYUK/KUCUK HARF "
                         "farki. Linux'ta bu iki ayri dosyadir.")
        lines.append(f"{parent} icinde ({len(names)} oge):")
        lines += [f"  {n}" for n in names[:25]]
        if len(names) > 25:
            lines.append(f"  ... (+{len(names) - 25})")
    else:
        lines.append(f"Ust klasor de yok: {parent}")
    raise typer.BadParameter("\n".join(lines))


@app.command()
def process(
    capture: Path = typer.Argument(..., help="Stray Scanner klasoru, .r3d, veya duz video (TEST)"),
    out: Path = typer.Option(..., "--out", help="Vaka cikti klasoru"),
    n_frames: int = typer.Option(300, help="Secilecek kare sayisi"),
    max_dim: int = typer.Option(1600, help="COLMAP karesinin uzun kenari (0=kucultme)"),
    until: str = typer.Option("measure", help=f"Bu adimdan sonra dur: {STEPS}"),
    sift_gpu: bool = typer.Option(True, "--sift-gpu/--no-sift-gpu",
                                  help="COLMAP SIFT'i GPU'da kos (macOS'ta kapat)"),
    matcher: str = typer.Option("sequential", help="sequential | exhaustive"),
    resume: bool = typer.Option(False, "--resume", help="Var olan ara ciktilari atla"),
):
    t0 = time.time()
    cfg = Config(out_dir=out, n_frames=n_frames, use_gpu=sift_gpu,
                 max_dim=max_dim or None)
    out.mkdir(parents=True, exist_ok=True)
    stop = STEPS.index(until)

    # Girdi ya ARKit yakalamasi ya da (SADECE TEST icin) duz video.
    # Once VARLIK kontrolu: yol yoksa sessizce ARKit dalina dusup "taninmayan
    # bicim" demek yaniltici olur, asil sebep yolun yanlis olmasidir.
    _require_exists(capture)
    test_mode = capture.is_file() and capture.suffix.lower() in VIDEO_SUFFIXES
    if test_mode:
        print("=" * 70)
        print("[poc] TEST MODU: duz video — ARKit pozu yok, LiDAR yok.")
        print("[poc] Fotogrametri (frames+sfm+mvs) kosar, OLCEK KOSMAZ.")
        print("[poc] Cikan model BIRIMSIZ olur: acilar ve Goode orani anlamli,")
        print("[poc] mm cinsinden uzunluk/genislik/sapma URETILEMEZ.")
        print("=" * 70)
        video, cap = capture, None
    else:
        from .pipeline.arkit import load_capture
        cap = load_capture(capture)
        video = cap.rgb_path

    # 1. frames — videodan kare cikar (kaynak kare indeksleri korunur)
    if resume and cfg.frames_index().exists():
        print(f"[frames] atlandi (resume): {cfg.frames_dir()} mevcut")
    else:
        from .pipeline.frames import select_frames
        select_frames(video, cfg.frames_dir(), cfg.n_frames,
                      cfg.blur_min_var, cfg.max_dim, cfg.frames_index())
    if stop < 1:
        return _done(t0)

    # 2. sfm — fotogrametri, birimsiz
    model = cfg.sparse_dir() / "0"
    if resume and (model / "cameras.bin").exists():
        print(f"[sfm] atlandi (resume): {model} mevcut")
    else:
        from .pipeline.sfm import run_sfm
        fmap = None
        if cap is not None and cfg.frames_index().exists():
            fmap = json.loads(cfg.frames_index().read_text())["frame_of_image"]
        model = run_sfm(cfg.frames_dir(), cfg.colmap_dir(), cfg.colmap_bin,
                        cfg.camera_model, cfg.seq_overlap, cfg.use_gpu, matcher,
                        capture=cap, frame_of_image=fmap)
    if stop < 2:
        return _done(t0)

    # 3. mvs — yogun yuzey (CUDA sart; macOS'ta Colab'a devret)
    mesh_raw = out / "mesh_raw.ply"
    if resume and mesh_raw.exists():
        print(f"[mvs] atlandi (resume): {mesh_raw} mevcut")
    else:
        from .pipeline.mvs import run_mvs
        mesh_raw = run_mvs(cfg.frames_dir(), model, cfg.dense_dir(),
                           cfg.colmap_bin, cfg.poisson_depth, cfg.poisson_trim)
    if stop < 3:
        return _done(t0)

    # 4. scale — markersiz: ARKit poz (+ LiDAR capraz kontrol)
    scale_json = out / "scale.json"
    if test_mode:
        print("[scale] TEST MODU — atlandi. Model birimsiz kalacak.")
        scale = 1.0
    elif resume and scale_json.exists():
        scale = json.loads(scale_json.read_text())["scale_mm_per_unit"]
        print(f"[scale] atlandi (resume): {scale:.6f} mm/birim")
    else:
        from .pipeline.scale import compute_scale
        masks = cfg.masks_dir() if cfg.masks_dir().is_dir() else None
        scale = compute_scale(out, model, capture, scale_json, masks,
                              cfg.scale_agreement_pct)
    if stop < 4:
        return _done(t0)

    # 5. export — test modunda birimsiz oldugunu dosya adindan belli et
    from .pipeline.export import export_glb
    glb = out / ("model_unitless.glb" if test_mode else "model.glb")
    export_glb(mesh_raw, scale, glb)
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
def scale(
    capture: Path = typer.Argument(..., help="Stray Scanner klasoru veya .r3d"),
    out: Path = typer.Option(..., "--out", help="Vaka klasoru (frames_index.json burada)"),
):
    """Sadece olcek: ARKit pozu + LiDAR -> scale.json"""
    from .pipeline.scale import compute_scale
    cfg = Config(out_dir=out)
    masks = cfg.masks_dir() if cfg.masks_dir().is_dir() else None
    compute_scale(out, cfg.sparse_dir() / "0", capture, out / "scale.json",
                  masks, cfg.scale_agreement_pct)


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
