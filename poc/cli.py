"""Command-line interface for metric facial reconstruction."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from .config import ReconstructionConfig
from .logging_utils import configure_logging, format_duration, get_logger
from .state import CaseManifest, stage_signature

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Metric facial surface reconstruction from Stray Scanner capture data.",
)

STAGES = (
    "ingest",
    "mask",
    "sfm",
    "scale",
    "mvs",
    "texture",
    "export",
    "landmarks",
    "measure",
)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _run_stage(
    manifest: CaseManifest,
    name: str,
    parameters: dict[str, Any],
    outputs: list[Path],
    action: Callable[[], dict[str, Any] | None],
    *,
    dependency_signature: str = "",
    resume: bool,
) -> str:
    signature = stage_signature(
        name, manifest.capture_hash, parameters, dependency=dependency_signature
    )
    if resume and manifest.is_current(name, signature, outputs):
        get_logger().info("Stage %-10s | skipped; outputs and parameters are current", name)
        return signature
    if resume and manifest.has_stale_record(name, signature):
        raise RuntimeError(
            f"Cannot resume stage '{name}' because its inputs or parameters changed. "
            "Use a new output directory or run without --resume to rebuild it."
        )
    for output in outputs:
        _remove_path(output)
    manifest.start(name, signature, parameters)
    started_at = time.monotonic()
    get_logger().info("Stage %-10s | started", name)
    try:
        metadata = action() or {}
    except Exception as error:
        manifest.fail(name, error)
        raise
    metadata["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
    manifest.complete(name, metadata)
    get_logger().info(
        "Stage %-10s | complete in %s", name, format_duration(metadata["elapsed_seconds"])
    )
    return signature


@app.command()
def inspect(
    capture: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    json_output: Annotated[
        bool, typer.Option("--json", help="Print machine-readable JSON.")
    ] = False,
) -> None:
    """Validate a Stray export without starting reconstruction."""
    configure_logging()
    from .pipeline.arkit import load_capture

    loaded = load_capture(capture)
    summary = loaded.validation_summary or {}
    if json_output:
        typer.echo(json.dumps(summary, indent=2))


@app.command("download-models")
def download_models(
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Model asset directory.")
    ] = Path("models"),
) -> None:
    """Download the official MediaPipe model required for masking and landmarks."""
    configure_logging()
    from .model_assets import download_face_landmarker

    typer.echo(str(download_face_landmarker(output_dir)))


@app.command()
def reconstruct(
    capture: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o", help="Case output directory.")],
    face_landmarker_model: Annotated[
        Path,
        typer.Option(
            "--face-landmarker-model",
            exists=True,
            dir_okay=False,
            help="MediaPipe face_landmarker.task path; obtain it with 'poc download-models'.",
        ),
    ],
    until: Annotated[str, typer.Option(help=f"Final stage: {', '.join(STAGES)}")] = "measure",
    resume: Annotated[
        bool, typer.Option(help="Reuse only manifest-verified stage outputs.")
    ] = False,
    frame_count: Annotated[int, typer.Option(min=60, max=240)] = 120,
    max_dimension: Annotated[int, typer.Option(min=800, max=1920)] = 1400,
    rotation: Annotated[
        str, typer.Option(help="none, clockwise, or counterclockwise")
    ] = "clockwise",
    sift_gpu: Annotated[bool, typer.Option("--sift-gpu/--no-sift-gpu")] = True,
    matcher: Annotated[str, typer.Option(help="sequential or exhaustive")] = "sequential",
    mvs_references: Annotated[int, typer.Option(min=40, max=160)] = 96,
    mvs_source_images: Annotated[int, typer.Option(min=4, max=12)] = 6,
    mvs_geometric: Annotated[bool, typer.Option("--mvs-geometric/--no-mvs-geometric")] = False,
    mvs_cache_gb: Annotated[int, typer.Option(min=0, max=12)] = 0,
    texture_scale: Annotated[float, typer.Option(min=0.5, max=2.0)] = 1.0,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the complete surface reconstruction and measurement pipeline."""
    if until not in STAGES:
        raise typer.BadParameter(f"Unknown stage '{until}'; expected one of {STAGES}")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = ReconstructionConfig(
        output_dir=output,
        frame_count=frame_count,
        frame_max_dimension=max_dimension,
        rotation=rotation,
        mvs_reference_count=mvs_references,
        mvs_source_images=mvs_source_images,
        mvs_geometric_consistency=mvs_geometric,
        mvs_cache_gb=mvs_cache_gb or None,
        texture_scale_factor=texture_scale,
    )
    configure_logging(config.log_path, verbose=verbose)
    logger = get_logger()
    logger.info("Case output | %s", output)
    logger.info("Detailed log | %s", config.log_path)

    from .pipeline.arkit import load_capture

    loaded_capture = load_capture(capture)
    manifest = CaseManifest(config.manifest_path, capture)
    stop_index = STAGES.index(until)
    dependency = ""

    ingest_parameters = {
        "frame_count": config.frame_count,
        "minimum_sharpness": config.minimum_sharpness,
        "max_dimension": config.frame_max_dimension,
        "rotation": config.rotation,
    }

    def ingest_action() -> dict:
        from .pipeline.frames import select_frames

        capture_summary = loaded_capture.validation_summary or {}
        names = select_frames(
            loaded_capture,
            config.images_dir,
            config.frame_index_path,
            target_count=config.frame_count,
            minimum_sharpness=config.minimum_sharpness,
            max_dimension=config.frame_max_dimension,
            rotation=config.rotation,
        )
        return {"capture": capture_summary, "selected_images": len(names)}

    dependency = _run_stage(
        manifest,
        "ingest",
        ingest_parameters,
        [config.images_dir, config.frame_index_path],
        ingest_action,
        resume=resume,
    )
    if stop_index == 0:
        return

    model_digest = hashlib.sha256(face_landmarker_model.read_bytes()).hexdigest()
    mask_parameters = {
        "method": "mediapipe_tasks_face_landmarker_convex_hull_v1",
        "model_sha256": model_digest,
    }

    def mask_action() -> dict:
        from .pipeline.masking import run_masking

        return run_masking(
            config.images_dir,
            config.masks_dir,
            config.reconstruction_images_dir,
            face_landmarker_model,
        )

    dependency = _run_stage(
        manifest,
        "mask",
        mask_parameters,
        [config.masks_dir, config.reconstruction_images_dir],
        mask_action,
        dependency_signature=dependency,
        resume=resume,
    )
    if stop_index == 1:
        return

    sfm_parameters = {
        "sift_gpu": sift_gpu,
        "matcher": matcher,
        "sequential_overlap": config.sequential_overlap,
    }
    selected_sparse_model: Path | None = None

    def sfm_action() -> dict:
        nonlocal selected_sparse_model
        from .pipeline.sfm import run_sfm

        selected_sparse_model = run_sfm(
            config.reconstruction_images_dir,
            config.frame_index_path,
            config.colmap_dir,
            masks_dir=config.masks_dir,
            colmap_binary=config.colmap_binary,
            use_gpu=sift_gpu,
            sequential_overlap=config.sequential_overlap,
            matcher=matcher,
        )
        (config.colmap_dir / "selected_sparse_model.txt").write_text(
            str(selected_sparse_model.resolve()), encoding="utf-8"
        )
        return {"selected_model": selected_sparse_model.name}

    dependency = _run_stage(
        manifest,
        "sfm",
        sfm_parameters,
        [config.colmap_dir / "selected_sparse_model.txt"],
        sfm_action,
        dependency_signature=dependency,
        resume=resume,
    )
    selected_sparse_model = Path(
        (config.colmap_dir / "selected_sparse_model.txt").read_text(encoding="utf-8").strip()
    )
    if stop_index == 2:
        return

    scale_parameters = {"maximum_disagreement_percent": config.scale_agreement_percent}

    def scale_action() -> dict:
        from .pipeline.scale import compute_scale

        scale = compute_scale(
            selected_sparse_model,
            loaded_capture,
            config.frame_index_path,
            config.scale_path,
            masks_dir=config.masks_dir,
            maximum_disagreement_percent=config.scale_agreement_percent,
        )
        return {"scale_mm_per_unit": scale}

    dependency = _run_stage(
        manifest,
        "scale",
        scale_parameters,
        [config.scale_path],
        scale_action,
        dependency_signature=dependency,
        resume=resume,
    )
    scale_mm_per_unit = float(
        json.loads(config.scale_path.read_text(encoding="utf-8"))["scale_mm_per_unit"]
    )
    if stop_index == 3:
        return

    mvs_parameters = {
        "reference_count": config.mvs_reference_count,
        "source_images": config.mvs_source_images,
        "geometric_consistency": config.mvs_geometric_consistency,
        "cache_gb": config.mvs_cache_gb,
        "poisson_depth": config.poisson_depth,
    }

    def mvs_action() -> dict:
        from .pipeline.mvs import run_mvs

        run_mvs(
            config.reconstruction_images_dir,
            selected_sparse_model,
            config.dense_dir,
            config.raw_mesh_path,
            colmap_binary=config.colmap_binary,
            cache_size_gb=config.mvs_cache_gb,
            max_image_size=config.mvs_max_image_size,
            source_images=config.mvs_source_images,
            maximum_references=config.mvs_reference_count,
            geometric_consistency=config.mvs_geometric_consistency,
            poisson_depth=config.poisson_depth,
        )
        return {"mesh": str(config.raw_mesh_path)}

    dependency = _run_stage(
        manifest,
        "mvs",
        mvs_parameters,
        [config.raw_mesh_path],
        mvs_action,
        dependency_signature=dependency,
        resume=resume,
    )
    if stop_index == 4:
        return

    def texture_action() -> dict:
        from .pipeline.texture import run_texture_mapping

        textured_mesh, texture_image = run_texture_mapping(
            config.dense_dir,
            config.raw_mesh_path,
            config.texture_dir,
            colmap_binary=config.colmap_binary,
            texture_scale_factor=config.texture_scale_factor,
        )
        return {
            "textured_mesh": str(textured_mesh),
            "texture_image": str(texture_image),
        }

    dependency = _run_stage(
        manifest,
        "texture",
        {
            "method": "colmap_mesh_texturer",
            "color_correction": True,
            "texture_scale_factor": config.texture_scale_factor,
        },
        [config.textured_mesh_path, config.texture_image_path],
        texture_action,
        dependency_signature=dependency,
        resume=resume,
    )
    if stop_index == 5:
        return

    def export_action() -> dict:
        from .pipeline.export import export_glb
        from .pipeline.geometry import create_authoritative_geometry

        geometry = create_authoritative_geometry(
            config.raw_mesh_path,
            scale_mm_per_unit,
            config.authoritative_mesh_path,
            config.geometry_metadata_path,
        )
        export_glb(
            config.authoritative_mesh_path,
            config.geometry_metadata_path,
            config.glb_path,
            textured_mesh_path=config.textured_mesh_path,
        )
        return {
            "authoritative_mesh": str(config.authoritative_mesh_path),
            "geometry_id": geometry["geometry_id"],
            "glb": str(config.glb_path),
            "units": "metres",
        }

    dependency = _run_stage(
        manifest,
        "export",
        {
            "geometry_schema": 1,
            "glb_units": "metres",
            "topology_policy": "registered_face_identity",
        },
        [config.authoritative_mesh_path, config.geometry_metadata_path, config.glb_path],
        export_action,
        dependency_signature=dependency,
        resume=resume,
    )
    if stop_index == 6:
        return

    def landmarks_action() -> dict:
        from .pipeline.landmarks import triangulate_landmarks

        result = triangulate_landmarks(
            config.reconstruction_images_dir,
            selected_sparse_model,
            config.frame_index_path,
            config.authoritative_mesh_path,
            config.geometry_metadata_path,
            scale_mm_per_unit,
            config.landmarks_path,
            face_landmarker_model,
        )
        return {"landmark_count": len(result["landmarks"]), "definition": result["definition"]}

    dependency = _run_stage(
        manifest,
        "landmarks",
        {
            "definition": "provisional_mediapipe_surface_landmarks_v2",
            "model_sha256": model_digest,
            "surface_binding": "triangle_index_and_barycentric_coordinates",
        },
        [config.landmarks_path],
        landmarks_action,
        dependency_signature=dependency,
        resume=resume,
    )
    if stop_index == 7:
        return

    def measure_action() -> dict:
        from .pipeline.measure import run_measurements

        measurements = run_measurements(config.landmarks_path, config.measurements_path)
        return {"measurement_count": len(measurements)}

    _run_stage(
        manifest,
        "measure",
        {"definition": "provisional_surface_rhinoplasty_measurements_v1"},
        [config.measurements_path],
        measure_action,
        dependency_signature=dependency,
        resume=resume,
    )
    logger.info(
        "Pipeline complete | model: %s | measurements: %s",
        config.glb_path,
        config.measurements_path,
    )


if __name__ == "__main__":
    app()
