import json
from pathlib import Path


def test_colab_notebook_is_valid_and_has_no_saved_outputs() -> None:
    notebook = json.loads(Path("colab_reconstruction.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(cell.get("outputs") == [] for cell in code_cells)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "download-models" in source
    assert "--face-landmarker-model" in source
    assert "--resume" in source
    assert "Dorsal Hump Reduction (mm)" in source
    assert "simulate-dorsal-hump" in source
    assert "Create simulation" in source
    assert "using the completed Drive case" in source
    assert 'previous_button.close()' in source
    assert '"--output-dir"' in source
    assert "completed_dorsal_requests" not in source
    assert "EXPECTED_DORSAL_SOLVER_ID" in source
    assert "DORSAL_VAULT_SOLVER_ID" in source
    assert "Refusing to run a stale dorsal-hump solver" in source
    assert "import json" in source
    assert "previous_button.on_click(previous_handler, remove=True)" in source
    assert "_dorsal_simulation_in_progress" in source
    assert '"viewer_glb"' in source
    assert '"affected_roi_render_png"' in source
    assert '"profile_before_png"' in source
    assert '"moved_vertices_heatmap_png"' in source
    assert '"front_displacement_heatmap_png"' in source
    assert '"profile_displacement_heatmap_png"' in source


def test_poisson_diagnostic_notebook_is_valid_and_non_authoritative() -> None:
    notebook = json.loads(Path("colab_poisson_diagnostics.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(cell.get("outputs") == [] for cell in code_cells)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_poisson_diagnostic" in source
    assert "PoissonDiagnosticConfig" in source
    assert "depth_sweep = " in source
    assert "depths = (production_depth, *remaining_depths)" in source
    assert "RUN_DEPTH_SWEEP" not in source
    assert "face_dense_fused.ply" in source
    assert "non-authoritative" in source.lower()
