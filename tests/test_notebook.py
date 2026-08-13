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
