"""Compare automated measurements with future manual reference measurements."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_report(
    reference_csv: Path,
    case_directories: list[Path],
    output_csv: Path = Path("measurement_error_report.csv"),
) -> pd.DataFrame:
    reference = pd.read_csv(reference_csv)
    rows = []
    for case_directory in case_directories:
        measurement_path = case_directory / "measurements.json"
        if not measurement_path.exists():
            continue
        document = json.loads(measurement_path.read_text(encoding="utf-8"))
        measurements = document.get("measurements", document)
        for name, value in measurements.items():
            rows.append({"case_id": case_directory.name, "measurement": name, "predicted": value})
    predicted = pd.DataFrame(rows, columns=["case_id", "measurement", "predicted"])
    report = reference.merge(predicted, on=["case_id", "measurement"], how="outer")
    report["error"] = report["predicted"] - report["true_value"]
    report["absolute_error"] = report["error"].abs()
    report.to_csv(output_csv, index=False)
    return report
