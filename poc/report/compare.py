"""Dogruluk raporu: kumpas CSV'si vs otomatik olcumler.

Kullanim:
    python -m poc.report.compare calipers.csv vaka_001 vaka_002 ...

calipers.csv kolonlari: case_id, measurement, true_value
(measurement adlari measurements.json anahtarlariyla ayni olmali)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def build_report(calipers_csv: Path, case_dirs: list[Path],
                 out_csv: Path = Path("error_report.csv")) -> pd.DataFrame:
    truth = pd.read_csv(calipers_csv)

    rows = []
    for d in case_dirs:
        mj = d / "measurements.json"
        if not mj.exists():
            print(f"UYARI: {mj} yok, atlandi")
            continue
        m = json.loads(mj.read_text())
        for k, v in m.items():
            rows.append({"case_id": d.name, "measurement": k, "predicted": v})
    pred = pd.DataFrame(rows)

    df = truth.merge(pred, on=["case_id", "measurement"], how="outer")
    df["error"] = df["predicted"] - df["true_value"]
    df["abs_error"] = df["error"].abs()

    stats = (df.dropna(subset=["abs_error"])
               .groupby("measurement")["abs_error"]
               .agg(median="median", mae="mean", n="count")
               .round(3))
    df.to_csv(out_csv, index=False)
    print("\n=== Olcum basina hata (mutlak) ===")
    print(stats.to_string())
    print(f"\nDetay tablo: {out_csv}")

    # Basari cizgisi kontrolu: mm olcumlerinde medyan <= 2.0 mm
    mm_rows = stats[stats.index.str.endswith("_mm")]
    if not mm_rows.empty:
        worst = mm_rows["median"].max()
        ok = "GECTI" if worst <= 2.0 else "GECEMEDI"
        print(f"\nmm olcumlerinde en kotu medyan: {worst:.2f} mm -> {ok} (esik 2.0)")
    return df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    build_report(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
