"""Summarize quality-screened SMAP SPL3SMP_E soil moisture over Kuttanad."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "smap"
OUT_PATH = ROOT / "data" / "processed" / "smap_soil_moisture_kuttanad.json"
DATES = ("2018-07-01", "2018-07-13", "2018-07-25", "2018-08-06", "2018-08-18")
BBOX = {"west": 76.30, "east": 76.50, "south": 9.30, "north": 9.50}


def _read_date(path: Path) -> dict:
    result = {
        "granule": path.name,
        "soil_moisture_m3_m3": None,
        "recommended_cell_count": 0,
        "unmasked_cell_count": 0,
        "bbox_cell_count": 0,
        "quality_flag_counts": {},
        "status": None,
    }
    quality_counts: dict[str, int] = {}
    value_sum = 0.0
    value_count = 0

    with Dataset(path, "r") as h5:
        for pass_name in ("AM", "PM"):
            group = h5.groups[f"Soil_Moisture_Retrieval_Data_{pass_name}"]
            suffix = "_pm" if pass_name == "PM" else ""
            lat = np.ma.asarray(group.variables[f"latitude{suffix}"][:])
            lon = np.ma.asarray(group.variables[f"longitude{suffix}"][:])
            moisture = np.ma.asarray(group.variables[f"soil_moisture{suffix}"][:])
            quality = np.ma.asarray(group.variables[f"retrieval_qual_flag{suffix}"][:])

            inside = (
                (lat >= BBOX["south"])
                & (lat <= BBOX["north"])
                & (lon >= BBOX["west"])
                & (lon <= BBOX["east"])
            ).filled(False)
            moisture_in = moisture[inside]
            quality_in = quality[inside]
            result["bbox_cell_count"] += int(inside.sum())

            values = moisture_in.filled(np.nan).astype(np.float64)
            flags = quality_in.filled(65535).astype(np.uint16)
            unmasked = (
                ~np.ma.getmaskarray(moisture_in)
                & np.isfinite(values)
                & (values >= 0.02)
                & (values <= 0.50)
            )
            # NSIDC identifies quality flags 0 and 8 as recommended for soil moisture.
            recommended = unmasked & np.isin(flags, (0, 8))
            result["unmasked_cell_count"] += int(unmasked.sum())
            result["recommended_cell_count"] += int(recommended.sum())

            for flag, count in zip(*np.unique(quality_in.compressed(), return_counts=True)):
                key = str(int(flag))
                quality_counts[key] = quality_counts.get(key, 0) + int(count)
            if recommended.any():
                value_sum += float(values[recommended].sum())
                value_count += int(recommended.sum())

    result["quality_flag_counts"] = quality_counts
    if value_count:
        result["soil_moisture_m3_m3"] = round(value_sum / value_count, 5)
        result["status"] = f"{value_count} recommended-quality SMAP cells in the study area"
    elif result["bbox_cell_count"]:
        result["status"] = (
            "No recommended-quality SMAP retrievals in the study area; "
            f"{result['unmasked_cell_count']} unmasked retrieval(s) failed the quality screen"
        )
    else:
        result["status"] = "No SMAP grid cells fall inside the study-area bounds"
    return result


def main() -> None:
    files = {re.search(r"_(\d{8})_", path.name).group(1): path for path in RAW_DIR.glob("*.h5")}
    by_date = {}
    for day in DATES:
        compact = day.replace("-", "")
        path = files.get(compact)
        if path is None:
            by_date[day] = {
                "soil_moisture_m3_m3": None,
                "recommended_cell_count": 0,
                "unmasked_cell_count": 0,
                "bbox_cell_count": 0,
                "quality_flag_counts": {},
                "status": "SMAP granule not present in data/raw/smap",
            }
        else:
            by_date[day] = _read_date(path)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "source": "NASA SMAP SPL3SMP_E Version 6",
                "units": "m3/m3",
                "quality_rule": "Use retrieval_qual_flag 0 or 8, per NSIDC Version 6 user guide",
                "spatial_bounds": BBOX,
                "dates": by_date,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_PATH}")
    for day, summary in by_date.items():
        print(
            f"{day}: value={summary['soil_moisture_m3_m3']} m3/m3; "
            f"recommended={summary['recommended_cell_count']}; "
            f"unmasked={summary['unmasked_cell_count']}; {summary['status']}"
        )


if __name__ == "__main__":
    main()
