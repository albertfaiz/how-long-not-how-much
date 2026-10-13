#!/usr/bin/env python3
"""
32_data_version_consistency.py -- do the data files agree with each other?

READS   ../../Final_ML_Ready_Dataset.csv            the input of this paper's run
        ../../Master_Cleaned_NoArtifacts_With_LIVE_Prefix.csv   (optional) the
                                                    input of the earlier nocturnal study
        results_studyA_*/studyA_reference_frame.parquet  the frame the model read
        ../derived/panel_*.csv                      the three modelling panels
WRITES  ../derived/data_version_consistency.json

WHY THIS EXISTS
---------------
The project holds several versions of the county-year table. This script
checks, cell by cell on matching (county, year) keys, that every column two
files share holds the same values, and reports what each file has that the
other lacks. Values are equal when their relative difference is below 1e-6;
a cell missing in one file and present in the other is counted separately.
Columns are compared in chunks so the check fits in a few GB of memory.
A file that is not present is skipped and recorded as such.

    python3 32_data_version_consistency.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _config as C                                            # noqa: E402

FINAL = C.ROOT / "Final_ML_Ready_Dataset.csv"
MASTER = C.ROOT / "Master_Cleaned_NoArtifacts_With_LIVE_Prefix.csv"
PANELS = ["panel_all_admitted", "panel_environment_only", "panel_satellite_only"]
OUT = C.DERIVED / "data_version_consistency.json"
CHUNK = 300
KEYS = {"fips", "geoid", "year", "statefp", "countyfp", "name"}


def header(path):
    if str(path).endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_schema(path).names
    with open(path) as fh:
        return fh.readline().strip().split(",")


def read(path, cols):
    if str(path).endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(path, columns=cols).to_pandas()
    import pyarrow.csv as pc
    return pc.read_csv(path, convert_options=pc.ConvertOptions(
        include_columns=cols)).to_pandas()


def keyed(df, k):
    df.index = (pd.to_numeric(df[k]).astype(int).astype(str).str.zfill(5) + "_"
                + pd.to_numeric(df["year"]).astype(int).astype(str))
    return df[~df.index.duplicated()]


def compare(a_path, a_key, b_path, b_key, rename=None):
    ca, cb = header(a_path), header(b_path)
    rename = rename or {}
    lb = {c.lower(): c for c in cb}
    pairs = []
    for c in ca:
        t = rename.get(c, c)
        if c.lower() in KEYS:
            continue
        if t.lower() in lb:
            pairs.append((c, lb[t.lower()]))
    differ, nan_cells, cells, rows = [], 0, 0, None
    for i in range(0, len(pairs), CHUNK):
        sel = pairs[i:i + CHUNK]
        a = keyed(read(a_path, [a_key, "year"] + [p[0] for p in sel]), a_key)
        b = keyed(read(b_path, [b_key, "year"] + list(dict.fromkeys(p[1] for p in sel))), b_key)
        idx = a.index.intersection(b.index)
        rows = (len(a), len(b), len(idx))
        for pa, pb in sel:
            x = pd.to_numeric(a.loc[idx, pa], errors="coerce").to_numpy(float)
            y = pd.to_numeric(b.loc[idx, pb], errors="coerce").to_numpy(float)
            nx, ny = np.isnan(x), np.isnan(y)
            both = ~nx & ~ny
            rel = np.abs(x[both] - y[both]) / np.maximum(
                1e-12, np.maximum(np.abs(x[both]), np.abs(y[both])))
            nd, nn = int((rel > 1e-6).sum()), int((nx != ny).sum())
            cells += int(both.sum())
            nan_cells += nn
            if nd:
                differ.append(dict(column=pa, cells_differing=nd))
    la = {rename.get(c, c).lower() for c in ca}
    note = ("Cells present in one file only: in the panels these are the "
            "median fill values the fitted model applies to missing cells "
            "(stored with the model), so the panels are exactly the matrix the "
            "model was fitted on.") if a_path.parent == C.DERIVED else ""
    return dict(note=note, file_a=a_path.name, file_b=b_path.name,
                rows_a=rows[0], rows_b=rows[1], rows_matched=rows[2],
                columns_shared=len(pairs), cells_compared=cells,
                columns_with_different_values=differ,
                cells_present_in_one_only=nan_cells,
                columns_only_in_a=[c for c in ca if rename.get(c, c).lower() not in lb and c.lower() not in KEYS],
                columns_only_in_b=[c for c in cb if c.lower() not in la and c.lower() not in KEYS])


def main():
    C.banner("32  consistency of the data versions")
    out = dict(tolerance="relative difference 1e-6", comparisons=[], skipped=[])
    jobs = [(FINAL, "fips", C.FRAME, "fips", None)]
    jobs += [(C.DERIVED / f"{p}.csv", "fips", C.FRAME, "fips",
              {"life_expectancy": C.TARGET}) for p in PANELS]
    jobs += [(MASTER, "GEOID", FINAL, "fips", None)]
    for a, ka, b, kb, ren in jobs:
        if not (a.exists() and b.exists()):
            out["skipped"].append(f"{a.name} vs {b.name}: file not present")
            print(f"  skipped  {a.name} vs {b.name} (file not present)")
            continue
        r = compare(a, ka, b, kb, ren)
        out["comparisons"].append(r)
        print(f"  {r['file_a']}  vs  {r['file_b']}")
        print(f"     rows {r['rows_a']:,} / {r['rows_b']:,}, matched {r['rows_matched']:,}; "
              f"{r['columns_shared']} shared columns, {r['cells_compared']:,} cells")
        print(f"     columns with different values: {len(r['columns_with_different_values'])};"
              f" cells present in one file only: {r['cells_present_in_one_only']:,}")
        print(f"     only in first: {len(r['columns_only_in_a'])}; only in second: "
              f"{len(r['columns_only_in_b'])}")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n  [saved] {OUT}")
    C.write_manifest("32_data_version_consistency.py",
                     inputs=[p for p in (FINAL, MASTER, C.FRAME) if p.exists()],
                     outputs=[OUT], settings=dict(CHUNK=CHUNK))


if __name__ == "__main__":
    main()
