#!/usr/bin/env python3
"""
31_census_pair_correlation.py -- how strongly are the two summaries the census
compares correlated with each other?

The census of script 12 compares, for each comparable field, its most
attributed central summary (mean or median) with its most attributed
non-central one. The correlation filter of the admission step removes one
member of a pair only at |r| >= C.COLLINEAR_CUT, so the compared summaries can
still be strongly correlated. This script measures that correlation over the
modelling panel, which is what decides whether a tree ensemble can route a
prediction through either member (Discussion, permutation importance).

READS   ../derived*/summary_statistic_census.csv, the modelling panel
WRITES  ../derived*/census_pair_correlation.json

    python3 31_census_pair_correlation.py
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

S12 = __import__("12_summary_statistic_census_and_matched_refit")
OUT_JSON = C.DERIVED / "census_pair_correlation.json"


def main():
    cen = pd.read_csv(C.DERIVED / "summary_statistic_census.csv")
    cen = cen[cen["comparable"].astype(bool)]
    panel = C.load_panel()
    feats = panel["features"]
    # map (field, stat) -> column name, with the parser the census itself used
    lookup = {}
    for f in feats:
        parsed = S12.split_name(f)
        if parsed:
            field, _cls, stat = parsed
            lookup[(field, stat)] = f
    rows = []
    for _, r in cen.iterrows():
        a = lookup.get((r["field"], r["central_stat"]))
        b = lookup.get((r["field"], r["noncentral_stat"]))
        if a is None or b is None:
            rows.append(dict(field=r["field"], central=a, noncentral=b, r=None))
            continue
        x = panel["X"][a].to_numpy(float)
        y = panel["X"][b].to_numpy(float)
        rows.append(dict(field=r["field"], central=a, noncentral=b,
                         r=float(np.corrcoef(x, y)[0, 1])))
    d = pd.DataFrame(rows)
    ar = d["r"].dropna().abs()
    out = dict(n_fields=int(len(d)), n_measured=int(ar.size),
               median_abs_r=float(ar.median()),
               q25_abs_r=float(ar.quantile(0.25)), q75_abs_r=float(ar.quantile(0.75)),
               min_abs_r=float(ar.min()), max_abs_r=float(ar.max()),
               share_above_0_8=float((ar >= 0.8).mean()),
               collinear_cut=C.COLLINEAR_CUT if hasattr(C, "COLLINEAR_CUT") else 0.98,
               pairs=d.to_dict(orient="records"))
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))
    C.write_manifest(__file__, [C.DERIVED / "summary_statistic_census.csv", C.FRAME],
                     [OUT_JSON], settings={})


if __name__ == "__main__":
    main()
