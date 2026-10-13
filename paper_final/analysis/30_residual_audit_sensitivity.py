#!/usr/bin/env python3
"""
30_residual_audit_sensitivity.py -- does the residual audit of script 28 rest
on the health-related columns of the County Health Rankings?

Script 28 predicts the county residual of the environment-only model from the
socioeconomic columns the admission gate excluded. Sixteen of those columns
come from the County Health Rankings and several describe health behaviours or
health status (smoking, obesity, diabetes, physical inactivity, poor health).
They are not socioeconomic conditions in the strict sense, so the audit is
repeated here without them, and once more with the American Community Survey
columns alone. Same county means, same learner, same folds grouped by state.

READS   the same inputs as script 28
WRITES  ../derived*/residual_audit_sensitivity.json

    python3 30_residual_audit_sensitivity.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _config as C                                            # noqa: E402

S28 = __import__("28_residual_and_social_determinants")
OUT_JSON = C.DERIVED / "residual_audit_sensitivity.json"


def source_of(col):
    if col.startswith("sdoh_acs"):
        return "American Community Survey"
    if col.startswith("sdoh_chr"):
        return "County Health Rankings"
    if col.startswith("sdoh_saipe"):
        return "Small Area Income and Poverty Estimates"
    if col.startswith("econ"):
        return "Regional economic accounts"
    return "Summary indicators of the database"


def main():
    import pyarrow.parquet as pq
    C.banner("30  residual audit without the health-related columns")
    oof = pd.read_csv(C.OOF_A)
    oof["resid"] = oof[C.TARGET] - oof["pred"]
    county = oof.groupby(C.GROUP).agg(resid=("resid", "mean")).reset_index()
    county["state"] = county[C.GROUP].astype(int).map(lambda f: f"{f:05d}"[:2])

    gate = pd.read_csv(C.FEATURE_GATE_A)
    sdoh = gate[gate["reason"].astype(str).str.contains("socio")
                & ~gate["modality"].isin(S28.EXCLUDED_MODALITIES)]["feature"].tolist()
    names = set(pq.read_schema(C.FRAME).names)
    sdoh = [c for c in sdoh if c in names]
    fr = pq.read_table(C.FRAME, columns=[C.GROUP, C.YEAR] + sdoh).to_pandas()
    cm = fr.groupby(C.GROUP)[sdoh].mean()
    keep = [c for c in sdoh if cm[c].notna().mean() >= S28.MIN_COUNTY_COVERAGE
            and cm[c].nunique() > 5]
    years = fr.loc[fr[keep].notna().any(axis=1), C.YEAR]
    d = county.merge(cm[keep], left_on=C.GROUP, right_index=True, how="inner")
    y = d["resid"].to_numpy()
    st = d["state"].to_numpy()

    by_source = pd.Series([source_of(c) for c in keep]).value_counts().to_dict()
    sets = {
        "all_columns": keep,
        "without_county_health_rankings": [c for c in keep
                                           if not c.startswith("sdoh_chr")],
        "american_community_survey_only": [c for c in keep
                                           if c.startswith("sdoh_acs")],
    }
    out = dict(n_counties=int(len(d)), columns_by_source=by_source,
               years=[int(years.min()), int(years.max())], arms={})
    for name, cols in sets.items():
        X = d[cols].fillna(d[cols].median())
        r2 = S28.grouped_oof_r2(X, y, st)
        out["arms"][name] = dict(n_columns=len(cols), oof_r2=r2)
        print(f"  {name:<34} {len(cols):>4} columns   R2 {r2:+.3f}")
    print(f"  columns by source: {by_source}")
    print(f"  years present: {out['years']}")
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"  wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
