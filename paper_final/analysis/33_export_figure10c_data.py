#!/usr/bin/env python3
"""
33_export_figure10c_data.py -- the data behind Figure 10C in one small file.

READS   the out-of-fold predictions of the production model
        the reference frame, for the six socioeconomic indicators
        ../derived/residual_social_determinants.json (script 28)
WRITES  ../derived/figure10c_residual_and_indicators.csv
        ../derived/figure10c_check.json

WHY THIS EXISTS
---------------
Figure 10C correlates each county's mean out-of-fold residual (observed minus
predicted life expectancy) with six socioeconomic indicators named in advance.
None of these indicators entered the model. This script writes exactly that
table, one row per county, so a reader can recompute every bar of the panel
without the full data set, and checks that the recomputed Spearman
correlations equal the ones script 28 reported.

The indicators are county means over the years each is observed in the
reference frame (2009 to 2019 for most), the same aggregation script 28 uses.
Nothing here is new analysis.

    python3 33_export_figure10c_data.py
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

OUT_CSV = C.DERIVED / "figure10c_residual_and_indicators.csv"
OUT_JSON = C.DERIVED / "figure10c_check.json"
SHORT = {
    "sdoh_saipe_pct_pov": "poverty_pct",
    "sdoh_saipe_median_hh_income": "median_household_income_usd",
    "sdoh_unemployment_rate": "unemployment_pct",
    "sdoh_acs_pct_uninsured": "uninsured_pct",
    "sdoh_bachelor's_degree_or_higher_%": "bachelors_or_higher_pct",
    "sdoh_disability_rate": "disability_pct",
}


def main():
    import pyarrow.parquet as pq
    from scipy.stats import spearmanr

    C.banner("33  data behind Figure 10C")
    oof = pd.read_csv(C.OOF_A)
    oof["resid"] = oof[C.TARGET] - oof["pred"]
    county = (oof.groupby(C.GROUP)
                 .agg(resid=("resid", "mean"), observed=(C.TARGET, "mean"),
                      predicted=("pred", "mean"))
                 .reset_index())
    county["state"] = county[C.GROUP].astype(int).map(lambda f: f"{f:05d}"[:2])

    fr = pq.read_table(C.FRAME, columns=[C.GROUP, C.YEAR] + list(SHORT)).to_pandas()
    cm = fr.groupby(C.GROUP)[list(SHORT)].mean()
    years = {SHORT[c]: [int(fr.loc[fr[c].notna(), C.YEAR].min()),
                        int(fr.loc[fr[c].notna(), C.YEAR].max())] for c in SHORT}
    d = county.merge(cm, left_on=C.GROUP, right_index=True, how="inner")
    d = d.rename(columns={C.GROUP: "fips", **SHORT})
    d["fips"] = d["fips"].astype(int).map(lambda f: f"{f:05d}")
    cols = ["fips", "state", "observed", "predicted", "resid"] + list(SHORT.values())
    d[cols].to_csv(OUT_CSV, index=False, float_format="%.6g")

    ref = json.load(open(C.DERIVED / "residual_social_determinants.json"))
    ref = {r["column"]: r for r in ref["indicators"]}
    check = []
    for col, short in SHORT.items():
        ok = d[short].notna()
        rho = float(spearmanr(d.loc[ok, short], d.loc[ok, "resid"])[0])
        same = abs(rho - ref[col]["rho"]) < 1e-6
        check.append(dict(indicator=short, label=ref[col]["label"],
                          rho=rho, rho_script28=ref[col]["rho"], n=int(ok.sum()),
                          years=years[short], matches=same))
        print(f"  {ref[col]['label']:<42} rho {rho:+.3f}  "
              f"{'matches script 28' if same else 'DIFFERS from script 28'}")
    json.dump(dict(n_counties=int(len(d)), indicators=check,
                   note="County means; residual is observed minus predicted "
                        "out of fold. Reproduces Figure 10C."),
              open(OUT_JSON, "w"), indent=1)
    print(f"\n  [saved] {OUT_CSV.name}  ({len(d):,} counties)")
    C.write_manifest("33_export_figure10c_data.py",
                     inputs=[C.OOF_A, C.FRAME],
                     outputs=[OUT_CSV, OUT_JSON], settings={})
    if not all(x["matches"] for x in check):
        sys.exit(1)


if __name__ == "__main__":
    main()
