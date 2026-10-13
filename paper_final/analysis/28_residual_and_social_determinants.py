#!/usr/bin/env python3
"""
28_residual_and_social_determinants.py -- where an environment-only model is
blind, and whether that blindness is socioeconomic.

READS   the out-of-fold predictions of the production model
        the reference frame, for the socioeconomic columns the admission gate
        excluded from the model ("socioeconomic-not-in-scope")
        the modelling panel, for the environmental control
WRITES  ../derived*/residual_social_determinants.json
        ../derived*/residual_social_determinants_counties.csv

THE QUESTION
------------
No socioeconomic, demographic or clinical variable entered the model. Its
out-of-fold residual is therefore the part of county life expectancy the
physical environment does not account for. If that residual is structured by
socioeconomic disadvantage, the socioeconomic columns the model never saw
should predict it, and the environmental columns it did see should not.

Both are measured here with the same learner, on county means, under folds
grouped by state so that a county's neighbours do not predict it. The
socioeconomic indicators are named in advance (POLICY_INDICATORS) rather than
picked after looking, and their correlations carry a correction for the
number tested.

Residual sign: observed minus predicted. A negative residual is a county whose
life expectancy is worse than its physical environment predicts.

    python3 28_residual_and_social_determinants.py
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

OUT_JSON = C.DERIVED / "residual_social_determinants.json"
OUT_CSV = C.DERIVED / "residual_social_determinants_counties.csv"

# Named before any correlation was computed: the six indicators most often used
# to describe county socioeconomic disadvantage in the United States.
POLICY_INDICATORS = {
    "sdoh_saipe_pct_pov": "Population below the poverty line (%)",
    "sdoh_saipe_median_hh_income": "Median household income (USD)",
    "sdoh_unemployment_rate": "Unemployment rate (%)",
    "sdoh_acs_pct_uninsured": "Without health insurance (%)",
    "sdoh_bachelor's_degree_or_higher_%": "Bachelor's degree or higher (%)",
    "sdoh_disability_rate": "Disability rate (%)",
}
EXCLUDED_MODALITIES = {"Health outcomes"}   # behaviours and outcomes are not
                                             # socioeconomic conditions
MIN_COUNTY_COVERAGE = 0.80
LARGE_MISS = 3.0                             # years, as in the main text


def grouped_oof_r2(X, y, states, seed=0):
    """Out-of-fold R2 of a gradient-boosting model, folds grouped by state."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=8).split(X, y, states):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                          max_leaf_nodes=31,
                                          min_samples_leaf=20,
                                          l2_regularization=1.0,
                                          random_state=seed)
        m.fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    return float(r2_score(y, pred))


def main():
    import pyarrow.parquet as pq
    from scipy.stats import spearmanr

    C.banner("28  the residual and the social determinants the model never saw")
    oof = pd.read_csv(C.OOF_A)
    oof["resid"] = oof[C.TARGET] - oof["pred"]
    county = (oof.groupby(C.GROUP)
                 .agg(resid=("resid", "mean"), observed=(C.TARGET, "mean"),
                      predicted=("pred", "mean"))
                 .reset_index())
    county["state"] = county[C.GROUP].astype(int).map(lambda f: f"{f:05d}"[:2])
    print(f"  {len(county):,} counties, mean residual sd "
          f"{county.resid.std():.3f} years")

    gate = pd.read_csv(C.FEATURE_GATE_A)
    sdoh = gate[gate["reason"].astype(str).str.contains("socio")
                & ~gate["modality"].isin(EXCLUDED_MODALITIES)]["feature"].tolist()
    frame_cols = set(pq.read_schema(C.FRAME).names)
    sdoh = [c for c in sdoh if c in frame_cols]
    fr = pq.read_table(C.FRAME, columns=[C.GROUP, C.YEAR] + sdoh).to_pandas()
    cm = fr.groupby(C.GROUP)[sdoh].mean()
    keep = [c for c in sdoh if cm[c].notna().mean() >= MIN_COUNTY_COVERAGE
            and cm[c].nunique() > 5]
    cm = cm[keep]
    print(f"  {len(sdoh)} socioeconomic columns excluded by the gate, "
          f"{len(keep)} with county coverage of at least "
          f"{100 * MIN_COUNTY_COVERAGE:.0f} per cent")

    d = county.merge(cm, left_on=C.GROUP, right_index=True, how="inner")
    Xs = d[keep].fillna(d[keep].median())
    y = d["resid"].to_numpy()
    states = d["state"].to_numpy()

    # environmental control: county means of the model's own columns
    p = C.load_panel()
    env = (pd.concat([p["df"][[C.GROUP]], p["X"]], axis=1)
             .groupby(C.GROUP).mean())
    de = d[[C.GROUP]].merge(env, left_on=C.GROUP, right_index=True,
                            how="left").drop(columns=[C.GROUP])
    de = de.fillna(de.median())

    r2_sdoh = grouped_oof_r2(Xs, y, states)
    r2_env = grouped_oof_r2(de, y, states)
    print(f"\n  out-of-fold R2 for the county residual, folds grouped by state")
    print(f"    from the {len(keep)} socioeconomic columns the model never saw  "
          f"{r2_sdoh:+.3f}")
    print(f"    from the {de.shape[1]} environmental columns it did see        "
          f"{r2_env:+.3f}")

    rows, pvals = [], []
    for col, label in POLICY_INDICATORS.items():
        if col not in d.columns:
            continue
        ok = d[col].notna()
        rho, pv = spearmanr(d.loc[ok, col], d.loc[ok, "resid"])
        rows.append(dict(column=col, label=label, rho=float(rho), p=float(pv),
                         n=int(ok.sum())))
        pvals.append(pv)
    # Holm correction over the pre-declared six
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
    print("\n  pre-declared indicators, Spearman with the county residual")
    for r in rows:
        print(f"    {r['label']:<42} rho {r['rho']:+.3f}  Holm p {r['p_holm']:.2g}")

    miss = d["resid"] <= -LARGE_MISS
    prof = {}
    for col, label in POLICY_INDICATORS.items():
        if col in d.columns:
            pct = d[col].rank(pct=True)
            prof[col] = dict(label=label,
                             median_percentile_large_miss=float(
                                 100 * pct[miss].median()),
                             median_percentile_rest=float(
                                 100 * pct[~miss].median()))
    print(f"\n  {int(miss.sum())} counties where the model overpredicts by "
          f"{LARGE_MISS:g} years or more on average")
    for col, v in prof.items():
        print(f"    {v['label']:<42} median percentile "
              f"{v['median_percentile_large_miss']:.0f} against "
              f"{v['median_percentile_rest']:.0f}")

    d[[C.GROUP, "state", "observed", "predicted", "resid"]].to_csv(
        OUT_CSV, index=False)
    blob = dict(n_counties=int(len(d)), n_sdoh_columns=len(keep),
                n_env_columns=int(de.shape[1]),
                oof_r2_residual_from_sdoh=r2_sdoh,
                oof_r2_residual_from_environment=r2_env,
                residual_sd=float(d["resid"].std()),
                indicators=rows, large_miss_threshold=LARGE_MISS,
                n_large_miss=int(miss.sum()), large_miss_profile=prof,
                note="County means. Folds grouped by state. Residual is "
                     "observed minus predicted out of fold.")
    json.dump(blob, open(OUT_JSON, "w"), indent=1)
    print(f"\n  [saved] {OUT_JSON}")
    C.write_manifest("28_residual_and_social_determinants.py",
                     inputs=[C.OOF_A, C.FRAME, C.FEATURE_GATE_A],
                     outputs=[OUT_JSON, OUT_CSV],
                     settings=dict(MIN_COUNTY_COVERAGE=MIN_COUNTY_COVERAGE,
                                   LARGE_MISS=LARGE_MISS))


if __name__ == "__main__":
    main()
