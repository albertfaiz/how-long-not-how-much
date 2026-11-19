"""
12_fields_with_several_summaries.py
How far apart the summaries of one field are in attribution.

Every admitted column name carries its field and its statistic
("prism_tdmean_min" is the field prism_tdmean summarised by its minimum).
This step groups the admitted columns of the production ranking by field,
keeps the fields with more than one admitted summary, and for each field
divides the attribution of its most attributed summary by that of its least
attributed one. The final paper quotes the median of that spread over all
such fields (Section 3.2).

This is the final-paper copy of 12_summary_statistic_census_and_matched_refit.py.
The functions split_name and census are unchanged; the census test and the
matched refit, which the final paper no longer reports, are not included.

READS   results_studyA_*/tables/A_shap_ranking.csv   (the admitted columns)
WRITES  derived/summary_statistic_census.csv          (one row per field)
        derived/summary_statistic_census.json         (median and largest spread)

    python3 12_fields_with_several_summaries.py
"""
from __future__ import annotations

STAT_CLASS = {
    "mean": "central", "p50": "central", "median": "central",
    "min": "extreme", "max": "extreme",
    "p05": "extreme", "p10": "extreme", "p25": "extreme",
    "p70": "extreme", "p75": "extreme", "p90": "extreme", "p95": "extreme",
    "stddev": "dispersion", "std": "dispersion", "range": "dispersion",
    "iqr": "dispersion", "cv": "dispersion",
}

MIN_STATS_PER_FIELD = 2

import json
import os
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as C

TOKEN = re.compile(r"^(?P<field>.+)_(?P<stat>[a-z0-9]+)$")

def split_name(name: str):
    m = TOKEN.match(name)
    if not m: return None
    stat = m.group("stat")
    cls = STAT_CLASS.get(stat)
    if cls is None: return None
    return m.group("field"), cls, stat

def census(rank: pd.DataFrame) -> pd.DataFrame:
    parsed = []
    for _, r in rank.iterrows():
        s = split_name(r.feature)
        if s is None: continue
        field, cls, stat = s
        parsed.append(dict(field=field, stat=stat, cls=cls, feature=r.feature, shap=float(r.mean_abs_shap),
                           rank=int(r["rank"]), modality=r.modality, label=r.get("label", r.feature)))
    p = pd.DataFrame(parsed)
    if p.empty: raise SystemExit("No feature name in the ranking carries a statistic token.")

    rows = []
    for field, g in p.groupby("field"):
        if len(g) < MIN_STATS_PER_FIELD: continue
        cen = g[g.cls == "central"]
        non = g[g.cls != "central"]
        best = g.loc[g.shap.idxmax()]
        worst = g.loc[g.shap.idxmin()]
        rows.append(dict(
            field=field, n_stats=len(g), modality=g.modality.iloc[0],
            has_central=len(cen) > 0, has_noncentral=len(non) > 0,
            best_stat=best.stat, best_cls=best.cls, best_shap=best.shap, best_rank=best["rank"], best_label=best.label,
            worst_stat=worst.stat, worst_shap=worst.shap, worst_rank=worst["rank"],
            spread=float(best.shap / worst.shap) if worst.shap > 0 else np.nan,
            central_stat=cen.loc[cen.shap.idxmax()].stat if len(cen) else "",
            central_shap=float(cen.shap.max()) if len(cen) else np.nan, central_rank=int(cen["rank"].min()) if len(cen) else -1,
            noncentral_stat=non.loc[non.shap.idxmax()].stat if len(non) else "",
            noncentral_shap=float(non.shap.max()) if len(non) else np.nan, noncentral_rank=int(non["rank"].min()) if len(non) else -1,
        ))
    out = pd.DataFrame(rows).sort_values("best_shap", ascending=False)
    both = out.has_central & out.has_noncentral
    out["comparable"] = both
    out["noncentral_wins"] = np.where(both, out.noncentral_shap > out.central_shap, False)
    out["ratio"] = np.where(both, out.noncentral_shap / out.central_shap, np.nan)
    return out


def main():
    C.seed_everything()
    rank = C.load_table("shap_ranking")
    if "mean_abs_shap" not in rank.columns:
        raise SystemExit("The ranking has no mean_abs_shap column")
    cen = census(rank)
    C.banner("fields with more than one admitted summary")
    print(f"  fields                                   {len(cen)}")
    print(f"  median spread, most over least attributed {cen.spread.median():.2f}x")
    C.DERIVED.mkdir(parents=True, exist_ok=True)
    p_csv = C.DERIVED / "summary_statistic_census.csv"
    cen.to_csv(p_csv, index=False)
    p_json = C.DERIVED / "summary_statistic_census.json"
    json.dump(dict(n_fields=int(len(cen)),
                   median_spread=float(cen.spread.median()),
                   max_spread=float(cen.spread.max())),
              open(p_json, "w"), indent=1)
    C.write_manifest(__file__, [], [p_csv, p_json], notes="fields and spreads")


if __name__ == "__main__":
    main()
