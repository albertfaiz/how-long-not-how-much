"""
14_information_theory_and_regional_invariance.py
Two independent checks on the summary-statistic result, neither of which uses
the quantity the result was measured with.

WHY THIS SCRIPT EXISTS
----------------------
Script 12 reports that across the fields carrying two kinds of summary, the
non-central summary receives more attribution than the central one, and that
swapping every field from one kind to the other barely changes out-of-fold
skill. Both of those are measured inside a fitted model. Two objections follow
and each has a check here.

1.  MUTUAL INFORMATION.  TreeSHAP divides credit between correlated columns,
    and two summaries of one field are correlated, so the attribution gap could
    be a property of that division rather than of the data. Mutual information
    between a single column and the outcome uses no model at all. If the gap is
    about information content, the non-central summary should carry more
    mutual information; if it is about what a model does with two near
    equivalent columns, it should not.

2.  REGIONAL INVARIANCE.  A turning point located on a national panel can be an
    artefact of how the climate variable is distributed across the country
    rather than a property of the response. Splitting the panel at the median
    longitude puts the arid west on one side and the humid east on the other,
    with very different distributions of every moisture variable. A turning
    point that is a physiological limit should appear in the same place on both
    sides; one that is an artefact of the national distribution should not.

The crossing is located with the estimator of script 06, imported from it, at
the same bin count and under the same rule, on the same attribution surface
script 06 used. The regional values are therefore directly comparable with the
national values in the threshold table, which values from a different estimator
would not be.

READS   derived/summary_statistic_census.csv       (script 12)
        derived/shap_dependence_candidates.csv     (script 05)
        the modelling panel                        (mutual information only)
        conus_counties_5070.gpkg                   (county centroids)
WRITES  derived/information_theory_census.csv
        derived/information_theory_census.json
        derived/threshold_regional_invariance.json
TIME    8 to 20 minutes, almost all of it in the mutual information estimator

    python3 14_information_theory_and_regional_invariance.py
    python3 14_information_theory_and_regional_invariance.py --skip-mi
    python3 14_information_theory_and_regional_invariance.py --mi-sample 20000
"""
from __future__ import annotations

# ===========================================================================
#  SETTINGS
# ===========================================================================
# Mutual information is estimated with the k-nearest-neighbour method of
# Kraskov and co-authors as implemented in scikit-learn. It is O(n log n) per
# column and the panel has 52,309 rows, so a sample keeps the whole script
# inside a coffee break. 0 uses every row.
MI_SAMPLE = 0
MI_NEIGHBOURS = 3            # scikit-learn's default; raising it smooths

# The three variables that describe one nocturnal physical state. Names are the
# column names in the panel, not labels.
TRIPLET = [
    "prism_tdmean_min",          # annual minimum dew point
    "eng2_wetbulb_night",        # nocturnal wet-bulb temperature
    "eng_night_cooling_eff",     # night-time cooling efficiency, tail form
]

# Where the panel is cut in two. "median" splits at the median longitude of the
# county-years, which puts about half the observations on each side; a fixed
# meridian splits at a named line and leaves the halves unequal.
SPLIT = "median"             # "median" | "meridian"
MERIDIAN = -100.0            # degrees east, used only when SPLIT="meridian"

# A region must hold at least this many county-years for a crossing located in
# it to be reported. Below it the estimator is being asked for more than the
# data supports.
MIN_REGION_ROWS = 2000
# ===========================================================================

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as C

HERE = os.path.dirname(os.path.abspath(__file__))


def canonical_crossing():
    """The crossing estimator of script 06, imported rather than reimplemented.

    A second implementation of this estimator would give values that differ in
    the second decimal from the ones in the threshold table, and no reader
    could tell whether the difference was regional or procedural.
    """
    import importlib.util
    path = os.path.join(HERE, "06_bootstrap_turning_points_and_gate.py")
    if not os.path.exists(path):
        raise SystemExit(
            f"06_bootstrap_turning_points_and_gate.py is not beside this file.\n"
            f"    This script locates crossings with that script's estimator so\n"
            f"    the regional values are comparable with the threshold table.\n"
            f"    Copy the whole analysis folder rather than single scripts.")
    spec = importlib.util.spec_from_file_location("_s06", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
#  1. Mutual information: the same question with no model in it
# ---------------------------------------------------------------------------
def pick_columns(cen_row, columns):
    """(central column, non-central column) for one field, or None.

    The census stores the best statistic of each kind separately. An earlier
    version of this check used the field's overall best statistic as the
    non-central one, which for the nine fields whose best statistic IS central
    compared a column with itself and recorded a tie. Those nine are the fields
    where the comparison is most interesting, so the bug mattered.
    """
    field = cen_row["field"]
    c_stat = str(cen_row.get("central_stat") or "")
    n_stat = str(cen_row.get("noncentral_stat") or "")
    if not c_stat or not n_stat:
        return None
    c_col, n_col = f"{field}_{c_stat}", f"{field}_{n_stat}"
    if c_col == n_col:
        return None
    if c_col not in columns or n_col not in columns:
        return None
    return c_col, n_col


def mutual_information_census(panel, args):
    from sklearn.feature_selection import mutual_info_regression

    print("\n" + "=" * 74)
    print("  1. MUTUAL INFORMATION, one column at a time, no model")
    print("=" * 74)

    cen = C.load_table("summary_statistic_census")
    cen = cen[cen["comparable"].astype(bool)].copy()
    X, y = panel["X"], panel["y"]

    n = len(y)
    idx = np.arange(n)
    if args.mi_sample and args.mi_sample < n:
        rng = C.seed_everything()
        idx = np.sort(rng.choice(n, size=args.mi_sample, replace=False))
    print(f"  {len(cen)} comparable fields, {len(idx):,} of {n:,} county-years")

    t0 = time.time()
    rows, skipped = [], []
    for _, r in cen.iterrows():
        pick = pick_columns(r, X.columns)
        if pick is None:
            skipped.append(r["field"])
            continue
        c_col, n_col = pick
        sub = X[[c_col, n_col]].iloc[idx]
        sub = sub.fillna(sub.median())
        mi = mutual_info_regression(sub, y[idx], random_state=C.SEED,
                                    n_neighbors=MI_NEIGHBOURS)
        c_mi, n_mi = float(mi[0]), float(mi[1])
        rows.append(dict(
            field=r["field"], modality=r.get("modality", ""),
            central_col=c_col, central_mi=c_mi,
            noncentral_col=n_col, noncentral_mi=n_mi,
            noncentral_wins_mi=bool(n_mi > c_mi),
            mi_ratio=float(n_mi / c_mi) if c_mi > 0 else np.nan,
            shap_central=float(r["central_shap"]),
            shap_noncentral=float(r["noncentral_shap"]),
            noncentral_wins_shap=bool(r["noncentral_wins"]),
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(
            "No field had both a central and a non-central column in the "
            "panel.\n    The census and the panel disagree about column "
            "names; rerun script 12.")

    k = int(df.noncentral_wins_mi.sum())
    nn = len(df)
    med = float(df.mi_ratio.median())

    # the same exact sign test script 12 uses, so the two counts are comparable
    s06 = None
    from math import comb
    probs = [comb(nn, i) / 2 ** nn for i in range(nn + 1)]
    p = float(sum(pr for pr in probs if pr <= probs[k] + 1e-12))

    print(f"\n  non-central carries more mutual information in {k} of {nn}")
    print(f"  exact two-sided sign test                 p = {p:.3g}")
    print(f"  median ratio, non-central over central    {med:.2f}")
    if skipped:
        print(f"  {len(skipped)} field(s) skipped, no distinct column pair: "
              f"{', '.join(skipped[:6])}{' ...' if len(skipped) > 6 else ''}")

    # how the two measurements agree, field by field
    agree = int((df.noncentral_wins_mi == df.noncentral_wins_shap).sum())
    shap_k = int(df.noncentral_wins_shap.sum())
    print(f"\n  for comparison, on the same {nn} fields:")
    print(f"      attribution favours the non-central summary in {shap_k}")
    print(f"      mutual information favours it in                {k}")
    print(f"      the two agree on                                {agree}")
    print("\n  Reading: attribution is measured inside a fitted model holding")
    print("  the other variables; mutual information is marginal and model")
    print("  free. They answer different questions, and a gap between them is")
    print("  evidence about where the attribution difference comes from, not")
    print("  a failure of either.")
    print(f"  Time: {time.time() - t0:.0f}s")

    p_csv = C.DERIVED / "information_theory_census.csv"
    df.to_csv(p_csv, index=False)
    blob = dict(n_fields=nn, n_noncentral_wins_mi=k,
                n_noncentral_wins_shap=shap_k, n_agree=agree,
                sign_test_p=p, median_mi_ratio=med,
                mi_sample=int(len(idx)), mi_neighbours=MI_NEIGHBOURS,
                skipped=skipped)
    p_json = C.DERIVED / "information_theory_census.json"
    json.dump(blob, open(p_json, "w"), indent=1)
    print(f"[saved] {p_csv}")
    print(f"[saved] {p_json}")
    return [p_csv, p_json]


# ---------------------------------------------------------------------------
#  2. Regional invariance of the located turning points
# ---------------------------------------------------------------------------
def county_longitude():
    """Longitude of each county centroid, in degrees, keyed by FIPS."""
    import geopandas as gpd
    gpkg = os.path.join(HERE, "conus_counties_5070.gpkg")
    if not os.path.exists(gpkg):
        raise SystemExit(
            f"The county boundaries are missing:\n    {gpkg}\n"
            f"    This check needs county centroids to split the panel.")
    g = gpd.read_file(gpkg, layer="county")
    if "fips" not in g.columns:
        g["fips"] = pd.to_numeric(g["STATE"] + g["COUNTY"], errors="coerce")
    cen = g.geometry.centroid.to_crs("EPSG:4326")
    out = pd.DataFrame({"fips": pd.to_numeric(g["fips"], errors="coerce"),
                        "lon": cen.x.to_numpy()})
    return out.dropna().astype({"fips": int}).set_index("fips")["lon"]


def regional_invariance(args):
    print("\n" + "=" * 74)
    print("  2. REGIONAL INVARIANCE of the located turning points")
    print("=" * 74)

    s06 = canonical_crossing()
    print(f"  estimator: script 06, rule '{s06.CROSSING_RULE}', bin statistic "
          f"'{s06.BIN_STATISTIC}', {C.N_BINS} bins")

    dep = C.load_table("shap_dependence_candidates")
    if "fips" not in dep.columns:
        raise SystemExit(
            "shap_dependence_candidates.csv has no fips column, so the rows "
            "cannot be placed geographically. Rerun script 05.")
    lon = county_longitude()
    dep = dep.assign(lon=dep["fips"].map(lon))
    dep = dep[dep["lon"].notna()]

    split = getattr(args, "split", None) or SPLIT
    cut = float(dep["lon"].median()) if split == "median" else MERIDIAN
    west = dep["lon"] < cut
    east = ~west
    print(f"  split at {cut:.2f} degrees east: "
          f"{int(west.sum()):,} county-years west, {int(east.sum()):,} east")

    results = {}
    # The three nocturnal variables are the test the paper is built on. With
    # --all-candidates every variable the threshold table reports is located in
    # both halves as well, so a reader can see which turning points are
    # climate-invariant and which are not, instead of only the ones that are.
    feats = TRIPLET
    if getattr(args, "all_candidates", False):
        feats = [c for c in dep.columns
                 if f"shap_{c}" in dep.columns and c not in ("fips",)]
    for feat in feats:
        if feat not in dep.columns or f"shap_{feat}" not in dep.columns:
            print(f"\n  {feat}: not in the dependence sample, skipped")
            results[feat] = dict(reason="absent from the dependence sample")
            continue

        def locate(mask):
            d = dep[mask]
            if len(d) < MIN_REGION_ROWS:
                return None
            x = d[feat].to_numpy(float)
            y = d["shap_" + feat].to_numpy(float)
            lo, hi = np.percentile(x[np.isfinite(x)], s06.TRIM_PCT)
            return s06.crossing(x, y, C.N_BINS, lo, hi)

        all_, w, e = locate(dep.index == dep.index), locate(west), locate(east)
        row = dict(national=all_, west=w, east=e,
                   n_west=int(west.sum()), n_east=int(east.sum()))
        if w is not None and e is not None:
            span = float(np.nanpercentile(dep[feat], 99)
                         - np.nanpercentile(dep[feat], 1))
            row["delta"] = abs(w - e)
            row["delta_pct_of_range"] = 100 * abs(w - e) / span if span else None
            row["span"] = span
        results[feat] = row

        print(f"\n  {feat}")
        for k in ("national", "west", "east"):
            v = row.get(k)
            print(f"      {k:<9} {'none located' if v is None else f'{v:.3f}'}")
        if "delta" in row:
            print(f"      west minus east: {row['delta']:.3f} "
                  f"({row['delta_pct_of_range']:.1f} per cent of the "
                  f"observed range)")

    located = [r for r in results.values() if r.get("delta") is not None]
    if located:
        worst = max(r["delta_pct_of_range"] for r in located)
        print(f"\n  largest regional disagreement across the {len(located)} "
              f"variables located in both halves: {worst:.1f} per cent of the "
              f"observed range")
    # The median split and the 100th-meridian split are both kept, in separate
    # files, because they answer different questions: the median gives two
    # halves of equal size, the meridian is the classical arid-humid divide.
    name = ("threshold_regional_invariance.json" if split == "median"
            else "threshold_regional_invariance_100W.json")
    if getattr(args, "all_candidates", False):
        name = name.replace(".json", "_all.json")
    p = C.DERIVED / name
    json.dump(dict(split=split, cut_degrees_east=cut,
                   estimator=dict(rule=s06.CROSSING_RULE,
                                  bin_statistic=s06.BIN_STATISTIC,
                                  n_bins=C.N_BINS,
                                  trim_pct=list(s06.TRIM_PCT)),
                   results=results), open(p, "w"), indent=1, default=str)
    print(f"[saved] {p}")
    return [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mi", action="store_true",
                    help="leave out the mutual information census, which is "
                         "the slow half")
    ap.add_argument("--skip-regional", action="store_true")
    ap.add_argument("--split", choices=("median", "meridian"), default=SPLIT,
                    help="median longitude, or the 100th meridian west")
    ap.add_argument("--all-candidates", action="store_true",
                    help="locate every threshold candidate, not only the "
                         "three nocturnal variables")
    ap.add_argument("--mi-sample", type=int, default=MI_SAMPLE,
                    help="county-years used for mutual information; 0 is all")
    a = ap.parse_args()

    C.banner("14  mutual information and regional invariance")
    C.seed_everything()
    out = []
    if not a.skip_mi:
        panel = C.load_panel()
        out += mutual_information_census(panel, a)
    if not a.skip_regional:
        out += regional_invariance(a)
    C.write_manifest(__file__, [], out,
                     notes=f"mi_sample={a.mi_sample}, split={a.split}, "
                           f"all_candidates={a.all_candidates}")
    print("\n" + "=" * 74)
    print("  next:  python3 08_build_manuscript_constants.py")
    print("=" * 74)


if __name__ == "__main__":
    main()
