"""
14_regional_invariance_of_turning_points.py
Does a located turning point move when the country is cut in two?

A turning point located on a national panel can be an artefact of how the
variable is distributed across the country rather than a property of the
fitted response. This step splits the attributed county-years at the median
longitude or at the 100th meridian, relocates each crossing on each side with
the estimator of script 06 (same bins, same rule, same attribution surface),
and reports the west-minus-east shift, in units and as a share of the
variable's observed range (Sections 2.9 and 3.5, Figure 7C).

This is the final-paper copy of 14_information_theory_and_regional_invariance.py.
Its functions are unchanged; the mutual-information census, which the final
paper no longer reports, is not included.

READS   derived/shap_dependence_candidates.csv     (script 05)
        conus_counties_5070.gpkg                   (county centroids)
WRITES  derived/threshold_regional_invariance_all.json        (median longitude)
        derived/threshold_regional_invariance_100W_all.json   (100th meridian)

    python3 14_regional_invariance_of_turning_points.py --all-candidates
    python3 14_regional_invariance_of_turning_points.py --all-candidates --split meridian
"""
from __future__ import annotations

# ===========================================================================
#  SETTINGS
# ===========================================================================
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
    ap.add_argument("--split", choices=("median", "meridian"), default=SPLIT,
                    help="median longitude, or the 100th meridian west")
    ap.add_argument("--all-candidates", action="store_true",
                    help="locate every threshold candidate, not only the "
                         "three nocturnal variables")
    a = ap.parse_args()

    C.banner("14  regional invariance of the turning points")
    C.seed_everything()
    out = regional_invariance(a)
    C.write_manifest(__file__, [], out,
                     notes=f"split={a.split}, all_candidates={a.all_candidates}")


if __name__ == "__main__":
    main()
