"""
35_check_method_statements.py -- check the statements about methods and data
that the manuscript makes in words, against the code and the result files.

The constants builder checks every sentence that describes a NUMBER in words
("weakly clustered", "barely move"). This step checks the other kind: sentences
that say HOW something was done ("weighted by inverse distance", "dropped the
later one", "a crossing was found in every resample") or WHAT the data show
without printing a number ("the Virginia units are independent cities"). Each
check reads the code that did the work, or a result file, and never refits.

    python3 35_check_method_statements.py

Writes derived/method_statement_checks.json. A failed check prints FAIL and the
section to reread; the step exits with status 1 if any check fails.

Reads: the pipeline code (../pipeline), the step scripts in this folder, the
result files in derived/, and studyA_reference_frame.parquet (for the terrain
check only). Runs in about ten seconds.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _config as C

HERE = Path(__file__).resolve().parent
PIPE = C.PAPER / "pipeline"
OUT = C.DERIVED / "method_statement_checks.json"

RESULTS = []


def check(section, statement, ok, evidence):
    RESULTS.append(dict(section=section, statement=statement,
                        status="PASS" if ok else "FAIL", evidence=evidence))
    print(f"  {'PASS' if ok else 'FAIL':<5} {section:<6} {statement}")
    print(f"        {evidence}")


def src(name):
    p = HERE / name
    if not p.exists():
        p = PIPE / name
    return p.read_text()


def js(name):
    return json.load(open(C.DERIVED / name))


def step(prefix):
    return src(next(p.name for p in HERE.glob(f"{prefix}_*.py")))


def main():
    C.banner("35  Statements about methods and data, checked against the code")

    # ------------------------------------------------------------ 2.5 filter
    core = src("pipeline_core_v2.py")
    pats = re.search(r"PROTECT_PATTERNS = \[(.*?)\]", core, re.S).group(1)
    want = {"heat indices": ["heat_?index", "humidex", "wbgt"],
            "wet-bulb": ["wet_?bulb"], "wind chill": ["wind_?chill"],
            "dew point and humidity": ["tdmean", "dew_?point", "specific_humidity"],
            "total column water vapour": ["total_column_water"],
            "minimum temperature": ["prism_tmin", "tmin_min"],
            "night-time LST": ["lst_night"], "diurnal range": ["diurnal"],
            "night cooling": ["night_cool"], "vapour-pressure deficit": ["vpd"],
            "hot nights and days": ["hot_?night", "days_tmin_over", "days_tmax_over"],
            "formaldehyde, NO2, ozone, PM2.5": ["formaldehyde", "nitrogen_dioxide",
                                                "ozone", "pm25_"]}
    miss = [k for k, v in want.items() if not all(x in pats for x in v)]
    check("2.5", "The protected list covers the patterns the text names",
          not miss, f"PROTECT_PATTERNS in pipeline_core_v2.py; missing: {miss or 'none'}")
    check("2.5", "Of two unprotected columns the filter drops the later one, "
          "and of a mixed pair the unprotected one",
          "for b in cols[i + 1:]" in core
          and "victim = b if b not in protected else a" in core,
          "redundancy_filter: loops a before b, victim = b unless b is protected")
    check("2.5", "The correlation cut is 0.98 and the filter never sees the outcome",
          "corr_threshold=0.98" in core and "never sees the target" in core,
          "redundancy_filter(X, corr_threshold=0.98, ...): X only")
    check("2.5", "The R2 interval resamples county-years with the predictions held fixed",
          "def bootstrap_r2_ci(y, pred" in core
          and "r2_score(y[b], pred[b])" in core,
          "bootstrap_r2_ci: rows drawn with replacement, no refit")

    # ---------------------------------------------------- 2.6 baselines, brackets
    s02 = step("02")
    check("2.6", "Five brackets 60-70, 70-74, 74-76.5, 76.5-79, 79-95, each including its lower bound",
          "BRACKET_EDGES = [60, 70, 74, 76.5, 79, 95]" in s02 and "right=False" in s02,
          "step 02: BRACKET_EDGES and pd.cut(..., right=False)")
    s03 = step("03")
    check("2.6", "Nearest-county baseline: same year, four nearest training counties, "
          "weighted by inverse distance",
          "KNN_NEIGHBOURS = 4" in s03 and "KNN_SAME_YEAR = True" in s03
          and "w = 1.0 / np.maximum(dist, 1e-6)" in s03
          and "itr = tr[yr[tr] == v]" in s03,
          "step 03 b_knn: training rows of the same year, haversine distance, 1/d weights")

    # ---------------------------------------------------------- 2.7 attribution
    s23 = step("23")
    inf = js("influence_drop.json")["topk_overlap"]
    check("2.7, 3.2", "The influence refit drops the counties with the largest mean absolute "
          "out-of-fold residual and is attributed by path-dependent TreeSHAP; "
          "the tail form of NCE ranks first in it",
          '(o[tc] - o[pc]).abs()' in s23 and "path dependent" in inf["attribution"]
          and inf["refit_top"][0] == C.NCE_TAIL,
          f"step 23; influence_drop.json: arm {inf['arm']}, top {inf['refit_top'][0]}, "
          f"{inf['shared']} of top {inf['topk']} shared")

    # ------------------------------------------------------ 2.9 lifetime tests
    s25 = step("25")
    check("2.9", "The class comparison is a one-sided Mann-Whitney test by species",
          'alternative="greater"' in s25 and "stats.mannwhitneyu" in s25,
          "step 25: mannwhitneyu(short, long, alternative='greater')")
    check("2.9", "Each lifetime is multiplied by ten raised to an independent uniform draw",
          "10 ** rng.uniform(-VALUE_SPREAD_DEX, VALUE_SPREAD_DEX, len(sp))" in s25,
          "step 25: one draw per species, VALUE_SPREAD_DEX each side")
    lt = js("lifetime_literature_test.json")
    per = pd.DataFrame(lt["species_table"])
    long3 = per[per["lifetime_days"] >= 21]
    ok = len(long3) == 3 and bool((long3["median_ratio"] < 1).all())
    ev = ", ".join(f"{r['species']} {r['lifetime_days']:g} d, median ratio "
                   f"{r['median_ratio']:.2f}" for _, r in long3.iterrows())
    check("3.3", "The three species that persist three weeks or longer all favoured the mean",
          ok, ev)

    # --------------------------------------------------- 2.10 turning points
    s06 = step("06")
    check("2.10", "The bootstrap resamples the attributed county-years and does not refit "
          "or redraw the background",
          "idx = rng.integers(0, n, n)" in s06
          and "crossing(x[idx], y[idx]" in s06 and ".fit(" not in s06,
          "step 06 analyse(): rows of the one attributed sample drawn with replacement")
    check("2.10", "The range is that between the 1st and 99th percentiles; 45 bins; "
          "a smoothing spline through the bin medians",
          "TRIM_PCT = (1, 99)" in s06 and "UnivariateSpline" in s06
          and 'BIN_STATISTIC = "median"' in s06,
          "step 06: TRIM_PCT, UnivariateSpline, BIN_STATISTIC")
    tp = js("turning_points.json")
    res = tp["results"] if isinstance(tp["results"], list) else list(tp["results"].values())
    found = [r["found_frac"] for r in res]
    width = [r["width_frac"] for r in res]
    check("3.4", "Neither numerical limit decided any candidate: a crossing in every "
          "resample, every interval narrower than 40% of the range",
          min(found) == 1.0 and max(width) < 0.4,
          f"turning_points.json: found_frac min {min(found):.3f}; "
          f"width_frac max {max(width):.3f} ({len(res)} candidates)")
    stable = {r["feature"] for r in res if r.get("stable", r.get("passes"))}
    four = {"eng2_night_cooling_efficiency", "eng2_wetbulb_hot", "prism_tmin_min",
            "usda_cropland_usda_corn_pct"}
    check("3.4", "Of the other six candidates four pass outright, night-time LST fails "
          "and PM2.5 passes only formally",
          four <= stable and "lst_night_1km_mean" not in stable and "pm25_mean" in stable
          and len(stable) == tp["n_stable"],
          f"stable: {len(stable)} of {len(res)}")
    s14 = step("14")
    check("2.10", "The regional shift is the absolute west-minus-east difference, as a share "
          "of the 1st-to-99th-percentile range",
          'row["delta"] = abs(w - e)' in s14 and "np.nanpercentile(dep[feat], 99)" in s14,
          "step 14: delta = |w - e|, span = p99 - p1")
    nc = js("null_crossing.json")["per_variable"]
    nc = list(nc.values()) if isinstance(nc, dict) else nc
    none = sum(1 for r in nc if r["permuted_crossing_found"] == 0)
    check("3.5", "For most candidates no permuted curve crossed zero", none > len(nc) / 2,
          f"null_crossing.json: no permuted crossing for {none} of {len(nc)}")

    dep = pd.read_csv(C.DERIVED / "shap_dependence_candidates.csv")

    def span(f):
        x = dep[f].to_numpy(float); y = dep["shap_" + f].to_numpy(float)
        lo, hi = np.nanpercentile(x, (1, 99))
        m = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
        x, y = x[m], y[m]
        q = np.quantile(x, np.linspace(0, 1, 46)); q[-1] += 1e-9
        b = [np.median(y[(x >= q[i]) & (x < q[i + 1])]) for i in range(45)]
        return float(max(b) - min(b))
    big = {f: span(f) for f in ("prism_tdmean_min", C.NCE_TAIL,
                               "cams_fot_formaldehyde_above75th_percentile")}
    small = {f: span(f) for f in ("eng2_wetbulb_night", "eng2_hcho_no2_ratio")}
    check("3.4", "The night-time wet-bulb and ratio curves (Figure 6B, E) span less "
          "attribution than panels A, C and D",
          max(small.values()) < min(big.values()),
          "binned-median span (years): " + ", ".join(f"{k} {v:.4f}" for k, v in
                                                     {**big, **small}.items()))

    # ---------------------------------------------- 2.3 engineered variables
    eng = src("engineered_features_v2.py")
    check("2.3", "Night-time wet-bulb: annual-mean daily minimum with the driest-cell "
          "dew point; warm-hour form: annual-mean daily maximum with the mean dew point",
          'needs=("prism_tmin_mean", "prism_tdmean_min")' in eng
          and 'needs=("prism_tmax_mean", "prism_tdmean_mean")' in eng,
          "engineered_features_v2.py: wetbulb_night, wetbulb_hot")
    check("2.4", "A 12 degC drop gives NCE 0.40 at 30 degC and 0.60 at 20 degC",
          round(12 / 30, 2) == 0.40 and round(12 / 20, 2) == 0.60,
          "12/30 = 0.40, 12/20 = 0.60")

    rank = pd.read_csv(C.RANKING_A)["feature"].astype(str)
    era = [c for c in rank if c.startswith("era5_")]
    terra = [c for c in rank if c.startswith("terra_")]
    prism = [c for c in rank if c.startswith("prism_")]
    check("2.3", "Admitted ERA5 columns give humidity, pressure, wind, snow and terrain "
          "height; PRISM gives vapour-pressure deficit; TerraClimate has no VPD column",
          all(any(k in c for c in era) for k in ("humidity", "pressure", "wind", "snow",
                                                  "geopotential"))
          and any("vpd" in c for c in prism) and not any("vpd" in c for c in terra),
          f"{len(era)} ERA5, {len(terra)} TerraClimate, {len(prism)} PRISM columns admitted")

    frame = pd.read_parquet(C.FRAME, columns=[C.GROUP, "prism_tdmean_min",
                                               "prism_tdmean_mean", "dem_p10", "dem_p90"])
    panel = set(pd.read_csv(C.DERIVED / "residual_social_determinants_counties.csv")
                ["fips"].astype(int))
    g = frame.groupby(C.GROUP).mean()
    g = g[g.index.astype(int).isin(panel)]
    from scipy.stats import spearmanr
    r_hi = spearmanr(g["prism_tdmean_min"], g["dem_p90"], nan_policy="omit").correlation
    r_gap = spearmanr(g["prism_tdmean_mean"] - g["prism_tdmean_min"],
                      g["dem_p90"] - g["dem_p10"], nan_policy="omit").correlation
    check("2.3, 4.3", "The driest cell is often on a county's highest ground",
          r_hi < -0.5 and r_gap > 0.5,
          f"{len(g)} counties: Spearman(min dew point, 90th-pct elevation) {r_hi:.2f}; "
          f"Spearman(mean minus min dew point, relief p90-p10) {r_gap:.2f}")

    # ------------------------------------------------------- 2.11 / 3.6 error
    s13 = step("13")
    check("2.11", "Moran's I is the slope of the neighbours' mean residual on the county's "
          "own (row-standardised, four nearest)",
          "lag = vals[nb].mean(axis=1)" in s13 and "np.sum(vals * lag) / denom" in s13,
          "step 13 residual_structure: I = sum(z * lag) / sum(z^2)")
    cty = pd.read_csv(C.DERIVED / "residual_social_determinants_counties.csv")
    rs = js("residual_social_determinants.json")
    miss = cty[cty["resid"] <= -rs["large_miss_threshold"]].copy()
    f5 = miss["fips"].astype(int)
    va = f5[(f5 // 1000 == 51)]
    va_city = va[(va % 1000) >= 500]
    others = {24510: "Baltimore", 29510: "St. Louis", 32510: "Carson City"}
    check("3.6", "Among the overpredicted counties, the Virginia units, Baltimore, "
          "St. Louis and Carson City are independent cities",
          len(miss) == rs["n_large_miss"] and len(va) == len(va_city) > 0
          and set(others) <= set(f5),
          f"{len(miss)} counties overpredicted by {rs['large_miss_threshold']:g} years or "
          f"more; Virginia units {sorted(va.tolist())} (all FIPS >= 51500); "
          + ", ".join(f"{v} {k}" for k, v in others.items()))
    sign = {r["column"]: r["rho"] for r in rs["indicators"]}
    neg = [k for k in sign if any(w in k for w in ("pov", "unemploy", "uninsured", "disab"))]
    pos = [k for k in sign if any(w in k for w in ("income", "bachelor"))]
    check("3.6", "Overprediction where poverty, unemployment, uninsurance and disability "
          "are high; underprediction where income and education are high",
          len(neg) == 4 and len(pos) == 2 and all(sign[k] < 0 for k in neg)
          and all(sign[k] > 0 for k in pos),
          "Spearman rho: " + ", ".join(f"{k.replace('sdoh_', '')} {v:+.2f}"
                                       for k, v in sign.items()))

    n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
    json.dump(dict(checks=RESULTS, n_checks=len(RESULTS), n_fail=n_fail),
              open(OUT, "w"), indent=1)
    print(f"\n  {len(RESULTS) - n_fail} of {len(RESULTS)} statements match the code and "
          f"the result files")
    print(f"  [saved] {OUT}")
    C.write_manifest(__file__, [C.RANKING_A, C.FRAME], [OUT])
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
