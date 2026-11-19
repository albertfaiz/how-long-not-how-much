#!/usr/bin/env python3
# =============================================================================
#  pipeline_A_moisture_v2.py
#  STUDY A  ->  Papers 1 (flagship) and 2 (nocturnal discovery)
#
#  Environmental only. No socioeconomic, no census, no health-outcome columns.
#  Window 2003-2019, set by the first year of ERA5 and CAMS availability.
#
#  Stages
#    1  load superset, add engineered layer, save reference frame
#    2  temporal-coverage gate at 95 percent, environment-only scope
#    3  unsupervised redundancy filter with protected physiological features
#    4  learner tournament under identical county-grouped protocol
#    5  refit winner, bootstrap interval, out-of-fold predictions
#    6  interventional TreeSHAP on the winner, out of fold
#    7  discovery analyses: group hierarchy, thermal-index audit, dew-point
#       profile, threshold with bootstrap, orthogonality, era stability,
#       modality leave-one-out, error structure, residual spatial statistic
#    8  optional forward prediction into 2020-2023
#
#  Every stage writes artefacts; nothing downstream needs to refit a model.
#  Run:  python pipeline_A_moisture_v2.py --csv Final_ML_Ready_Dataset.csv --gpu
# =============================================================================

from __future__ import annotations
import argparse, json, time
import numpy as np
import pandas as pd

import pipeline_core_v2 as core
from pipeline_core_v2 import (TARGET, GROUP_COL, YEAR_COL, Artifacts,
                              modality_of, is_protected)

WINDOW = (2003, 2019)
FORWARD = (2020, 2023)
TAG = "studyA"

# leading candidates for the nocturnal-moisture analyses
DP_MIN = "prism_tdmean_min"
T_MIN = "prism_tmin_min"


# ---------------------------------------------------------------------------
#  physical grouping used for the attribution hierarchy
# ---------------------------------------------------------------------------
def physical_group(f: str) -> str:
    import re
    g = [
        (r"night_cool|nightcool|lst_night|wetbulb_night|rel_humidity_night|"
         r"nocturnal|tmin_min|days_tmin_over|urban_moisture_trap|"
         r"surface_air_night_offset", "Nocturnal thermal"),
        (r"tdmean|dew_?point|wetbulb|wet_bulb|humidex|rel_humidity|"
         r"specific_humidity|absolute_humidity|vapour_pressure|"
         r"total_column_water|column_moisture", "Atmospheric moisture"),
        (r"vpd|aridity|evaporative_fraction|terra_pet|terra_aet|terra_def|"
         r"ventilation_index", "Evaporative demand"),
        (r"cams_|hcho|no2|ozone|oxidant|aerosol|aod|pm25|pm10|pm1|"
         r"photochem|biogenic|combustion|acid_precursor|black_carbon|"
         r"radical|nox_reservoir|particulate", "Air quality and composition"),
        (r"tmax|tmean|prism_tmin|heat_index|wbgt|discomfort|degree_days|"
         r"days_tmax_over|diurnal|lst_day|summer_t|winter_t|era5_2m_temperature|"
         r"era5_temperature|era5_fot_temperature|day_night_lst", "Heat and temperature"),
        (r"ghm_|impervious|ntl_|urban|econ_", "Built environment"),
        (r"ndvi|evi|savi|ndmi|meta_canopy|leaf_area|vegetation_heat", "Vegetation"),
        (r"soil_|dem_|geopotential|topo|elevation", "Soil and terrain"),
        (r"jrc_|water|ndwi|snow|terra_ro|terra_swe|runoff", "Water and hydrology"),
        (r"wind|era5_10m_wind", "Wind"),
        (r"pdsi|drought", "Drought"),
        (r"ppt|precip", "Precipitation"),
        (r"usda_|live_|crop", "Agriculture and livestock"),
    ]
    for pat, label in g:
        if re.search(pat, f, re.I):
            return label
    return "Other"


def smooth_xy(x, y, frac=0.3):
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        s = lowess(y, x, frac=frac, return_sorted=True)
        return s[:, 0], s[:, 1]
    except Exception:
        o = np.argsort(x)
        xs, ys = np.asarray(x)[o], np.asarray(y)[o]
        k = max(15, len(xs) // 25)
        return xs, pd.Series(ys).rolling(k, center=True, min_periods=1).mean().values


def zero_crossing(x, y, n_bins=40):
    """Value at which the binned mean response passes through zero.

    Binning rather than smoothing makes each bootstrap replicate O(n) instead
    of O(n^2), which matters at 50,000+ rows. The estimate is restricted to the
    data-dense interior (5th to 95th percentile) so that a wiggle in a sparse
    tail cannot be mistaken for a threshold. That restriction is deliberate: an
    unconstrained derivative on this response returns a value far outside the
    observed range, which is an artefact rather than a finding.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 50:
        return np.nan
    lo, hi = np.percentile(x, [5, 95])
    m = (x >= lo) & (x <= hi)
    x, y = x[m], y[m]
    if len(x) < 50:
        return np.nan
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 5:
        return np.nan
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    cx = np.bincount(idx, weights=x, minlength=len(edges) - 1)
    cy = np.bincount(idx, weights=y, minlength=len(edges) - 1)
    cn = np.bincount(idx, minlength=len(edges) - 1).astype(float)
    good = cn >= 5
    if good.sum() < 5:
        return np.nan
    bx, by = cx[good] / cn[good], cy[good] / cn[good]
    sign = np.sign(by)
    cross = np.where(np.diff(sign) != 0)[0]
    if not len(cross):
        return np.nan
    j = cross[len(cross) // 2]
    denom = by[j + 1] - by[j]
    if abs(denom) < 1e-12:
        return float(bx[j])
    return float(bx[j] - by[j] * (bx[j + 1] - bx[j]) / denom)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="Final_ML_Ready_Dataset.csv")
    ap.add_argument("--outdir", default="results_studyA")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--fast", action="store_true", help="fewer trees, for a smoke test")
    ap.add_argument("--coverage", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=500,
                    help="bootstrap replicates for the threshold interval")
    ap.add_argument("--skip-era", action="store_true",
                    help="skip the era sub-model stability step (expensive)")
    ap.add_argument("--skip-loo", action="store_true",
                    help="skip modality leave-one-out (expensive)")
    ap.add_argument("--forward", action="store_true",
                    help="also predict 2020-2023 where inputs exist (no ground truth)")
    args = ap.parse_args()

    t_start = time.time()
    A = Artifacts(args.outdir, TAG)
    print("=" * 74)
    print(f"STUDY A  environment only  window {WINDOW[0]}-{WINDOW[1]}")
    print("=" * 74)

    # -- 1  load and engineer ------------------------------------------------
    df_all = core.load_master(args.csv, add_engineered=True)
    df = df_all[(df_all[YEAR_COL] >= WINDOW[0]) &
                (df_all[YEAR_COL] <= WINDOW[1])].reset_index(drop=True)
    print(f"[window] {len(df):,} rows, {df[GROUP_COL].nunique():,} counties, "
          f"{df[YEAR_COL].nunique()} years")
    A.frame("reference_frame", df)

    # -- 2  coverage gate ----------------------------------------------------
    feats, gate = core.coverage_gate(df, WINDOW, min_coverage=args.coverage,
                                     allow_sdoh=False, drop_raw_ntl=True)
    A.table("feature_gate", gate)

    # -- 3  redundancy filter -----------------------------------------------
    feats = core.redundancy_filter(df[feats])
    A.json("feature_set", dict(n_features=len(feats), features=feats))

    # -- 4  learner tournament ----------------------------------------------
    print("\n[4] learner tournament")
    table, results, best = core.run_tournament(
        df, feats, n_folds=args.folds, use_gpu=args.gpu, fast=args.fast)
    A.table("model_tournament", table)

    zoo = core.build_models(use_gpu=args.gpu, fast=args.fast)
    best_ctor, best_shap_ok = zoo[best]
    res = results[best]

    # -- 5  headline metrics -------------------------------------------------
    y = res["data"][TARGET].values
    ci = core.bootstrap_r2_ci(y, res["oof"])
    m = dict(res["metrics"]); m.update(model=best, r2_ci_lo=ci[0], r2_ci_hi=ci[1],
                                       window=list(WINDOW))
    print(f"\n[5] best model {best}: R2={m['r2']:.4f} "
          f"(95% CI {ci[0]:.4f}-{ci[1]:.4f}) MAE={m['mae']:.4f} RMSE={m['rmse']:.4f}")
    print(f"    within 1 yr {m['within_1yr']:.1f}% | within 2 yr {m['within_2yr']:.1f}%")

    oof = res["data"][[GROUP_COL, YEAR_COL, TARGET]].copy()
    oof["pred"] = res["oof"]
    oof["resid"] = oof[TARGET] - oof["pred"]
    A.table("oof_predictions", oof)
    A.table("cv_folds", res["folds"])

    # -- 6  attribution ------------------------------------------------------
    if best_shap_ok:
        print("\n[6] interventional TreeSHAP on the winning model")
        ranking, per_obs, sdata = core.interventional_shap(
            df, feats, best_ctor, n_folds=args.folds)
        ranking["group"] = ranking.feature.map(physical_group)
        A.table("shap_ranking", ranking)
        keep_cols = [c for c in [DP_MIN, T_MIN, "eng2_wetbulb_night", "eng2_wetbulb_hot",
                                 "eng2_night_cooling_efficiency", "lst_night_1km_mean",
                                 "eng2_hcho_no2_ratio", "pm25_mean"] if c in feats]
        dep = sdata.loc[per_obs.index, keep_cols].copy()
        for c in keep_cols:
            dep["shap_" + c] = per_obs[c].values
        A.table("dependence_data", dep)
        print("    top 15:")
        for _, r in ranking.head(15).iterrows():
            print(f"      #{int(r['rank']):<4d} {r.feature[:46]:<46} "
                  f"|phi|={r.mean_abs_shap:.4f}  [{r.group}]")
    else:
        ranking, per_obs = None, None
        print("\n[6] winning model does not support TreeSHAP; attribution skipped")

    # -- 7  discovery analyses ----------------------------------------------
    print("\n[7] discovery analyses")

    if ranking is not None:
        hier = (ranking.groupby("group")
                .agg(sum_shap=("mean_abs_shap", "sum"),
                     per_feature=("mean_abs_shap", "mean"),
                     n_features=("feature", "size"),
                     best_rank=("rank", "min"),
                     top_feature=("feature", "first"))
                .sort_values("sum_shap", ascending=False))
        hier["pct_of_total"] = 100 * hier.sum_shap / hier.sum_shap.sum()
        rng = np.random.default_rng(0)
        lo, hi = [], []
        for g in hier.index:
            v = ranking.loc[ranking.group == g, "mean_abs_shap"].values
            b = [rng.choice(v, len(v), replace=True).sum() for _ in range(2000)]
            lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
        hier["ci_lo"], hier["ci_hi"] = lo, hi
        A.table("group_hierarchy", hier.reset_index())
        print("  group hierarchy (per-feature order):")
        for g, r in hier.sort_values("per_feature", ascending=False).head(8).iterrows():
            print(f"    {g[:34]:<34} n={int(r.n_features):>4} "
                  f"sum={r.sum_shap:.3f} perfeat={r.per_feature:.4f}")

        FAM = {
            "Pure moisture (dew point)": r"tdmean",
            "Composite wet-bulb": r"wetbulb|wet_bulb",
            "Composite WBGT / heat index / humidex": r"wbgt|heat_index|humidex|discomfort",
            "Pure dry heat (air temperature)": r"prism_tm(ax|in|ean)|era5_2m_temperature",
            "Evaporative demand (VPD)": r"vpd",
            "Air composition (CAMS)": r"^cams_|hcho_no2",
            "Particulates": r"pm25|pm10|pm1_",
        }
        rows = []
        for fam, pat in FAM.items():
            s = ranking[ranking.feature.str.contains(pat, regex=True, case=False)]
            if not len(s):
                continue
            rows.append(dict(family=fam, n=len(s), best_rank=int(s["rank"].min()),
                             best_feature=s.iloc[0].feature,
                             best_shap=float(s.mean_abs_shap.max()),
                             sum_shap=float(s.mean_abs_shap.sum()),
                             per_feature=float(s.mean_abs_shap.mean())))
        fam = pd.DataFrame(rows).sort_values("best_shap", ascending=False)
        A.table("thermal_index_family", fam)
        print("  thermal-index family audit:")
        for _, r in fam.iterrows():
            print(f"    {r.family[:38]:<38} n={int(r.n):>3} best=#{int(r.best_rank):<4d} "
                  f"{r.best_shap:.4f}")

        prof = ranking[ranking.feature.str.startswith("prism_tdmean")].copy()
        if len(prof):
            A.table("dewpoint_profile", prof)

        s = ranking.mean_abs_shap.sort_values(ascending=False).values
        cum = np.cumsum(s) / s.sum()
        A.table("concentration", pd.DataFrame(
            {"k": np.arange(1, len(s) + 1), "cum_share": cum}))
        lor = np.cumsum(np.sort(s)) / s.sum()
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        m["attribution_gini"] = float(1 - 2 * trap(lor, dx=1 / len(s)))
        m["n_feat_50pct"] = int(np.searchsorted(cum, .5) + 1)
        m["n_feat_80pct"] = int(np.searchsorted(cum, .8) + 1)

    # threshold and orthogonality
    if per_obs is not None and DP_MIN in feats:
        x = sdata.loc[per_obs.index, DP_MIN].values
        phi = per_obs[DP_MIN].values
        ok = ~np.isnan(x) & ~np.isnan(phi)
        x, phi = x[ok], phi[ok]
        pt = zero_crossing(x, phi)
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(args.n_boot):
            s_ = rng.integers(0, len(x), len(x))
            c = zero_crossing(x[s_], phi[s_])
            if not np.isnan(c):
                bs.append(c)
        thr = dict(zero_cross=pt,
                   ci_lo=float(np.percentile(bs, 2.5)) if bs else np.nan,
                   ci_hi=float(np.percentile(bs, 97.5)) if bs else np.nan,
                   pct_above=float((df[DP_MIN] > pt).mean() * 100) if not np.isnan(pt) else np.nan,
                   n_boot=len(bs))
        es = 6.112 * np.exp(17.67 * pt / (pt + 243.5)) if not np.isnan(pt) else np.nan
        thr["abs_humidity_g_per_kg"] = float(0.622 * es / (1013.25 - es) * 1000) \
            if not np.isnan(pt) else np.nan
        A.table("dpmin_threshold", pd.DataFrame([thr]))
        print(f"  dew-point threshold: {pt:.3f} C "
              f"[{thr['ci_lo']:.3f}, {thr['ci_hi']:.3f}], "
              f"{thr['pct_above']:.1f}% of county-years above")

        if T_MIN in feats:
            raw_r = float(df[[DP_MIN, T_MIN]].corr().iloc[0, 1])
            shap_r = float(np.corrcoef(per_obs[DP_MIN], per_obs[T_MIN])[0, 1])
            orth = dict(raw_corr=raw_r, shap_corr=shap_r,
                        dp_shap_range=float(per_obs[DP_MIN].max() - per_obs[DP_MIN].min()),
                        t_shap_range=float(per_obs[T_MIN].max() - per_obs[T_MIN].min()))
            orth["range_ratio"] = orth["dp_shap_range"] / max(orth["t_shap_range"], 1e-9)
            A.table("orthogonality", pd.DataFrame([orth]))
            print(f"  orthogonality: raw r={raw_r:+.3f}  SHAP r={shap_r:+.3f}  "
                  f"range ratio {orth['range_ratio']:.1f}x")

            # dose-response inside minimum-temperature quartiles
            tq = pd.qcut(sdata.loc[per_obs.index, T_MIN], 4,
                         labels=["Q1 coldest", "Q2", "Q3", "Q4 warmest"])
            srows = []
            for lev in tq.cat.categories:
                mm = (tq == lev).values
                if mm.sum() < 50:
                    continue
                xs = sdata.loc[per_obs.index, DP_MIN].values[mm]
                ps = per_obs[DP_MIN].values[mm]
                good = ~np.isnan(xs) & ~np.isnan(ps)
                srows.append(dict(stratum=str(lev), n=int(good.sum()),
                                  zero_cross=zero_crossing(xs[good], ps[good]),
                                  slope=float(np.polyfit(xs[good], ps[good], 1)[0])))
            A.table("temp_strata", pd.DataFrame(srows))

    # era stability
    eras = [] if args.skip_era else [(WINDOW[0], WINDOW[0] + 4),
                                     (WINDOW[1] - 4, WINDOW[1])]
    erows = []
    for a, b in eras:
        sub = df[(df[YEAR_COL] >= a) & (df[YEAR_COL] <= b)]
        if sub[GROUP_COL].nunique() < 100:
            continue
        rk, _, _ = core.interventional_shap(sub, feats, best_ctor, n_folds=5,
                                            sample_size=1500, verbose=False)
        row = dict(era=f"{a}-{b}", n_obs=len(sub))
        for f in [DP_MIN, "eng2_night_cooling_efficiency"]:
            r = rk[rk.feature == f]
            if len(r):
                row[f + "_rank"] = int(r["rank"].iloc[0])
                row[f + "_shap"] = float(r.mean_abs_shap.iloc[0])
        erows.append(row)
    if erows:
        A.table("era_stability", pd.DataFrame(erows))
        print(f"  era stability: {erows}")

    # modality leave-one-out
    loo = []
    if not args.skip_loo:
        print("  modality leave-one-out")
    mods = [] if args.skip_loo else sorted({modality_of(f) for f in feats})
    for mk in mods:
        keep = [f for f in feats if modality_of(f) != mk]
        if len(keep) < 20:
            continue
        r = core.cv_evaluate(df, keep, best_ctor, n_folds=5)["metrics"]["r2"]
        loo.append(dict(modality=mk, n_removed=len(feats) - len(keep),
                        r2_without=r, delta=r - m["r2"]))
        print(f"    {mk[:38]:<38} R2={r:.4f}  delta={r - m['r2']:+.4f}")
    if loo:
        A.table("modality_loo", pd.DataFrame(loo).sort_values("delta"))

    # error structure and residual spatial statistic
    oof["bracket"] = pd.cut(oof[TARGET], [60, 70, 74, 76.5, 79, 95],
                            labels=["60-70", "70-74", "74-76.5", "76.5-79", "79-95"])
    br = (oof.groupby("bracket", observed=True)
          .agg(n=("resid", "size"), mae=("resid", lambda s: s.abs().mean()),
               mean_resid=("resid", "mean")).reset_index())
    A.table("bracket_error", br)
    per_year = (oof.groupby(YEAR_COL)
                .apply(lambda g: pd.Series({
                    "r2": 1 - ((g.resid ** 2).sum() /
                               ((g[TARGET] - g[TARGET].mean()) ** 2).sum()),
                    "mae": g.resid.abs().mean(), "n": len(g)}))
                .reset_index())
    A.table("per_year", per_year)

    cty = oof.groupby(GROUP_COL).resid.mean().reset_index()
    cty["st"] = cty[GROUP_COL].astype(str).str.zfill(5).str[:2]
    gm = cty.resid.mean(); dev = cty.resid - gm
    num = W = 0.0
    for _, sub in cty.groupby("st"):
        v = (sub.resid - gm).values; k = len(v)
        if k > 1:
            num += v.sum() ** 2 - (v ** 2).sum(); W += k * (k - 1)
    den = (dev ** 2).sum()
    m["morans_I_resid"] = float((len(cty) / W) * (num / den)) if W and den else None
    print(f"  Moran's I of residuals: {m['morans_I_resid']}")

    # -- 8  final model and optional forward prediction ---------------------
    print("\n[8] refit on all rows and save model")
    full_model, Xf, fdata, med = core.fit_full_model(df, feats, best_ctor)
    A.model("final_model", dict(model=full_model, features=feats,
                                median_fill=med.to_dict(), window=WINDOW,
                                model_name=best))

    if args.forward:
        fw = df_all[(df_all[YEAR_COL] >= FORWARD[0]) &
                    (df_all[YEAR_COL] <= FORWARD[1])].copy()
        if len(fw):
            avail = fw[feats].notna().mean()
            usable = avail[avail >= 0.80].index.tolist()
            print(f"  forward window {FORWARD[0]}-{FORWARD[1]}: {len(fw):,} rows, "
                  f"{len(usable)}/{len(feats)} features at >=80% coverage")
            if len(usable) >= 0.7 * len(feats):
                pred = full_model.predict(fw[feats].fillna(med))
                out = fw[[GROUP_COL, YEAR_COL]].copy()
                out["predicted_le"] = pred
                out["note"] = "extrapolation beyond target availability; no ground truth"
                A.table("forward_predictions", out)
            else:
                print("  too few features available; forward prediction skipped")

    A.json("metrics", m)
    A.close()
    print(f"\nSTUDY A complete in {(time.time()-t_start)/60:.1f} min "
          f"-> {args.outdir}/")


if __name__ == "__main__":
    main()
