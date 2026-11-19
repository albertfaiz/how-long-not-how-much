#!/usr/bin/env python3
# =============================================================================
#  pipeline_B_vanishing_v2.py
#  STUDY B  ->  Paper 3 (the vanishing environmental increment)
#
#  Environmental superset fused with the socioeconomic panel. Window 2009-2019,
#  set by the first year of socioeconomic availability.
#
#  Stages
#    1  load superset, add engineered layer, save reference frame
#    2  temporal-coverage gate at 80 percent, socioeconomic scope allowed
#    3  unsupervised redundancy filter with protected physiological features
#    4  learner tournament, fused feature set
#    5  headline metrics, bootstrap interval, out-of-fold predictions
#    6  information partition, saturation-richness curve
#    7  perturbation stress tests, conformal calibration
#    8  interventional TreeSHAP, composition, debiased partial screen
#    9  spurious-proxy audit, residual diagnostics
#   10  environment-only backcast into the pre-socioeconomic era
#
#  Run:  python pipeline_B_vanishing_v2.py --csv Final_ML_Ready_Dataset.csv --gpu
# =============================================================================

from __future__ import annotations
import argparse, time
import numpy as np
import pandas as pd

import pipeline_core_v2 as core
from pipeline_core_v2 import (TARGET, GROUP_COL, YEAR_COL, Artifacts,
                              modality_of, is_sdoh)

WINDOW = (2009, 2019)
BACKCAST = (2003, 2008)     # env-only era; CAMS and ERA5 start in 2003
TAG = "studyB"


def split_sources(features):
    sdoh = [f for f in features if is_sdoh(f)]
    env = [f for f in features if not is_sdoh(f)]
    return sdoh, env


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="Final_ML_Ready_Dataset.csv")
    ap.add_argument("--outdir", default="results_studyB")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--coverage", type=float, default=0.80)
    args = ap.parse_args()

    t0 = time.time()
    B = Artifacts(args.outdir, TAG)
    print("=" * 74)
    print(f"STUDY B  satellite + socioeconomic fusion  window {WINDOW[0]}-{WINDOW[1]}")
    print("=" * 74)

    # -- 1  load -------------------------------------------------------------
    df_all = core.load_master(args.csv, add_engineered=True)
    df = df_all[(df_all[YEAR_COL] >= WINDOW[0]) &
                (df_all[YEAR_COL] <= WINDOW[1])].reset_index(drop=True)
    print(f"[window] {len(df):,} rows, {df[GROUP_COL].nunique():,} counties")
    B.frame("reference_frame", df)

    # -- 2  gate -------------------------------------------------------------
    feats, gate = core.coverage_gate(df, WINDOW, min_coverage=args.coverage,
                                     allow_sdoh=True, drop_raw_ntl=True)
    B.table("feature_gate", gate)
    n_leak = int((gate.reason == "LEAKAGE-outcome-derived").sum())
    print(f"[gate] {n_leak} outcome-derived columns blocked before modelling")

    # -- 3  filter -----------------------------------------------------------
    feats = core.redundancy_filter(df[feats])
    sdoh, env = split_sources(feats)
    print(f"[scope] {len(feats)} features: {len(sdoh)} socioeconomic, "
          f"{len(env)} environmental")
    B.json("feature_set", dict(n_total=len(feats), n_sdoh=len(sdoh),
                               n_env=len(env), sdoh=sdoh, env=env))

    # -- 4  tournament -------------------------------------------------------
    print("\n[4] learner tournament, fused feature set")
    table, results, best = core.run_tournament(
        df, feats, n_folds=args.folds, use_gpu=args.gpu, fast=args.fast)
    B.table("model_tournament", table)
    zoo = core.build_models(use_gpu=args.gpu, fast=args.fast)
    best_ctor, best_shap_ok = zoo[best]
    res = results[best]

    # -- 5  headline ---------------------------------------------------------
    y = res["data"][TARGET].values
    ci = core.bootstrap_r2_ci(y, res["oof"])
    M = dict(res["metrics"]); M.update(model=best, r2_ci_lo=ci[0], r2_ci_hi=ci[1],
                                       window=list(WINDOW), n_sdoh=len(sdoh),
                                       n_env=len(env), n_leakage_blocked=n_leak)
    print(f"\n[5] fused {best}: R2={M['r2']:.4f} (95% CI {ci[0]:.4f}-{ci[1]:.4f}) "
          f"RMSE={M['rmse']:.4f} MAE={M['mae']:.4f}")
    oof = res["data"][[GROUP_COL, YEAR_COL, TARGET]].copy()
    oof["pred"] = res["oof"]; oof["resid"] = oof[TARGET] - oof["pred"]
    B.table("oof_predictions", oof)
    B.table("cv_folds", res["folds"])

    # -- 6  information partition -------------------------------------------
    print("\n[6] information partition")
    R_SE = M["r2"]
    R_S = core.cv_evaluate(df, sdoh, best_ctor, n_folds=args.folds)["metrics"]["r2"]
    R_E = core.cv_evaluate(df, env, best_ctor, n_folds=args.folds)["metrics"]["r2"]
    part = dict(R_S=R_S, R_E=R_E, R_SE=R_SE,
                U_E=R_SE - R_S, U_S=R_SE - R_E, Shared=R_S + R_E - R_SE)
    part["U_E_pct"] = 100 * part["U_E"] / R_SE
    part["U_S_pct"] = 100 * part["U_S"] / R_SE
    part["Shared_pct"] = 100 * part["Shared"] / R_SE
    B.table("info_partition", pd.DataFrame([part]))
    print(f"    socioeconomic only R2={R_S:.4f} | environment only R2={R_E:.4f} "
          f"| fused R2={R_SE:.4f}")
    print(f"    unique to environment {part['U_E']:.4f} ({part['U_E_pct']:.2f}%) | "
          f"unique to socioeconomic {part['U_S']:.4f} | shared {part['Shared']:.4f}")
    if part["Shared"] < 0:
        # R_S + R_E < R_SE means the two streams are SYNERGISTIC rather than
        # redundant: together they explain more than the sum of their parts.
        # That is a different finding from the vanishing increment and must be
        # reported as such rather than described as redundancy.
        print("    NOTE: shared component is negative, so the two sources are "
              "synergistic\n          rather than redundant in this window. "
              "Report this explicitly; it is\n          NOT the vanishing-increment "
              "result and must not be worded as one.")
        part["interpretation"] = "synergistic (negative shared component)"
    else:
        part["interpretation"] = "redundant (positive shared component)" 

    # saturation-richness curve: rank socioeconomic features first
    print("\n    saturation-richness curve")
    rank_s = core.interventional_shap(df, sdoh, best_ctor, n_folds=4,
                                      sample_size=1500, verbose=False)[0] \
        if best_shap_ok else None
    order = list(rank_s.feature) if rank_s is not None else sdoh
    sizes = [n for n in (1, 5, 20, 50, 100, 200, len(order)) if n <= len(order)]
    srows = []
    for n in sorted(set(sizes)):
        base = order[:n]
        rs = core.cv_evaluate(df, base, best_ctor, n_folds=5)["metrics"]["r2"]
        rf = core.cv_evaluate(df, base + env, best_ctor, n_folds=5)["metrics"]["r2"]
        srows.append(dict(n_sdoh=n, R_S=rs, R_SE=rf, increment=rf - rs))
        print(f"      n={n:>4}  base R2={rs:.4f}  fused R2={rf:.4f}  "
              f"increment={rf-rs:+.4f}")
    sat = pd.DataFrame(srows)
    pos = sat[sat.increment > 0]
    if len(pos) >= 3:
        b, loga = np.polyfit(np.log(pos.n_sdoh), np.log(pos.increment), 1)
        sat["power_fit"] = np.exp(loga) * sat.n_sdoh ** b
        M["sat_exponent"] = float(b); M["sat_coef"] = float(np.exp(loga))
    B.table("saturation_curve", sat)

    # -- 7  perturbation and calibration ------------------------------------
    print("\n[7] perturbation stress tests")
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(0)
    data = res["data"]
    X, yv, g = data[feats].copy(), data[TARGET].values, data[GROUP_COL].values

    def oof_r2(Xin):
        pred = np.full(len(data), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(Xin, yv, g):
            med = Xin.iloc[tr].median(numeric_only=True)
            mm = best_ctor(); mm.fit(Xin.iloc[tr].fillna(med), yv[tr])
            pred[te] = mm.predict(Xin.iloc[te].fillna(med))
        return float(r2_score(yv, pred))

    pert = {"baseline": oof_r2(X)}
    for label, cols in [("SDOH_shuffled", sdoh), ("EO_shuffled", env)]:
        Z = X.copy(); Z[cols] = Z[cols].values[rng.permutation(len(Z))]
        pert[label] = oof_r2(Z)
    for frac in (0.1, 0.3, 0.6):
        Z = X.copy()
        Z[env] = Z[env].mask(rng.random(Z[env].shape) < frac)
        pert[f"EO_missing_{int(frac*100)}pct"] = oof_r2(Z)
    for sd in (0.1, 0.5, 1.0):
        Z = X.copy()
        Z.loc[:, env] = Z[env].values + rng.normal(0, 1, Z[env].shape) * \
            Z[env].std(axis=0).values * sd
        pert[f"EO_noise_{int(sd*100)}pct"] = oof_r2(Z)
    B.table("perturbation", pd.DataFrame([pert]))
    drop_s = pert["baseline"] - pert["SDOH_shuffled"]
    drop_e = pert["baseline"] - pert["EO_shuffled"]
    print(f"    shuffling socioeconomic costs {drop_s:.4f}; "
          f"shuffling environment costs {drop_e:.4f} "
          f"(asymmetry {drop_s/max(drop_e,1e-9):.1f}x)")
    M["pert_asymmetry"] = float(drop_s / max(drop_e, 1e-9))

    # conformal calibration
    idx = rng.permutation(len(data))
    ntr, ncal = int(.6 * len(data)), int(.2 * len(data))
    tr, cal, te = idx[:ntr], idx[ntr:ntr + ncal], idx[ntr + ncal:]
    med = X.iloc[tr].median(numeric_only=True)
    cm = best_ctor(); cm.fit(X.iloc[tr].fillna(med), yv[tr])
    r_cal = np.abs(yv[cal] - cm.predict(X.iloc[cal].fillna(med)))
    q = float(np.quantile(r_cal, 0.8))
    naive = float(np.quantile(np.abs(yv[tr] - cm.predict(X.iloc[tr].fillna(med))), 0.8))
    pr = cm.predict(X.iloc[te].fillna(med))
    conf = dict(nominal=0.8,
                coverage_before=float(np.mean(np.abs(yv[te] - pr) <= naive)),
                coverage_after=float(np.mean(np.abs(yv[te] - pr) <= q)),
                width_before=2 * naive, width_after=2 * q)
    B.table("conformal", pd.DataFrame([conf]))
    print(f"    conformal coverage {conf['coverage_before']:.3f} -> "
          f"{conf['coverage_after']:.3f} (nominal 0.80)")

    # -- 8  attribution and debiased screen ---------------------------------
    if best_shap_ok:
        print("\n[8] interventional TreeSHAP on the fused model")
        ranking, per_obs, sdata = core.interventional_shap(
            df, feats, best_ctor, n_folds=args.folds)
        ranking["source"] = np.where(ranking.feature.map(is_sdoh),
                                     "Socioeconomic", "Environmental")
        B.table("shap_ranking", ranking)
        comp = (ranking.groupby("source")
                .agg(n=("feature", "size"), sum_shap=("mean_abs_shap", "sum"),
                     best_rank=("rank", "min")))
        comp["pct_attribution"] = 100 * comp.sum_shap / comp.sum_shap.sum()
        B.table("source_composition", comp.reset_index())
        print("    top 15:")
        for _, r in ranking.head(15).iterrows():
            print(f"      #{int(r['rank']):<4d} {r.feature[:44]:<44} "
                  f"|phi|={r.mean_abs_shap:.4f} [{r.source}]")

        # debiased partial screen on the leading features
        print("\n    debiased partial screen")
        from sklearn.model_selection import KFold
        top = list(ranking.feature.head(8))
        drows = []
        Xall = data[feats]
        for f in top:
            others = [c for c in feats if c != f]
            Xo = Xall[others].fillna(Xall[others].median())
            d = Xall[f].fillna(Xall[f].median()).values
            yr_, dr_ = np.zeros(len(d)), np.zeros(len(d))
            for tr2, te2 in KFold(5, shuffle=True, random_state=0).split(Xo):
                my = best_ctor(); my.fit(Xo.iloc[tr2], yv[tr2])
                md = best_ctor(); md.fit(Xo.iloc[tr2], d[tr2])
                yr_[te2] = yv[te2] - my.predict(Xo.iloc[te2])
                dr_[te2] = d[te2] - md.predict(Xo.iloc[te2])
            beta = np.polyfit(dr_, yr_, 1)[0]
            drows.append(dict(feature=f, partial_r2=max(0.0, r2_score(yr_, beta * dr_))))
            print(f"      {f[:46]:<46} partial R2={drows[-1]['partial_r2']:.5f}")
        B.table("dml_partial_r2", pd.DataFrame(drows).sort_values(
            "partial_r2", ascending=False))

        # -- 9  spurious-proxy audit ----------------------------------------
        print("\n[9] spurious-proxy audit")
        susp = ranking[(ranking["rank"] <= 60) &
                       (ranking.feature.str.contains(
                           r"^live_|^usda_|_class_|buffalo|corn|soy|wheat",
                           regex=True, case=False))]
        arows = []
        for f in susp.feature:
            if f not in df.columns:
                continue
            gg = df.groupby(GROUP_COL)[f]
            within, between = gg.std().mean(), gg.mean().std()
            st = (df.assign(_st=df[GROUP_COL].astype(str).str.zfill(5).str[:2])
                  .groupby("_st")[f].mean())
            arows.append(dict(
                feature=f, rank=int(ranking.loc[ranking.feature == f, "rank"].iloc[0]),
                between_within=float(between / max(within, 1e-9)),
                state_concentration=float(st.std() / max(abs(st.mean()), 1e-9)),
                nonzero_share=float((df[f].fillna(0) != 0).mean()),
                observed_share=float(df[f].notna().mean())))
        if arows:
            aud = pd.DataFrame(arows).sort_values("between_within", ascending=False)
            B.table("proxy_audit", aud)
            print(aud.round(3).to_string(index=False))
            # sensitivity: does removing them change anything?
            keep2 = [f for f in feats if f not in set(aud.feature)]
            r2_wo = core.cv_evaluate(df, keep2, best_ctor, n_folds=5)["metrics"]["r2"]
            s2, e2 = split_sources(keep2)
            rs2 = core.cv_evaluate(df, s2, best_ctor, n_folds=5)["metrics"]["r2"]
            M["proxy_sensitivity_r2_with"] = R_SE
            M["proxy_sensitivity_r2_without"] = r2_wo
            M["proxy_sensitivity_UE_without"] = r2_wo - rs2
            print(f"    sensitivity: fused R2 {R_SE:.4f} -> {r2_wo:.4f}; "
                  f"unique-environment {part['U_E']:.4f} -> {r2_wo - rs2:.4f}")
        else:
            print("    no candidates flagged in the top 60")
    else:
        ranking = None

    # residual outliers
    worst = (oof.groupby(GROUP_COL).resid.mean()
             .sort_values().head(15).reset_index())
    B.table("residual_outliers", worst)
    print("\n    largest shortfalls:")
    print(worst.head(6).round(2).to_string(index=False))

    # -- 10  backcast --------------------------------------------------------
    print("\n[10] environment-only backcast")
    bc = df_all[(df_all[YEAR_COL] >= BACKCAST[0]) &
                (df_all[YEAR_COL] <= BACKCAST[1])].dropna(subset=[TARGET])
    if len(bc):
        med_e = df[env].median()
        bm = best_ctor(); bm.fit(df[env].fillna(med_e), df[TARGET].values)
        pred = bm.predict(bc[env].fillna(med_e))
        from sklearn.metrics import r2_score as _r2
        M["backcast_r2"] = float(_r2(bc[TARGET].values, pred))
        M["backcast_n"] = int(len(bc))
        out = bc[[GROUP_COL, YEAR_COL, TARGET]].copy(); out["pred"] = pred
        B.table("backcast_predictions", out)
        print(f"    {BACKCAST[0]}-{BACKCAST[1]} env-only R2={M['backcast_r2']:.4f} "
              f"(n={len(bc):,})")

    # -- final model ---------------------------------------------------------
    full_model, Xf, fdata, medf = core.fit_full_model(df, feats, best_ctor)
    B.model("final_model", dict(model=full_model, features=feats,
                                median_fill=medf.to_dict(), window=WINDOW,
                                model_name=best))
    M.update(part)
    B.json("metrics", M)
    B.close()
    print(f"\nSTUDY B complete in {(time.time()-t0)/60:.1f} min -> {args.outdir}/")


if __name__ == "__main__":
    main()
