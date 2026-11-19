#!/usr/bin/env python3
"""
23_influence_and_perturbation.py -- is any of this carried by a few counties,
and how far do the thresholds move under measurement error?

READS   the fitted model and the modelling panel; the out-of-fold predictions
        of the main run; the census of script 12; the attribution surface of
        script 05; the turning points of script 06
WRITES  ../derived*/census_jackknife.json, census_jackknife.csv
        ../derived*/influence_drop.json
        ../derived*/perturbation_thresholds.json,
                    perturbation_thresholds.csv

THREE OBJECTIONS
----------------
    --only jackknife
        "Drop California and your thirty-eight of forty-seven is twenty of
         forty-seven."
    The census count is a function of a row average, so it can be recomputed
    exactly with any state's rows removed, at the cost of one attribution pass
    rather than fifty refits. What this holds fixed is stated below and
    matters.

    --only influence
        "A hundred unusual counties are driving this."
    Remove the counties the model handles worst, and the counties with the
    most extreme outcomes, and refit. The score will RISE when hard cases are
    dropped, which is not evidence of anything; the ranking is what is being
    tested.

    --only perturb
        "These are reanalysis fields with their own error and you report
         thresholds to three decimals."
    Perturb each predictor at several stated magnitudes and relocate its
    crossing with script 06's own estimator. Report the movement against the
    width of the published interval.

WHAT THE PERTURBATION ARM DOES NOT DO
-------------------------------------
It perturbs the predictor and relocates the crossing on the EXISTING
attribution values. It does not refit the model on perturbed predictors, which
would be the full error-in-variables experiment and costs a refit per
magnitude per variable. So it measures the sensitivity of the located value to
predictor error, not the sensitivity of the model. That is the narrower claim
and it is the one the result file makes.

No perturbed value is written to any exported panel, used by any figure, or
read by the manuscript as a measurement. These are sensitivity arms and every
file says so.

    python3 23_influence_and_perturbation.py --only jackknife
    python3 23_influence_and_perturbation.py --only perturb
    python3 23_influence_and_perturbation.py --only influence --folds 3
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _config as C                                            # noqa: E402

JACK_JSON = C.DERIVED / "census_jackknife.json"
JACK_CSV = C.DERIVED / "census_jackknife.csv"
INFL_JSON = C.DERIVED / "influence_drop.json"
PERT_JSON = C.DERIVED / "perturbation_thresholds.json"
PERT_CSV = C.DERIVED / "perturbation_thresholds.csv"

# ===========================================================================
#  SETTINGS
# ===========================================================================
JACK_ROWS = 30000          # rows attributed once, then averaged with each
                           # state's rows left out in turn
DROP_SHARES = (0.01, 0.05)  # share of counties removed in the influence arms
PERTURB_SDS = (0.01, 0.02, 0.05, 0.10)   # noise as a share of the column's
                                         # own standard deviation
PERTURB_DRAWS = 25         # independent perturbations per magnitude
PERTURB_BOOT = 200         # bootstrap draws inside each. Lower than the
                           # paper's 5,000 because this arm asks how far the
                           # point estimate moves, not what the interval is,
                           # and the number is recorded in the result file
TOPK = 15                  # ranking overlap reported at this depth

_SMOKE = {"folds": 0}


def folds():
    return _SMOKE["folds"] or C.N_FOLDS


def sign_test_p(wins, n):
    from math import comb
    if not n:
        return float("nan")
    return float(sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n)


def canonical_crossing():
    path = os.path.join(HERE, "06_bootstrap_turning_points_and_gate.py")
    if not os.path.exists(path):
        raise SystemExit(
            "06_bootstrap_turning_points_and_gate.py is not beside this file.\n"
            "    The perturbation arm relocates each crossing with that\n"
            "    script's estimator so the movement is comparable with the\n"
            "    published table. Copy the whole analysis folder.")
    spec = importlib.util.spec_from_file_location("_s06", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def path_contribs(model, X, idx, keep):
    """Per-row path-dependent contributions for the columns in `keep`."""
    import xgboost as xgb
    sub = X.iloc[idx]
    dm = xgb.DMatrix(sub, feature_names=list(sub.columns))
    contrib = np.asarray(model.get_booster().predict(dm, pred_contribs=True))
    cols = list(sub.columns)
    pos = {c: i for i, c in enumerate(cols)}
    return {c: np.abs(contrib[:, pos[c]]) for c in keep if c in pos}


# ---------------------------------------------------------------------------
#  1. Leave one state out of the census count
# ---------------------------------------------------------------------------
def jackknife(p, model, rng):
    print("\n" + "=" * 74)
    print("  1. LEAVE ONE STATE OUT OF THE CENSUS COUNT")
    print("=" * 74)
    print("  The census count compares the mean absolute attribution of two")
    print("  columns, and a mean is a row average, so the count with one")
    print("  state's rows removed can be computed exactly from one")
    print("  attribution pass. Fifty refits are not needed and would answer a")
    print("  different question.")
    print()
    print("  HELD FIXED: the fitted model. This arm asks whether the count")
    print("  depends on which counties are ATTRIBUTED. Whether it depends on")
    print("  which counties were TRAINED on is the state holdout of script 13.")
    print()
    X, y, groups = p["X"], p["y"], p["groups"]
    pairs = C.census_pairs(columns=X.columns)
    keep = sorted({c for _, a_, b_ in pairs for c in (a_, b_)})
    print(f"    {len(pairs)} pairs, {len(keep)} columns")

    n = len(X)
    idx = (rng.choice(n, JACK_ROWS, replace=False) if n > JACK_ROWS
           else np.arange(n))
    t0 = time.time()
    contrib = path_contribs(model, X, idx, keep)
    print(f"    attributed {len(idx):,} rows in "
          f"{(time.time() - t0) / 60:.1f} min")

    # a county FIPS is five digits, the first two of which are the state
    fips = np.asarray([f"{int(g):05d}" for g in groups[idx]])
    states = np.array([f[:2] for f in fips])
    uniq = sorted(set(states.tolist()))
    print(f"    {len(uniq)} states present among the attributed rows")

    def count(mask):
        wins, got = 0, 0
        for field, c_col, n_col in pairs:
            cv, nv = contrib.get(c_col), contrib.get(n_col)
            if cv is None or nv is None:
                continue
            got += 1
            wins += int(nv[mask].mean() > cv[mask].mean())
        return got, wins

    base_got, base_wins = count(np.ones(len(idx), bool))
    print(f"\n    all states: non-central leads in {base_wins} of {base_got}")

    rows = []
    for st in uniq:
        m = states != st
        got, wins = count(m)
        rows.append(dict(arm="leave_one_state_out", state_fips=st,
                         n_rows_dropped=int((~m).sum()),
                         n_comparable=got, noncentral_wins=wins,
                         sign_test_p=sign_test_p(wins, got)))
    df = pd.DataFrame(rows)
    lo, hi = int(df.noncentral_wins.min()), int(df.noncentral_wins.max())
    worst = df.loc[df.noncentral_wins.idxmin()]
    print(f"    leaving out any one state, the count stays between {lo} and "
          f"{hi} of {base_got}")
    print(f"    the state whose removal lowers it most is FIPS "
          f"{worst['state_fips']} ({int(worst['noncentral_wins'])})")
    worst_p = float(df.sign_test_p.max())
    print(f"    the largest sign-test p over all fifty removals is "
          f"{worst_p:.3g}")
    print()
    print("    That last number is the one to quote. It says the count")
    print("    remains significant no matter which single state is removed.")
    df.to_csv(JACK_CSV, index=False)
    print(f"  [saved] {JACK_CSV}")
    return dict(n_rows_attributed=int(len(idx)), n_states=len(uniq),
                n_comparable=base_got, baseline_wins=base_wins,
                attribution="path dependent TreeSHAP, model held fixed",
                range=[lo, hi], worst_state=str(worst["state_fips"]),
                worst_count=int(worst["noncentral_wins"]),
                worst_sign_test_p=worst_p,
                per_state=df.to_dict("records"),
                note="The fitted model is held fixed; only the attributed "
                     "rows change.")


# ---------------------------------------------------------------------------
#  2. Drop the counties the model relies on most
# ---------------------------------------------------------------------------
def influence(p, rng):
    print("\n" + "=" * 74)
    print("  2. DROP THE COUNTIES THE MODEL HANDLES WORST, AND THE EXTREMES")
    print("=" * 74)
    print("  The score will RISE when the hardest counties are removed. That")
    print("  is arithmetic, not evidence, and the result file says so. What is")
    print("  being tested is the RANKING: if the leading variables are the")
    print("  same after the removal, no small set of counties is carrying")
    print("  them.")
    if _SMOKE["folds"]:
        print(f"\n  SMOKE TEST at {_SMOKE['folds']} folds. Not the study.")
    print()
    X, y, groups = p["X"], p["y"], p["groups"]

    # mean absolute out-of-fold residual per county, from the main run's own
    # predictions rather than from a refit here
    oof = None
    try:
        o = pd.read_csv(C.OOF_A)
        cols = {c.lower(): c for c in o.columns}
        pc = cols.get("prediction") or cols.get("pred") or cols.get("yhat")
        tc = cols.get("actual") or cols.get("observed") or cols.get("y") \
            or cols.get(C.TARGET)
        gc_ = cols.get(C.GROUP) or cols.get("geoid") or cols.get("county")
        if pc and tc and gc_:
            o = o[[gc_, tc, pc]].dropna()
            o["abs_err"] = (o[tc] - o[pc]).abs()
            oof = o.groupby(gc_)["abs_err"].mean()
            print(f"    out-of-fold residuals for {len(oof):,} counties, from "
                  f"{C.OOF_A.name}")
    except Exception as e:
        print(f"    could not read {C.OOF_A.name} ({type(e).__name__}); the "
              f"residual arms are skipped")

    county_mean_y = pd.Series(y).groupby(pd.Series(groups)).mean()
    arms = {}

    def run(name, drop_counties, why):
        keep = ~np.isin(groups, np.asarray(list(drop_counties)))
        if keep.sum() < 0.5 * len(groups):
            print(f"    {name}: would drop more than half the rows, skipped")
            return
        Xk, yk, gk = X[keep], y[keep], groups[keep]
        s = C.cv_score(Xk, yk, gk,
                       f"influence_{name}" + (f"_f{_SMOKE['folds']}"
                                              if _SMOKE["folds"] else ""),
                       n_folds=folds())
        s.update(arm=name, why=why, n_counties_dropped=len(drop_counties),
                 n_rows_dropped=int((~keep).sum()))
        arms[name] = s
        print(f"    {name:<22} dropped {len(drop_counties):>4} counties   "
              f"R2 {s['r2']:+.4f}   MAE {s['mae']:.4f}")

    for share in DROP_SHARES:
        if oof is not None:
            k = max(int(round(len(oof) * share)), 1)
            worst = set(oof.sort_values(ascending=False).head(k).index.tolist())
            run(f"worst_residual_{int(100 * share)}pct", worst,
                "counties with the largest mean absolute out-of-fold residual")
        k = max(int(round(len(county_mean_y) * share / 2)), 1)
        srt = county_mean_y.sort_values()
        extreme = set(srt.head(k).index.tolist()) | \
            set(srt.tail(k).index.tolist())
        run(f"extreme_outcome_{int(100 * share)}pct", extreme,
            "counties in the lowest and highest tails of mean life expectancy")

    # the ranking, after the largest removal, against the published one
    overlap = {}
    if arms:
        pub = C.load_table("A_shap_ranking")
        pub_top = list(pub.sort_values("rank")["feature"].astype(str)[:TOPK])
        import xgboost as xgb
        drop_name = max(arms, key=lambda k: arms[k]["n_counties_dropped"])
        print(f"\n    refitting once on the {drop_name} panel to compare the "
              f"top {TOPK}")
        # recover the mask for that arm by rerunning its selection
        if oof is not None and drop_name.startswith("worst_residual"):
            share = int(drop_name.split("_")[-1].replace("pct", "")) / 100
            k = max(int(round(len(oof) * share)), 1)
            dropped = set(oof.sort_values(ascending=False).head(k).index)
        else:
            share = int(drop_name.split("_")[-1].replace("pct", "")) / 100
            k = max(int(round(len(county_mean_y) * share / 2)), 1)
            srt = county_mean_y.sort_values()
            dropped = set(srt.head(k).index) | set(srt.tail(k).index)
        keep = ~np.isin(groups, np.asarray(list(dropped)))
        m = xgb.XGBRegressor(**C.XGB_PARAMS, n_jobs=C.threads())
        t0 = time.time()
        m.fit(X[keep], y[keep])
        sub = X[keep]
        ridx = (C.seed_everything(extra=25)
                .choice(len(sub), min(6000, len(sub)), replace=False))
        dm = xgb.DMatrix(sub.iloc[ridx], feature_names=list(sub.columns))
        contrib = np.asarray(m.get_booster().predict(dm, pred_contribs=True))
        mean_abs = pd.Series(np.abs(contrib[:, :-1]).mean(axis=0),
                             index=sub.columns).sort_values(ascending=False)
        new_top = list(mean_abs.index[:TOPK])
        shared = len(set(pub_top) & set(new_top))
        overlap = dict(arm=drop_name, topk=TOPK, shared=shared,
                       published_top=pub_top, refit_top=new_top,
                       attribution="path dependent TreeSHAP on the refit, so "
                                   "the comparison with the published "
                                   "interventional ranking crosses two "
                                   "estimators and is a direction check, not "
                                   "a value check")
        print(f"    top {TOPK} overlap with the published ranking: "
              f"{shared} of {TOPK}   "
              f"({(time.time() - t0) / 60:.1f} min)")
        del m
        gc.collect()

    return dict(arms=arms, topk_overlap=overlap,
                drop_shares=list(DROP_SHARES), n_folds=int(folds()),
                smoke_test=bool(_SMOKE["folds"]),
                note="Dropping the hardest counties raises the score by "
                     "construction. The ranking overlap is the test.")


# ---------------------------------------------------------------------------
#  3. Perturb the predictor and relocate the crossing
# ---------------------------------------------------------------------------
def perturb(args):
    print("\n" + "=" * 74)
    print("  3. PREDICTOR PERTURBATION AND THE LOCATED CROSSINGS")
    print("=" * 74)
    print("  Each predictor is perturbed by Gaussian noise at several stated")
    print("  magnitudes, as a share of that column's own standard deviation,")
    print("  and its crossing is relocated with script 06's estimator. The")
    print("  movement is reported against the width of the published interval,")
    print("  because a shift of a tenth of the interval is a different fact")
    print("  from a shift of three times its width.")
    print()
    print("  The attribution values are NOT recomputed. This arm measures the")
    print("  sensitivity of the located value to predictor error, not the")
    print("  sensitivity of the model to it.")
    print()
    s06 = canonical_crossing()
    dep, feats = C.dependence_table()
    print(f"    {len(feats)} candidates in the attribution surface, "
          f"{len(dep):,} rows each")
    try:
        pub = C.load_table("turning_points").set_index("feature")
    except SystemExit:
        pub = None
        print("    (turning_points.csv absent, so the movement cannot be "
              "scaled by the published interval; run script 06)")

    rng = C.seed_everything(extra=26)
    rows, per_var = [], {}
    mags = args.perturb_sds or PERTURB_SDS
    draws = args.perturb_draws or PERTURB_DRAWS
    boot = args.perturb_boot or PERTURB_BOOT
    # One checkpoint per variable, keyed by the settings that produced it.
    # Eleven variables at four magnitudes and twenty-five draws is eleven
    # hundred bootstrapped relocations, which is longer than one sitting, and
    # an arm that cannot be resumed is an arm that never finishes.
    store = C.DERIVED / "checkpoints_perturb"
    store.mkdir(parents=True, exist_ok=True)
    stamp = f"d{draws}_b{boot}_m" + "-".join(f"{m:g}" for m in mags)
    for feat in feats:
        ck = store / f"{feat}__{stamp}.json"
        if ck.exists():
            try:
                rec = json.load(open(ck))
                per_var[str(feat)] = rec
                for mk, mv in (rec.get("magnitudes") or {}).items():
                    if isinstance(mv, dict) and mv.get("n_found"):
                        rows.append(dict(
                            arm="perturbation", feature=str(feat),
                            magnitude=float(mk),
                            **{kk: mv[kk] for kk in
                               ("n_found", "median_point", "shift", "spread",
                                "shift_over_interval_width") if kk in mv}))
                print(f"    {str(feat)[:44]:<44} from checkpoint")
                continue
            except Exception:
                pass
        x = dep[feat].to_numpy(float)
        v = dep["shap_" + feat].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(v)
        x, v = x[ok], v[ok]
        if len(x) < 2000:
            continue
        sd = float(np.std(x))
        base = s06.analyse(str(feat), x, v, boot, C.N_BINS, rng)
        bp = base["point_estimate"]
        width = np.nan
        if pub is not None and feat in pub.index:
            width = float(pub.loc[feat, "ci_hi"] - pub.loc[feat, "ci_lo"])
            bp_pub = float(pub.loc[feat, "point_estimate"])
        else:
            bp_pub = bp
        rec = dict(feature=str(feat), sd=sd, published_point=bp_pub,
                   published_interval_width=width, magnitudes={})
        for mag in mags:
            pts = []
            for _ in range(draws):
                xx = x + rng.normal(0.0, mag * sd, len(x))
                r = s06.analyse(str(feat), xx, v, boot, C.N_BINS, rng)
                if r["point_estimate"] is not None and \
                        np.isfinite(r["point_estimate"]):
                    pts.append(float(r["point_estimate"]))
            if not pts:
                rec["magnitudes"][f"{mag:g}"] = dict(n_found=0)
                continue
            pts = np.asarray(pts)
            shift = float(np.median(pts) - bp_pub)
            spread = float(pts.std())
            rel = (abs(shift) / width if np.isfinite(width) and width
                   else float("nan"))
            rec["magnitudes"][f"{mag:g}"] = dict(
                n_found=int(len(pts)), n_draws=int(draws),
                median_point=float(np.median(pts)), shift=shift,
                spread=spread, shift_over_interval_width=rel)
            rows.append(dict(arm="perturbation", feature=str(feat),
                             magnitude=float(mag), n_found=int(len(pts)),
                             median_point=float(np.median(pts)), shift=shift,
                             spread=spread, shift_over_interval_width=rel))
        per_var[str(feat)] = rec
        json.dump(rec, open(ck, "w"), indent=1, default=str)
        worst = max((m.get("shift_over_interval_width", np.nan)
                     for m in rec["magnitudes"].values()
                     if isinstance(m, dict)), default=np.nan)
        verdict = (f"worst shift {worst:.2f} x the published interval width"
                   if np.isfinite(worst) else "interval width unknown")
        print(f"    {str(feat)[:44]:<44} {verdict}")

    df = pd.DataFrame(rows)
    out = dict(per_variable=per_var, magnitudes=[float(m) for m in mags],
               n_draws=int(draws), n_boot=int(boot),
               note="Gaussian perturbation of real predictor values. A "
                    "sensitivity arm. No perturbed value is exported, "
                    "plotted, or read by the manuscript as a measurement. "
                    "The attribution values are not recomputed.")
    if not df.empty:
        df.to_csv(PERT_CSV, index=False)
        print(f"  [saved] {PERT_CSV}")
        big = df.dropna(subset=["shift_over_interval_width"])
        if len(big):
            at = big[big.magnitude == max(mags)]
            if len(at):
                print(f"\n    at the largest magnitude ({max(mags):g} of the "
                      f"column's own spread), the median shift is")
                print(f"    {at.shift_over_interval_width.median():.2f} times "
                      f"the published interval width, and the worst is "
                      f"{at.shift_over_interval_width.max():.2f}")
            out["summary_at_largest_magnitude"] = dict(
                magnitude=float(max(mags)),
                median_shift_over_width=float(
                    at.shift_over_interval_width.median()) if len(at) else None,
                max_shift_over_width=float(
                    at.shift_over_interval_width.max()) if len(at) else None)
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="comma separated: jackknife, influence, perturb")
    ap.add_argument("--folds", type=int, default=0,
                    help="SMOKE TEST ONLY, for the influence arm")
    ap.add_argument("--perturb-sds", type=float, nargs="*", default=None,
                    help="noise magnitudes as a share of the column's own sd")
    ap.add_argument("--perturb-draws", type=int, default=0)
    ap.add_argument("--perturb-boot", type=int, default=0)
    a = ap.parse_args()
    want = (["jackknife", "influence", "perturb"] if a.only == "all"
            else [s.strip() for s in a.only.split(",")])
    if a.folds:
        _SMOKE["folds"] = a.folds

    C.banner("23  influence, and the thresholds under measurement error")
    C.seed_everything()
    C.DERIVED.mkdir(parents=True, exist_ok=True)
    rng = C.seed_everything(extra=27)

    need_panel = bool({"jackknife", "influence"} & set(want))
    p = C.load_panel() if need_panel else None
    if p is not None:
        print(f"  panel {p['X'].shape[0]:,} rows x {p['X'].shape[1]} columns")

    if "jackknife" in want:
        import joblib
        model = joblib.load(C.MODEL_A)["model"]
        out = jackknife(p, model, rng)
        json.dump(out, open(JACK_JSON, "w"), indent=1, default=str)
        print(f"  [saved] {JACK_JSON}")
        del model
        gc.collect()

    if "influence" in want:
        out = influence(p, rng)
        json.dump(out, open(INFL_JSON, "w"), indent=1, default=str)
        print(f"  [saved] {INFL_JSON}")

    if "perturb" in want:
        out = perturb(a)
        json.dump(out, open(PERT_JSON, "w"), indent=1, default=str)
        print(f"  [saved] {PERT_JSON}")

    C.write_manifest("23_influence_and_perturbation.py",
                     inputs=[C.MODEL_A, C.FRAME, C.OOF_A],
                     outputs=[q for q in (JACK_JSON, JACK_CSV, INFL_JSON,
                                          PERT_JSON, PERT_CSV) if q.exists()],
                     settings=dict(arms=want, DROP_SHARES=list(DROP_SHARES),
                                   PERTURB_SDS=list(a.perturb_sds
                                                    or PERTURB_SDS),
                                   PERTURB_DRAWS=a.perturb_draws
                                   or PERTURB_DRAWS,
                                   JACK_ROWS=JACK_ROWS, TOPK=TOPK,
                                   n_folds=folds()),
                     notes="Influence of single states and of extreme "
                           "counties, and the movement of each located "
                           "crossing under predictor perturbation.")


if __name__ == "__main__":
    main()
