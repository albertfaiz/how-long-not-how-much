"""
17_formaldehyde_pair_anatomy.py
Why the two formaldehyde duration variables do not replace each other.

THE OBSERVATION THIS SCRIPT EXPLAINS
------------------------------------
Two columns describe the fraction of the year a county spends above a fixed
national reference for formaldehyde: one at the 75th percentile, one at the
90th. They correlate at 0.991 and both are in the model, because both match a
protected pattern. The 75th-percentile column ranks first of the admitted set
and the 90th ranks fourth.

If they were two noisy measurements of one quantity, removing the first would
hand its attribution to the second, which would then rank first with roughly
the first's attribution. That is not what happens. Something survives the
removal that neither column carries on its own.

The explanation this script tests is that the information is in the BAND
between them. Time above the 75th percentile and time above the 90th differ by
the time spent between those two levels: moderately elevated but not extreme.
Two counties with the same time above the 90th can differ greatly in that band,
and the band is a distinct physical quantity, not a noisier version of either
parent.

WHAT IT RUNS
------------
1.  REMOVAL. Four refits on identical folds: both present, each one removed in
    turn, and both removed. Reports out-of-fold skill and, for each, where the
    removed variable's attribution went.

2.  INHERITANCE. After removing one, how much of its attribution does the
    survivor gain, and what rank does the survivor reach. A pair that is one
    measurement twice gives most of it back; a pair that is two measurements
    gives little.

3.  THE BAND. A derived column, time above the 75th minus time above the 90th,
    is added to a model with both parents removed, and to a model with both
    parents present. If the band is what the pair was carrying, the first
    recovers most of the lost skill with one column.

4.  THE GENERAL CASE. The same removal test for every other pair of columns in
    the admitted set that correlates above the collinearity cut, so the
    formaldehyde pair can be placed against its peers rather than described
    alone.

READS   the modelling panel, the attribution ranking
WRITES  derived/pair_anatomy.json
        derived/pair_anatomy_arms.csv
TIME    45 to 120 minutes. Every arm checkpoints, so it resumes.

    python3 17_formaldehyde_pair_anatomy.py
    python3 17_formaldehyde_pair_anatomy.py --only removal,band
    python3 17_formaldehyde_pair_anatomy.py --max-arms 2
"""
from __future__ import annotations

# ===========================================================================
#  SETTINGS
# ===========================================================================
# The pair the paper is built on. Named rather than discovered, so the script
# cannot quietly change which pair it is describing.
PAIR = ("cams_fot_formaldehyde_above75th_percentile",
        "cams_fot_formaldehyde_above_90th_percentile")
PAIR_LABEL = ("time above the 75th percentile",
              "time above the 90th percentile")

# The band is the first minus the second: time spent between the two levels.
# It is only meaningful when the first is the lower threshold, which is checked
# at run time rather than assumed.
BAND_NAME = "band_formaldehyde_75_to_90"

# For the general case: every admitted pair correlating at or above this, which
# is the cut the main run's redundancy filter uses.
PEER_CUT = 0.98
PEER_MAX = 6                 # how many peer pairs to test, strongest first

# Attribution settings for this diagnostic. They are smaller than the study's
# own because the question here is how one variable's attribution CHANGES
# between two models, which is a ratio, and because an interventional
# explanation costs rows times background rows times trees. Every arm uses the
# same two numbers, so the comparison between arms is exact; the absolute
# values are not comparable with the study ranking and are not used as such.
SHAP_ROWS = 3000             # county-years explained in each refit
SHAP_BACKGROUND = 120        # reference rows for the interventional value
# ===========================================================================

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as C

_BUDGET = {"left": 0, "limited": False}
_SMOKE = {"folds": 0}


def store():
    tag = f"_smoke{_SMOKE['folds']}" if _SMOKE["folds"] else ""
    d = C.DERIVED / f"checkpoints_pair_anatomy{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def arm(panel, drop, label, extra=None):
    """Refit without `drop`, optionally with an extra column, and attribute.

    Returns out-of-fold skill and the mean absolute attribution of every
    column, so the attribution a removed variable used to hold can be followed
    to wherever it went.
    """
    import shap
    import xgboost as xgb
    from sklearn.metrics import r2_score, mean_absolute_error

    ck = store() / f"{label}.json"
    if ck.exists():
        done = json.load(open(ck))
        print(f"    {label:<26} already done: R2 {done['r2']:.4f}")
        return done

    if _BUDGET["limited"] and _BUDGET["left"] <= 0:
        raise SystemExit(
            f"\n  --max-arms reached before fitting '{label}'.\n"
            f"    Everything so far is checkpointed in {store()}\n"
            f"    Run the same command again to continue.")
    _BUDGET["left"] -= 1

    X, y, groups = panel["X"], panel["y"], panel["groups"]
    cols = [c for c in X.columns if c not in set(drop)]
    Xa = X[cols]
    if extra is not None:
        Xa = pd.concat([Xa, extra], axis=1)

    t0 = time.time()
    nf = _SMOKE["folds"] or C.N_FOLDS
    tag = f"pair_{label}" + (f"_smoke{nf}" if _SMOKE["folds"] else "")
    oof = C.cv_oof(Xa, y, groups, tag, n_folds=nf)

    full = xgb.XGBRegressor(**C.XGB_PARAMS, n_jobs=C.threads())
    full.fit(Xa, y)
    rng = C.seed_everything()
    pick = rng.choice(len(y), size=min(SHAP_ROWS + SHAP_BACKGROUND, len(y)),
                      replace=False)
    # the background must not be part of what is explained, or a row acts as
    # its own reference
    bg, idx = np.sort(pick[:SHAP_BACKGROUND]), np.sort(pick[SHAP_BACKGROUND:])
    ex = shap.TreeExplainer(full, Xa.iloc[bg],
                            feature_perturbation="interventional")
    print(f"    {label:<26} attributing {len(idx):,} rows against "
          f"{len(bg)} reference rows", flush=True)
    sv = ex.shap_values(Xa.iloc[idx], check_additivity=False)
    mean_abs = np.abs(np.asarray(sv)).mean(axis=0)
    order = np.argsort(-mean_abs)
    attribution = {Xa.columns[k]: float(mean_abs[k]) for k in order}
    ranks = {c: i + 1 for i, c in enumerate(attribution)}

    out = dict(label=label, dropped=list(drop),
               n_folds=int(_SMOKE["folds"] or C.N_FOLDS),
               smoke_test=bool(_SMOKE["folds"]),
               extra=list(extra.columns) if extra is not None else [],
               n_features=int(Xa.shape[1]),
               shap_rows=int(len(idx)), shap_background=int(len(bg)),
               r2=float(r2_score(y, oof)),
               mae=float(mean_absolute_error(y, oof)),
               attribution=attribution, ranks=ranks,
               seconds=round(time.time() - t0, 1))
    json.dump(out, open(ck, "w"), indent=1)
    print(f"    {label:<26} R2 {out['r2']:.4f}  [checkpointed]", flush=True)
    del full, sv
    gc.collect()
    return out


def show(a, b, keys):
    """A compact comparison of two arms on named columns."""
    for k in keys:
        va = a["attribution"].get(k)
        vb = b["attribution"].get(k)
        ra = a["ranks"].get(k)
        rb = b["ranks"].get(k)
        fa = "removed" if va is None else f"{va:.4f} (#{ra})"
        fb = "removed" if vb is None else f"{vb:.4f} (#{rb})"
        print(f"      {k[:52]:<52} {fa:>18}  ->  {fb:>18}")


def pre_model_facts(panel):
    """What can be known about the pair before any model is fitted.

    This runs in seconds and it constrains which explanations are available,
    so it is printed first. Two of the three candidate explanations for the
    pair's behaviour can be ruled in or out from these numbers alone.
    """
    from sklearn.feature_selection import mutual_info_regression

    print("\n" + "=" * 74)
    print("  0. WHAT IS TRUE BEFORE ANY MODEL IS FITTED")
    print("=" * 74)
    lo_col, hi_col = PAIR
    X, y = panel["X"], panel["y"]
    band = (X[lo_col] - X[hi_col])

    r_pair = float(np.corrcoef(X[lo_col], X[hi_col])[0, 1])
    print(f"  the pair correlates at {r_pair:.4f}")
    print(f"\n  {'':<16}{'mean':>10}{'sd':>10}{'r with outcome':>18}"
          f"{'mutual info':>14}")
    sub = pd.DataFrame({"p_low": X[lo_col], "p_high": X[hi_col],
                        "band": band})
    mi = mutual_info_regression(sub, y, random_state=C.SEED, n_neighbors=3)
    for (name, v), m in zip(sub.items(), mi):
        print(f"  {name:<16}{v.mean():>10.4f}{v.std():>10.4f}"
              f"{float(np.corrcoef(v, y)[0, 1]):>18.4f}{m:>14.4f}")

    neg = int((band < -1e-9).sum())
    print(f"\n  band negative in {neg} of {len(band):,} county-years")
    print(f"  corr(band, lower threshold)  {np.corrcoef(band, X[lo_col])[0, 1]:+.4f}")
    print(f"  corr(band, higher threshold) {np.corrcoef(band, X[hi_col])[0, 1]:+.4f}")

    A = np.c_[np.ones(len(X)), X[lo_col], X[hi_col]]
    resid = band - A @ np.linalg.lstsq(A, band, rcond=None)[0]
    frac = float(resid.std() / band.std()) if band.std() else float("nan")
    print(f"\n  band residual after regressing on both parents: "
          f"{100 * frac:.2f} per cent of its own spread")
    print("\n  Reading. The band is an exact linear combination of the pair,")
    print("  so it adds nothing a linear model could not already form. A")
    print("  gradient-boosted tree splits on one column at a time and cannot")
    print("  form a difference, so for this model class the band is not")
    print("  redundant: representing it requires repeated splits on both")
    print("  parents, and removing either parent makes it unavailable. That")
    print("  is the hypothesis the arms below test.")
    return dict(correlation=r_pair,
                mi_low=float(mi[0]), mi_high=float(mi[1]),
                mi_band=float(mi[2]),
                r_outcome_low=float(np.corrcoef(X[lo_col], y)[0, 1]),
                r_outcome_high=float(np.corrcoef(X[hi_col], y)[0, 1]),
                r_outcome_band=float(np.corrcoef(band, y)[0, 1]),
                band_negative_rows=neg,
                band_residual_fraction=frac)


def removal_and_inheritance(panel, args):
    print("\n" + "=" * 74)
    print("  1 and 2. REMOVAL AND INHERITANCE")
    print("=" * 74)
    lo_col, hi_col = PAIR
    X = panel["X"]
    for c in PAIR:
        if c not in X.columns:
            raise SystemExit(f"{c} is not in the modelled set.")
    if X[lo_col].mean() <= X[hi_col].mean():
        raise SystemExit(
            f"{lo_col} should spend more time above its threshold than "
            f"{hi_col}; the pair is named the wrong way round in SETTINGS.")

    r = np.corrcoef(X[lo_col], X[hi_col])[0, 1]
    print(f"  the pair correlates at {r:.4f}")

    both = arm(panel, [], "both_present")
    no_lo = arm(panel, [lo_col], "without_p75")
    no_hi = arm(panel, [hi_col], "without_p90")
    neither = arm(panel, list(PAIR), "without_both")

    print("\n  out of fold:")
    for k, v in (("both present", both), ("without the 75th", no_lo),
                 ("without the 90th", no_hi), ("without both", neither)):
        print(f"      {k:<20} R2 {v['r2']:.4f}   MAE {v['mae']:.4f}   "
              f"{v['n_features']} columns")

    print("\n  where the attribution went when the 75th was removed:")
    show(both, no_lo, [hi_col])
    print("\n  where the attribution went when the 90th was removed:")
    show(both, no_hi, [lo_col])

    a_lo = both["attribution"][lo_col]
    a_hi = both["attribution"][hi_col]
    inherit_lo = (no_lo["attribution"][hi_col] - a_hi) / a_lo
    inherit_hi = (no_hi["attribution"][lo_col] - a_lo) / a_hi
    print(f"\n  the 90th inherits {100 * inherit_lo:.1f} per cent of what the "
          f"75th was carrying")
    print(f"  the 75th inherits {100 * inherit_hi:.1f} per cent of what the "
          f"90th was carrying")
    print(f"  skill cost of removing the 75th   {both['r2'] - no_lo['r2']:+.4f}")
    print(f"  skill cost of removing the 90th   {both['r2'] - no_hi['r2']:+.4f}")
    print(f"  skill cost of removing both       "
          f"{both['r2'] - neither['r2']:+.4f}")
    print("\n  Reading: if the pair were one measurement made twice, the")
    print("  survivor would take up most of the removed attribution and the")
    print("  skill cost would be near zero. A survivor that takes up little,")
    print("  with a skill cost larger than either column removed alone, says")
    print("  the two carry different information about the outcome.")

    return dict(correlation=float(r), both=both, without_p75=no_lo,
                without_p90=no_hi, without_both=neither,
                inherit_p90_of_p75=float(inherit_lo),
                inherit_p75_of_p90=float(inherit_hi))


def band(panel, args, prev):
    print("\n" + "=" * 74)
    print("  3. THE BAND BETWEEN THE TWO LEVELS")
    print("=" * 74)
    lo_col, hi_col = PAIR
    X = panel["X"]
    b = (X[lo_col] - X[hi_col]).rename(BAND_NAME).to_frame()
    neg = int((b[BAND_NAME] < -1e-9).sum())
    print(f"  {BAND_NAME}: mean {float(b[BAND_NAME].mean()):.4f}, "
          f"{neg} county-years negative")
    if neg > 0.01 * len(b):
        print("  More than one per cent are negative, which should not happen")
        print("  for nested thresholds. Treat the band result with care.")

    for name, other in (("corr with the 75th", lo_col),
                        ("corr with the 90th", hi_col)):
        print(f"  {name}: "
              f"{np.corrcoef(b[BAND_NAME], X[other])[0, 1]:+.4f}")

    only_band = arm(panel, list(PAIR), "band_replaces_pair", extra=b)
    with_band = arm(panel, [], "band_added_to_pair", extra=b)

    nb = prev["without_both"]
    bt = prev["both"]
    print("\n  out of fold:")
    print(f"      both parents present         R2 {bt['r2']:.4f}")
    print(f"      neither parent               R2 {nb['r2']:.4f}")
    print(f"      band alone, parents removed  R2 {only_band['r2']:.4f}")
    print(f"      band added to both parents   R2 {with_band['r2']:.4f}")
    lost = bt["r2"] - nb["r2"]
    got = only_band["r2"] - nb["r2"]
    frac = got / lost if lost > 1e-9 else float("nan")
    print(f"\n  removing the pair costs {lost:+.4f} R2")
    print(f"  one band column recovers {got:+.4f} of it, "
          f"{100 * frac:.0f} per cent")
    print(f"  the band's own attribution with the parents removed: "
          f"{only_band['attribution'].get(BAND_NAME, float('nan')):.4f} "
          f"(#{only_band['ranks'].get(BAND_NAME)})")
    print(f"  and with the parents present: "
          f"{with_band['attribution'].get(BAND_NAME, float('nan')):.4f} "
          f"(#{with_band['ranks'].get(BAND_NAME)})")
    return dict(band_only=only_band, band_with_pair=with_band,
                fraction_recovered=float(frac),
                negative_band_rows=neg)


def peers(panel, args):
    print("\n" + "=" * 74)
    print("  4. THE SAME TEST ON EVERY OTHER HIGHLY CORRELATED PAIR")
    print("=" * 74)
    X = panel["X"]
    rank = C.load_table("shap_ranking").sort_values("rank")
    top = [c for c in rank.feature.head(60) if c in X.columns]
    Z = X[top].to_numpy(dtype="float32")
    corr = np.corrcoef(Z, rowvar=False)
    np.fill_diagonal(corr, 0.0)

    found = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            if abs(corr[i, j]) >= PEER_CUT:
                found.append((abs(corr[i, j]), top[i], top[j]))
    found.sort(reverse=True)
    found = [f for f in found if set(f[1:]) != set(PAIR)][:PEER_MAX]
    if not found:
        print(f"  no other pair among the 60 leading columns reaches "
              f"{PEER_CUT}")
        return []
    print(f"  {len(found)} peer pair(s) at or above {PEER_CUT}")

    rows = []
    base = arm(panel, [], "both_present")
    for k, (r, a, b) in enumerate(found, 1):
        lab = f"peer{k}_without_{a[:24]}"
        out = arm(panel, [a], lab)
        ga = base["attribution"][a]
        gb = base["attribution"][b]
        inherit = (out["attribution"].get(b, 0.0) - gb) / ga if ga else np.nan
        rows.append(dict(pair_a=a, pair_b=b, correlation=float(r),
                         attribution_a=float(ga), attribution_b=float(gb),
                         inherited_fraction=float(inherit),
                         r2_with=float(base["r2"]), r2_without=float(out["r2"]),
                         r2_cost=float(base["r2"] - out["r2"])))
        print(f"    {a[:40]:<40} r={r:.3f}  partner inherits "
              f"{100 * inherit:5.1f}%  skill cost {base['r2'] - out['r2']:+.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="comma separated: removal, band, peers. Stage 0, the "
                         "model-free facts, always runs; it takes seconds.")
    ap.add_argument("--max-arms", type=int, default=0,
                    help="stop after this many arms are FITTED in this run")
    ap.add_argument("--folds", type=int, default=0,
                    help="SMOKE TEST ONLY. Fewer folds than the study uses, "
                         "to check the code path end to end in minutes. Any "
                         "number produced this way is not comparable with the "
                         "paper and the output says so.")
    a = ap.parse_args()
    want = (["removal", "band", "peers"] if a.only == "all"
            else [s.strip() for s in a.only.split(",")])
    if a.max_arms:
        _BUDGET["left"], _BUDGET["limited"] = a.max_arms, True

    if a.folds:
        _SMOKE["folds"] = a.folds
    C.banner("17  why the formaldehyde pair does not collapse to one column")
    if a.folds:
        print(f"  SMOKE TEST at {a.folds} folds instead of {C.N_FOLDS}.")
        print("  Results are written under a separate tag and are not")
        print("  comparable with the paper.")
    C.seed_everything()
    panel = C.load_panel()

    blob = {"pre_model": pre_model_facts(panel)}
    prev = None
    if "removal" in want or "band" in want:
        prev = removal_and_inheritance(panel, a)
        blob["removal"] = {k: (v if not isinstance(v, dict) else
                               {kk: v[kk] for kk in
                                ("label", "r2", "mae", "n_features")})
                           for k, v in prev.items()}
        blob["removal"]["correlation"] = prev["correlation"]
        blob["removal"]["inherit_p90_of_p75"] = prev["inherit_p90_of_p75"]
        blob["removal"]["inherit_p75_of_p90"] = prev["inherit_p75_of_p90"]
    if "band" in want:
        bd = band(panel, a, prev)
        blob["band"] = dict(
            fraction_recovered=bd["fraction_recovered"],
            negative_band_rows=bd["negative_band_rows"],
            band_only_r2=bd["band_only"]["r2"],
            band_with_pair_r2=bd["band_with_pair"]["r2"],
            band_attribution_alone=bd["band_only"]["attribution"].get(BAND_NAME),
            band_rank_alone=bd["band_only"]["ranks"].get(BAND_NAME),
            band_attribution_with_pair=bd["band_with_pair"]["attribution"].get(BAND_NAME),
            band_rank_with_pair=bd["band_with_pair"]["ranks"].get(BAND_NAME))
    out = []
    if "peers" in want:
        rows = peers(panel, a)
        if rows:
            p = C.DERIVED / "pair_anatomy_arms.csv"
            pd.DataFrame(rows).to_csv(p, index=False)
            print(f"[saved] {p}")
            out.append(p)
            blob["peers"] = rows

    p = C.DERIVED / "pair_anatomy.json"
    json.dump(blob, open(p, "w"), indent=1, default=str)
    print(f"[saved] {p}")
    out.append(p)
    C.write_manifest(__file__, [], out, notes=f"only={a.only}")


if __name__ == "__main__":
    main()
