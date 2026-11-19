#!/usr/bin/env python3
"""
17_drop_formaldehyde_durations.py -- what happens to skill and to attribution
when the formaldehyde durations are taken away.

The leading variable of the model is the time a county spends above the
national 75th percentile of formaldehyde. Its near-twin is the time above the
90th percentile (the two correlate at 0.99), and the annual means of the same
species rank far lower. This step refits the model four times on the same
folds: with both durations, without the first, without the second, and without
both. It reports the out-of-fold skill of each refit, which involves no
attribution, and where the attribution goes: to the other duration when one is
removed, and to the mean forms of formaldehyde when both are.

Each refit is attributed on 3,000 county-years against 120 reference rows, the
same in every arm, so attributions are comparable between arms; their ranks
are not the ranks of the production ranking.

READS   the modelling panel
WRITES  ../derived*/formaldehyde_drop.json
        ../derived*/checkpoints_pair_anatomy/   one file per refit
TIME    seconds from the stored refits; about 15 minutes per refit otherwise

    python3 17_drop_formaldehyde_durations.py
"""
from __future__ import annotations

# ===========================================================================
#  SETTINGS
# ===========================================================================
PAIR = ("cams_fot_formaldehyde_above75th_percentile",
        "cams_fot_formaldehyde_above_90th_percentile")
# the mean forms of the same species, followed through the four refits
MEAN_FORMS = ("cams_formaldehyde_x1e9", "cams_total_column_formaldehyde_x1e9")
SHAP_ROWS = 3000             # county-years explained in each refit
SHAP_BACKGROUND = 120        # reference rows for the interventional value
# ===========================================================================

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
OUT = C.DERIVED / "formaldehyde_drop.json"


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


def main():
    C.banner("17  refits without the formaldehyde durations")
    C.seed_everything()
    panel = C.load_panel()
    r = removal_and_inheritance(panel, None)
    arms = ("both", "without_p75", "without_p90", "without_both")
    out = dict(
        correlation=r["correlation"],
        shap_rows=int(r["both"]["shap_rows"]),
        shap_background=int(r["both"]["shap_background"]),
        r2={a: r[a]["r2"] for a in arms},
        mae={a: r[a]["mae"] for a in arms},
        inherit_p90_of_p75=r["inherit_p90_of_p75"],
        inherit_p75_of_p90=r["inherit_p75_of_p90"],
        rank={c: {a: r[a]["ranks"].get(c) for a in arms}
              for c in PAIR + MEAN_FORMS},
        attribution={c: {a: r[a]["attribution"].get(c) for a in arms}
                     for c in PAIR + MEAN_FORMS})
    print("\n  the mean forms of formaldehyde, with both durations and with neither:")
    for c in MEAN_FORMS:
        print(f"      {c:<40} rank {out['rank'][c]['both']:>4} -> "
              f"{out['rank'][c]['without_both']:>4}")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"  [saved] {OUT}")
    C.write_manifest(__file__, [C.FRAME, C.MODEL_A], [OUT],
                     settings=dict(refit_shap_rows=SHAP_ROWS,
                                   refit_shap_background=SHAP_BACKGROUND))


if __name__ == "__main__":
    main()
