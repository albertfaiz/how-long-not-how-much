#!/usr/bin/env python3
"""
19_temporal_validation.py -- can the model predict a year it has not seen?

READS   the modelling panel, through _config.load_panel, so the admission gate
        and both pruning filters are inherited and nothing here sees a column
        the published model did not see
WRITES  ../derived*/temporal_validation.json
        ../derived*/temporal_validation_folds.csv

THE OBJECTION
-------------
    "Every fold of yours contains the year it predicts. The folds are grouped
     by county, which stops a county appearing on both sides, but every year
     appears on both sides. This is interpolation in time dressed as
     prediction."

That is correct about the production protocol, and the production protocol is
the right one for the question the paper asks, which is about variation across
counties. It is the wrong protocol for a different question, and a referee is
entitled to ask the different question. Three answers:

  1. LEAVE ONE YEAR OUT. Train on every other year, predict the held-out one.
     The same counties appear on both sides, deliberately: this isolates
     extrapolation in time from extrapolation across counties, which the state
     holdout of script 13 already measures.

  2. FORWARD CHAINING. Train on everything up to year t, predict t + 1, and
     walk forward. Nothing after the prediction year is ever in the training
     set, which is the only protocol that answers "could this have been used
     at the time".

  3. ERA SPLIT. Split the window in half and ask whether the paper's central
     count -- how often the non-central summary of a field carries more
     information than the central one -- is the same in both halves. This one
     is model-free and takes seconds, by design: if the finding were an
     artefact of one period it would show here before any model is fitted.

WHAT HONEST LOOKS LIKE HERE
---------------------------
Forward chaining on a seventeen-year window with a slowly moving outcome is a
hard test and the number will be lower than the headline. That is the point. A
score reported under a protocol that nobody would use is worth less than a
lower score reported under one that answers a question.

    python3 19_temporal_validation.py
    python3 19_temporal_validation.py --only loyo
    python3 19_temporal_validation.py --max-fits 4      # check the code path
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _config as C                                            # noqa: E402

OUT_JSON = C.DERIVED / "temporal_validation.json"
OUT_CSV = C.DERIVED / "temporal_validation_folds.csv"
STORE = C.DERIVED / "checkpoints_temporal"

# ===========================================================================
#  SETTINGS
# ===========================================================================
MIN_TRAIN_YEARS = 3        # forward chaining starts once this many years are
                           # available to train on. Two years cannot carry a
                           # 1,200-tree model and the fit would say nothing.
ERA_MI_SAMPLE = 20000      # rows per era for the mutual-information census.
                           # The estimator is a k-nearest-neighbour one and its
                           # cost is superlinear in n.
ERA_MI_NEIGHBOURS = 3      # the default of mutual_info_regression, named here
                           # because it moves the numbers

_BUDGET = {"left": 0, "limited": False}


def _spend():
    if not _BUDGET["limited"]:
        return True
    if _BUDGET["left"] <= 0:
        return False
    _BUDGET["left"] -= 1
    return True


# ---------------------------------------------------------------------------
#  One fit, checkpointed
# ---------------------------------------------------------------------------
def fit_predict(X, y, train, test, tag):
    """Predictions for `test` from a model fitted on `train`, cached by tag."""
    import xgboost as xgb
    STORE.mkdir(parents=True, exist_ok=True)
    f = STORE / f"{tag}.npy"
    if f.exists():
        got = np.load(f)
        if len(got) == len(test):
            print(f"      {tag:<28} from checkpoint", flush=True)
            return got
        print(f"      {tag:<28} checkpoint has {len(got)} rows, needs "
              f"{len(test)}; refitting", flush=True)
    if not _spend():
        print(f"      {tag:<28} not fitted: fit budget spent", flush=True)
        return None
    t0 = time.time()
    m = xgb.XGBRegressor(**C.XGB_PARAMS, n_jobs=C.threads())
    m.fit(X.iloc[train], y[train])
    pred = m.predict(X.iloc[test])
    np.save(f, pred)
    del m
    gc.collect()
    print(f"      {tag:<28} fitted in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return pred


def score(y_true, pred):
    from sklearn.metrics import r2_score, mean_absolute_error
    return dict(r2=float(r2_score(y_true, pred)),
                mae=float(mean_absolute_error(y_true, pred)),
                n=int(len(y_true)))


# ---------------------------------------------------------------------------
#  1. Leave one year out
# ---------------------------------------------------------------------------
def leave_one_year_out(p):
    print("\n" + "=" * 74)
    print("  1. LEAVE ONE YEAR OUT")
    print("=" * 74)
    print("  Train on every other year, predict the held-out one. The same")
    print("  counties are on both sides on purpose: this isolates")
    print("  extrapolation in time. Extrapolation across place is script 13.")
    print()
    X, y = p["X"], p["y"]
    years = p["df"][C.YEAR].to_numpy()
    rows, preds = [], np.full(len(y), np.nan)
    for yr in sorted(np.unique(years)):
        te = np.where(years == yr)[0]
        tr = np.where(years != yr)[0]
        pr = fit_predict(X, y, tr, te, f"loyo_{int(yr)}")
        if pr is None:
            continue
        preds[te] = pr
        s = score(y[te], pr)
        s.update(arm="leave_one_year_out", held_out=int(yr),
                 n_train=int(len(tr)))
        rows.append(s)
        print(f"    {int(yr)}   n {s['n']:>6}   R2 {s['r2']:+.4f}   "
              f"MAE {s['mae']:.4f}")
    if not rows:
        return dict(folds=[], note="no fold completed"), rows
    done = ~np.isnan(preds)
    pooled = score(y[done], preds[done]) if done.any() else {}
    r2s = [r["r2"] for r in rows]
    out = dict(n_years=len(rows), pooled=pooled,
               r2_mean=float(np.mean(r2s)), r2_min=float(np.min(r2s)),
               r2_max=float(np.max(r2s)),
               worst_year=int(rows[int(np.argmin(r2s))]["held_out"]),
               folds=rows)
    print(f"\n    pooled over the years that ran: R2 {pooled.get('r2', float('nan')):+.4f}"
          f"   MAE {pooled.get('mae', float('nan')):.4f}")
    print(f"    per-year R2 spans {out['r2_min']:+.4f} to {out['r2_max']:+.4f}"
          f"   (worst year {out['worst_year']})")
    return out, rows


# ---------------------------------------------------------------------------
#  2. Forward chaining
# ---------------------------------------------------------------------------
def forward_chaining(p):
    print("\n" + "=" * 74)
    print("  2. FORWARD CHAINING")
    print("=" * 74)
    print("  Train on everything up to year t, predict t + 1. Nothing after")
    print("  the prediction year is ever in the training set. This is the only")
    print("  protocol here that answers whether the model could have been")
    print("  used at the time, and it is the hardest of the three.")
    print()
    X, y = p["X"], p["y"]
    years = p["df"][C.YEAR].to_numpy()
    uniq = sorted(np.unique(years))
    rows, preds = [], np.full(len(y), np.nan)
    for i in range(MIN_TRAIN_YEARS, len(uniq)):
        target = uniq[i]
        tr = np.where(years < target)[0]
        te = np.where(years == target)[0]
        pr = fit_predict(X, y, tr, te, f"fwd_{int(target)}")
        if pr is None:
            continue
        preds[te] = pr
        s = score(y[te], pr)
        s.update(arm="forward_chaining", held_out=int(target),
                 n_train=int(len(tr)), train_years=len(uniq[:i]))
        rows.append(s)
        print(f"    train <= {int(uniq[i - 1])}  predict {int(target)}   "
              f"n {s['n']:>6}   R2 {s['r2']:+.4f}   MAE {s['mae']:.4f}")
    if not rows:
        return dict(folds=[], note="no fold completed"), rows
    done = ~np.isnan(preds)
    pooled = score(y[done], preds[done])
    r2s = [r["r2"] for r in rows]
    # does it improve as the training window grows? A positive slope says the
    # model is still data-limited in time rather than drifting.
    slope = float(np.polyfit([r["train_years"] for r in rows], r2s, 1)[0]) \
        if len(rows) > 2 else float("nan")
    out = dict(n_steps=len(rows), pooled=pooled,
               r2_first=float(r2s[0]), r2_last=float(r2s[-1]),
               r2_mean=float(np.mean(r2s)),
               slope_r2_per_train_year=slope, folds=rows)
    print(f"\n    pooled over the steps that ran: R2 {pooled['r2']:+.4f}   "
          f"MAE {pooled['mae']:.4f}")
    print(f"    first step R2 {out['r2_first']:+.4f}, last {out['r2_last']:+.4f}"
          f", trend {slope:+.4f} per extra training year")
    return out, rows


# ---------------------------------------------------------------------------
#  3. Era split, model-free
# ---------------------------------------------------------------------------
def era_split(p):
    """Is the central count the same in the first half of the window and the
    second? Mutual information, so no model is fitted and no attribution
    method is involved."""
    from sklearn.feature_selection import mutual_info_regression
    print("\n" + "=" * 74)
    print("  3. ERA SPLIT, WITH NO MODEL IN IT")
    print("=" * 74)
    print("  The paper's central count, recomputed inside each half of the")
    print("  window using mutual information alone. If the finding belonged to")
    print("  one period rather than to the data, it would show here, before")
    print("  any model is fitted and without any attribution method.")
    print()
    df, X = p["df"], p["X"]
    years = df[C.YEAR].to_numpy()
    mid = float(np.median(years))
    eras = {"early": years <= mid, "late": years > mid}
    print(f"    window split at {mid:g}: "
          + ", ".join(f"{k} n={int(v.sum())}" for k, v in eras.items()))

    pairs = C.census_pairs(columns=X.columns)
    print(f"    {len(pairs)} fields have a central and a non-central column "
          f"in the model")

    rng = C.seed_everything(extra=19)
    rows, per_era = [], {}
    for era, mask in eras.items():
        idx = np.where(mask)[0]
        if len(idx) > ERA_MI_SAMPLE:
            idx = rng.choice(idx, ERA_MI_SAMPLE, replace=False)
        yy = p["y"][idx]
        wins, got = 0, 0
        for fieldname, c_col, n_col in pairs:
            sub = X.iloc[idx][[c_col, n_col]].to_numpy()
            mi = mutual_info_regression(
                sub, yy, n_neighbors=ERA_MI_NEIGHBOURS,
                random_state=C.SEED)
            got += 1
            win = bool(mi[1] > mi[0])
            wins += int(win)
            rows.append(dict(arm="era_split", era=era, field=fieldname,
                             central_mi=float(mi[0]), noncentral_mi=float(mi[1]),
                             noncentral_wins=win))
        per_era[era] = dict(n_rows=int(len(idx)), n_fields=got,
                            noncentral_wins=wins,
                            share=float(wins / got) if got else float("nan"))
        print(f"    {era:<6} n {len(idx):>6}   non-central carries more "
              f"information in {wins} of {got}")

    a, b = per_era.get("early", {}), per_era.get("late", {})
    out = dict(split_at=mid, eras=per_era)
    if a and b:
        # Two independent counts of the same quantity. A two-sided Fisher
        # exact test on the 2x2 of (era) by (which description wins) says
        # whether the two halves disagree by more than sampling.
        from scipy.stats import fisher_exact
        table = [[a["noncentral_wins"], a["n_fields"] - a["noncentral_wins"]],
                 [b["noncentral_wins"], b["n_fields"] - b["noncentral_wins"]]]
        odds, pv = fisher_exact(table)
        out.update(fisher_odds=float(odds), fisher_p=float(pv),
                   share_difference=float(a["share"] - b["share"]))
        print(f"\n    difference between halves {out['share_difference']:+.3f}"
              f"   Fisher exact p = {pv:.4f}")
        print("    A large p here is the result worth having: it says the two")
        print("    halves of the window do not disagree about which")
        print("    description carries the information.")
    return out, rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="comma separated: loyo, forward, era")
    ap.add_argument("--max-fits", type=int, default=0,
                    help="stop after this many models are FITTED in this run. "
                         "Checkpointed fits from an earlier run do not count, "
                         "so the same command run again continues.")
    a = ap.parse_args()
    want = (["loyo", "forward", "era"] if a.only == "all"
            else [s.strip() for s in a.only.split(",")])
    if a.max_fits:
        _BUDGET["left"], _BUDGET["limited"] = a.max_fits, True

    C.banner("19  can the model predict a year it has not seen")
    C.seed_everything()
    C.DERIVED.mkdir(parents=True, exist_ok=True)
    p = C.load_panel()
    lo, hi = p["window"]
    print(f"  panel {p['X'].shape[0]:,} rows x {p['X'].shape[1]} columns, "
          f"years {lo} to {hi}")

    blob, rows = {}, []
    if "loyo" in want:
        out, r = leave_one_year_out(p)
        blob["leave_one_year_out"], rows = out, rows + r
    if "forward" in want:
        out, r = forward_chaining(p)
        blob["forward_chaining"], rows = out, rows + r
    if "era" in want:
        out, r = era_split(p)
        blob["era_split"], rows = out, rows + r

    # keep whatever an earlier partial run wrote
    if OUT_JSON.exists():
        try:
            old = json.load(open(OUT_JSON))
            for k, v in old.items():
                blob.setdefault(k, v)
        except Exception:
            pass
    blob["window"] = [int(lo), int(hi)]
    blob["n_rows"] = int(len(p["y"]))
    blob["n_features"] = int(p["X"].shape[1])
    json.dump(blob, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\n  [saved] {OUT_JSON}")
    if rows:
        new = pd.DataFrame(rows)
        if OUT_CSV.exists():
            try:
                prev = pd.read_csv(OUT_CSV)
                keys = [k for k in ("arm", "held_out", "era", "field")
                        if k in new.columns and k in prev.columns]
                new = (pd.concat([prev, new], ignore_index=True)
                       .drop_duplicates(subset=keys, keep="last"))
            except Exception:
                pass
        new.to_csv(OUT_CSV, index=False)
        print(f"  [saved] {OUT_CSV}")

    print("\n" + "-" * 74)
    print("  HOW TO READ THESE THREE NUMBERS TOGETHER")
    print("-" * 74)
    print("  The headline score is county-grouped and answers a question about")
    print("  variation across place. Leave-one-year-out answers a question")
    print("  about time with place held constant. Forward chaining answers")
    print("  both at once and under the hardest constraint, and is expected to")
    print("  be the lowest of the three. None of them is the others' check:")
    print("  they are three different questions and the paper should say which")
    print("  one each number answers.")

    C.write_manifest("19_temporal_validation.py",
                     inputs=[C.FRAME, C.MODEL_A],
                     outputs=[OUT_JSON, OUT_CSV],
                     settings=dict(MIN_TRAIN_YEARS=MIN_TRAIN_YEARS,
                                   ERA_MI_SAMPLE=ERA_MI_SAMPLE,
                                   ERA_MI_NEIGHBOURS=ERA_MI_NEIGHBOURS,
                                   arms=want),
                     notes="Temporal validation: leave one year out, forward "
                           "chaining, and a model-free era split of the "
                           "central count.")


if __name__ == "__main__":
    main()
