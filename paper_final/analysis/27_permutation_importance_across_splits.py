#!/usr/bin/env python3
"""
27_permutation_importance_across_splits.py -- how stable is loss-based
importance when the comparison is between two near-duplicate columns?

READS   the modelling panel, the census pairs
WRITES  ../derived*/permutation_across_splits.json
        ../derived*/permutation_across_splits.csv

WHY THIS EXISTS
---------------
Script 21 measured permutation importance on one county-disjoint held-out
split. Two runs of it gave the second learner 22 of 47 and 36 of 47, because
the held-out counties depended on which other arms had consumed the random
stream first. A third run reported 23 of 47 for the production learner, from a
model checkpoint written by xgboost 3.2 and read by xgboost 3.0, whose held-out
R2 was -969: a corrupted load, not a measurement.

Both of those are reasons to stop treating one split as an answer. This script
draws several county-disjoint splits, each from its own seed so no split
depends on anything else that ran, fits both learners fresh on each, checks
that each fit predicts sensibly before using it, and reports the distribution
of the count. If loss-based importance between near-duplicate columns is a
stable quantity, the count will be stable across splits. If it is a difference
of two small noisy numbers, it will not be.

    python3 27_permutation_importance_across_splits.py
    python3 27_permutation_importance_across_splits.py --splits 3
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from math import comb

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _config as C                                            # noqa: E402

OUT_JSON = C.DERIVED / "permutation_across_splits.json"
OUT_CSV = C.DERIVED / "permutation_across_splits.csv"
STORE = C.DERIVED / "checkpoints_perm_splits"

N_SPLITS = 5
HELDOUT_FRACTION = 0.25
PERM_ROWS = 12000
PERM_REPEATS = 5
MIN_HELDOUT_R2 = 0.30     # a fit below this is not used: it is broken, not
                          # weak, since every model in this paper scores
                          # above 0.6 on held-out counties


def sign_test_p(wins, n):
    if not n:
        return float("nan")
    return float(sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n)


def grouped_split(groups, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    out = set(rng.choice(uniq, int(round(len(uniq) * HELDOUT_FRACTION)),
                         replace=False).tolist())
    mask = np.fromiter((g in out for g in groups), bool, len(groups))
    return np.where(~mask)[0], np.where(mask)[0], rng


def permutation_drop(predict, X_out, y_out, cols, rng):
    from sklearn.metrics import r2_score
    base = r2_score(y_out, predict(X_out))
    out = {}
    for col in cols:
        saved = X_out[col].to_numpy().copy()
        drops = []
        for _ in range(PERM_REPEATS):
            X_out[col] = saved[rng.permutation(len(saved))]
            drops.append(base - r2_score(y_out, predict(X_out)))
        X_out[col] = saved
        out[col] = float(np.mean(drops))
    return out, float(base)


def count(vals, pairs):
    wins = got = 0
    for _f, c, n in pairs:
        if c in vals and n in vals:
            got += 1
            wins += int(vals[n] > vals[c])
    return got, wins


def one_split(k, p, pairs, cols):
    import xgboost as xgb
    from sklearn.ensemble import HistGradientBoostingRegressor
    X, y, groups = p["X"], p["y"], p["groups"]
    tr, te, rng = grouped_split(groups, C.SEED + 100 + k)
    if len(te) > PERM_ROWS:
        te = rng.choice(te, PERM_ROWS, replace=False)
    X_out, y_out = X.iloc[te].copy(), y[te]
    rec = dict(split=k, seed=C.SEED + 100 + k, n_train=int(len(tr)),
               n_heldout=int(len(te)))

    t0 = time.time()
    m = xgb.XGBRegressor(**C.XGB_PARAMS, n_jobs=C.threads())
    m.fit(X.iloc[tr], y[tr])
    vals, r2 = permutation_drop(m.predict, X_out, y_out, cols, rng)
    got, wins = count(vals, pairs)
    rec.update(production_r2=r2, production_n=got, production_wins=wins,
               production_p=sign_test_p(wins, got),
               production_valid=bool(r2 >= MIN_HELDOUT_R2))
    del m
    gc.collect()

    h = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.06, max_leaf_nodes=63,
        min_samples_leaf=20, l2_regularization=1.0, early_stopping=False,
        random_state=C.SEED)
    h.fit(X.iloc[tr], y[tr])
    vals, r2 = permutation_drop(h.predict, X_out, y_out, cols, rng)
    got, wins = count(vals, pairs)
    rec.update(second_r2=r2, second_n=got, second_wins=wins,
               second_p=sign_test_p(wins, got),
               second_valid=bool(r2 >= MIN_HELDOUT_R2),
               minutes=round((time.time() - t0) / 60, 2))
    del h
    gc.collect()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--max-new", type=int, default=0,
                    help="stop after this many splits are computed in this run")
    a = ap.parse_args()

    C.banner("27  loss-based importance across held-out splits")
    STORE.mkdir(parents=True, exist_ok=True)
    p = C.load_panel()
    pairs = C.census_pairs(columns=p["X"].columns)
    cols = sorted({c for _f, lo, hi in pairs for c in (lo, hi)})
    print(f"  {len(pairs)} pairs, {len(cols)} columns, {a.splits} splits")

    rows, new = [], 0
    for k in range(1, a.splits + 1):
        ck = STORE / f"split{k:02d}.json"
        if ck.exists():
            rows.append(json.load(open(ck)))
            print(f"    split {k}: from checkpoint")
            continue
        if a.max_new and new >= a.max_new:
            print(f"    split {k}: not computed in this run")
            continue
        rec = one_split(k, p, pairs, cols)
        json.dump(rec, open(ck, "w"), indent=1)
        rows.append(rec)
        new += 1
        print(f"    split {k}: production {rec['production_wins']}/"
              f"{rec['production_n']} (R2 {rec['production_r2']:.3f}), "
              f"second {rec['second_wins']}/{rec['second_n']} "
              f"(R2 {rec['second_r2']:.3f})  {rec['minutes']} min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    summary = {}
    for who in ("production", "second"):
        ok = df[df[f"{who}_valid"]] if f"{who}_valid" in df else df
        if len(ok):
            w = ok[f"{who}_wins"].astype(int)
            summary[who] = dict(
                n_splits=int(len(ok)), wins_min=int(w.min()),
                wins_max=int(w.max()), wins_median=float(w.median()),
                n_comparable=int(ok[f"{who}_n"].iloc[0]),
                r2_min=float(ok[f"{who}_r2"].min()),
                r2_max=float(ok[f"{who}_r2"].max()),
                splits_significant=int((ok[f"{who}_p"] < 0.05).sum()))
            s = summary[who]
            print(f"\n  {who}: {s['wins_min']} to {s['wins_max']} of "
                  f"{s['n_comparable']} across {s['n_splits']} splits "
                  f"(held-out R2 {s['r2_min']:.3f} to {s['r2_max']:.3f}), "
                  f"significant in {s['splits_significant']}")
    blob = dict(n_splits_requested=a.splits, n_splits_done=int(len(df)),
                heldout_fraction=HELDOUT_FRACTION, perm_rows=PERM_ROWS,
                perm_repeats=PERM_REPEATS, min_heldout_r2=MIN_HELDOUT_R2,
                summary=summary, splits=rows,
                note="Each split drawn from its own seed and both learners "
                     "fitted fresh on it. A fit below the minimum held-out R2 "
                     "is reported and excluded from the summary.")
    json.dump(blob, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\n  [saved] {OUT_JSON}")
    C.write_manifest("27_permutation_importance_across_splits.py",
                     inputs=[C.FRAME, C.MODEL_A], outputs=[OUT_JSON, OUT_CSV],
                     settings=dict(N_SPLITS=a.splits,
                                   HELDOUT_FRACTION=HELDOUT_FRACTION,
                                   PERM_ROWS=PERM_ROWS,
                                   PERM_REPEATS=PERM_REPEATS))


if __name__ == "__main__":
    main()
