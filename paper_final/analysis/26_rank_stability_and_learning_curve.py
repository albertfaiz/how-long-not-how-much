#!/usr/bin/env python3
"""
26_rank_stability_and_learning_curve.py -- how much does the ranking move, and
is the model still short of data?

READS   the fitted model and the modelling panel
WRITES  ../derived*/rank_stability.json
        ../derived*/rank_stability_draws.csv

TWO OBJECTIONS
--------------
    --only rank, --only reseed
        "Your ranking is one draw. Resample the counties and the top ten will
         be a different top ten."
    The ranking is a mean over rows, so resampling counties and recomputing it
    costs one attribution pass rather than one per draw: the per-row absolute
    contributions are computed once and the county-level bootstrap is a
    reweighted average over them. A thousand draws then take seconds, and the
    rank of every leading variable comes with a distribution instead of a
    point. The reseed arm is the other half of the same question: refit with a
    different random seed and see whether the ranking moves more than the
    bootstrap says it should.

    --only curve
        "With three thousand counties and five hundred and eighty one columns
         you are at the point where more data would change the answer."
    Fit on a tenth of the counties, a quarter, a half, three quarters and all
    of them. If the curve is still climbing at the right-hand end, the model
    is data-limited and the paper should say so rather than present the score
    as a ceiling.

WHAT THE BOOTSTRAP ARM HOLDS FIXED
----------------------------------
The fitted model. It resamples which counties are ATTRIBUTED, not which were
trained on, so it measures the sampling variability of the ranking given this
model. Variability across refits is what the reseed arm measures, and the two
are reported side by side because a bootstrap interval that is much narrower
than the refit spread would be a misleading interval.

    python3 26_rank_stability_and_learning_curve.py --only rank
    python3 26_rank_stability_and_learning_curve.py --only curve --curve-folds 4
"""
from __future__ import annotations

import argparse
import gc
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

OUT_JSON = C.DERIVED / "rank_stability.json"
OUT_CSV = C.DERIVED / "rank_stability_draws.csv"
STORE = C.DERIVED / "checkpoints_rank"

# ===========================================================================
#  SETTINGS
# ===========================================================================
ATTRIB_ROWS = 20000      # rows attributed once, matching the paper's
                         # SHAP_SAMPLE so the depth is comparable
BOOT_DRAWS = 1000        # county-level bootstrap draws over those rows
TOPK = (5, 10, 15, 25)   # overlap reported at each depth
RESEEDS = (1, 2, 3)      # extra random seeds for the refit arm
CURVE_SHARES = (0.10, 0.25, 0.50, 0.75, 1.00)
CURVE_FOLDS = 4          # fewer than the paper's eight, on purpose: a
                         # learning curve is read for its shape, and five
                         # points at eight folds costs three hours to answer a
                         # question four folds answers. Written into the file.


def path_abs_contribs(model, X, idx):
    """Per-row absolute path-dependent contributions, as a float32 matrix."""
    import xgboost as xgb
    sub = X.iloc[idx]
    dm = xgb.DMatrix(sub, feature_names=list(sub.columns))
    contrib = np.asarray(
        model.get_booster().predict(dm, pred_contribs=True))[:, :-1]
    return np.abs(contrib).astype("float32"), list(sub.columns)


# ---------------------------------------------------------------------------
#  1. The bootstrap over counties, with the model held fixed
# ---------------------------------------------------------------------------
def rank_bootstrap(p, model, rng, args):
    print("\n" + "=" * 74)
    print("  1. THE RANKING, RESAMPLED OVER COUNTIES")
    print("=" * 74)
    print("  The per-row contributions are computed once. Each bootstrap draw")
    print("  resamples counties with replacement and re-averages, so a")
    print("  thousand draws cost one attribution pass rather than a thousand.")
    print()
    X, groups = p["X"], p["groups"]
    n = len(X)
    idx = rng.choice(n, min(ATTRIB_ROWS, n), replace=False) if n > ATTRIB_ROWS \
        else np.arange(n)
    t0 = time.time()
    A, cols = path_abs_contribs(model, X, idx)
    print(f"    attributed {A.shape[0]:,} rows x {A.shape[1]} columns in "
          f"{(time.time() - t0) / 60:.1f} min")

    g = groups[idx]
    uniq, inverse = np.unique(g, return_inverse=True)
    print(f"    {len(uniq):,} counties among the attributed rows")

    # County sums and counts, so a draw is a fancy-indexed sum over 3,000
    # rows rather than a gather of 20,000. Resampling whole counties and
    # averaging the rows gives exactly this, and it is three orders of
    # magnitude faster.
    sums = np.zeros((len(uniq), A.shape[1]), "float64")
    np.add.at(sums, inverse, A.astype("float64"))
    counts = np.bincount(inverse, minlength=len(uniq)).astype("float64")

    base = A.mean(axis=0)
    base_order = np.argsort(-base)
    base_rank = np.empty(len(cols), int)
    base_rank[base_order] = np.arange(1, len(cols) + 1)

    draws = args.boot_draws or BOOT_DRAWS
    ranks = np.empty((draws, len(cols)), "int32")
    t0 = time.time()
    for b in range(draws):
        pick = rng.integers(0, len(uniq), len(uniq))
        m = sums[pick].sum(axis=0) / counts[pick].sum()
        order = np.argsort(-m)
        r = np.empty(len(cols), "int32")
        r[order] = np.arange(1, len(cols) + 1)
        ranks[b] = r
        if (b + 1) % 200 == 0:
            print(f"      {b + 1}/{draws} draws, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)

    # overlap of the leading set with the point-estimate leading set
    overlap = {}
    for k in TOPK:
        top0 = set(base_order[:k].tolist())
        shares = [len(top0 & set(np.argsort(ranks[b])[:k].tolist())) / k
                  for b in range(draws)]
        overlap[f"top{k}"] = dict(
            mean_share=float(np.mean(shares)),
            min_share=float(np.min(shares)),
            share_fully_identical=float(np.mean(np.asarray(shares) == 1.0)))
        print(f"    top {k:>2}: on average "
              f"{100 * overlap[f'top{k}']['mean_share']:.1f} per cent of the "
              f"leading set is the same, identical in "
              f"{100 * overlap[f'top{k}']['share_fully_identical']:.1f} "
              f"per cent of draws")

    # the per-variable rank distribution, for the leading variables
    rows_out = []
    lead = base_order[:max(TOPK)]
    for j in lead:
        rj = ranks[:, j]
        rows_out.append(dict(
            arm="county_bootstrap", feature=cols[j],
            point_rank=int(base_rank[j]), point_value=float(base[j]),
            rank_median=float(np.median(rj)),
            rank_p025=float(np.percentile(rj, 2.5)),
            rank_p975=float(np.percentile(rj, 97.5)),
            share_in_top10=float((rj <= 10).mean()),
            share_rank_1=float((rj == 1).mean())))
    print()
    print(f"    {'variable':<46} rank  95% interval  P(top 10)  P(first)")
    for r in rows_out[:12]:
        print(f"    {r['feature'][:46]:<46} {r['point_rank']:>4}  "
              f"{r['rank_p025']:>5.0f}-{r['rank_p975']:<5.0f}  "
              f"{r['share_in_top10']:>8.2f}  {r['share_rank_1']:>7.2f}")
    first = rows_out[0] if rows_out else {}
    if first:
        print()
        print(f"    The variable the paper ranks first holds first place in "
              f"{100 * first['share_rank_1']:.0f} per cent")
        print("    of county resamples. That number, not the point rank,")
        print("    is what should appear beside the ranking in the text.")
    return dict(n_rows_attributed=int(A.shape[0]), n_counties=int(len(uniq)),
                n_draws=int(draws),
                attribution="path dependent TreeSHAP, model held fixed",
                topk_overlap=overlap, leading=rows_out,
                note="Resamples which counties are attributed, not which were "
                     "trained on."), rows_out


# ---------------------------------------------------------------------------
#  2. Refit under other seeds
# ---------------------------------------------------------------------------
def reseed(p, rng, args):
    print("\n" + "=" * 74)
    print("  2. REFIT UNDER OTHER SEEDS")
    print("=" * 74)
    print("  The same data, the same settings, a different random_state. This")
    print("  is the variability the bootstrap above cannot see, because the")
    print("  bootstrap holds the fitted model fixed.")
    print()
    import xgboost as xgb
    STORE.mkdir(parents=True, exist_ok=True)
    X, y = p["X"], p["y"]
    idx = (rng.choice(len(X), min(8000, len(X)), replace=False))
    base = C.load_table("A_shap_ranking")
    base_top = {k: list(base.sort_values("rank")["feature"].astype(str)[:k])
                for k in TOPK}
    out, rows = {}, []
    for s in (args.reseeds or RESEEDS):
        path = STORE / f"reseed_{s}.json"
        params = dict(C.XGB_PARAMS)
        params["random_state"] = int(C.SEED) + int(s)
        m = xgb.XGBRegressor(**params, n_jobs=C.threads())
        if path.exists():
            m.load_model(str(path))
            print(f"    seed +{s}: model from checkpoint")
        else:
            t0 = time.time()
            m.fit(X, y)
            m.save_model(str(path))
            print(f"    seed +{s}: fitted in {(time.time() - t0) / 60:.1f} min")
        A, cols = path_abs_contribs(m, X, idx)
        mean = pd.Series(A.mean(axis=0), index=cols).sort_values(
            ascending=False)
        rec = {}
        for k in TOPK:
            shared = len(set(base_top[k]) & set(mean.index[:k]))
            rec[f"top{k}_shared"] = shared
            rec[f"top{k}"] = k
        rec["first"] = str(mean.index[0])
        rec["first_matches_published"] = bool(
            mean.index[0] == base_top[max(TOPK)][0])
        out[f"seed_plus_{s}"] = rec
        rows.append(dict(arm="reseed", seed_offset=int(s), **rec))
        print(f"      top 15 overlap {rec['top15_shared']}/15, "
              f"first is {rec['first'][:44]}")
        del m, A
        gc.collect()
    print()
    print("    An overlap here that is materially worse than the bootstrap")
    print("    interval above would mean the ranking's uncertainty is")
    print("    dominated by the fit rather than by the sample, and the")
    print("    bootstrap interval alone would understate it.")
    return dict(seeds=out,
                published_top=base_top,
                attribution="path dependent TreeSHAP on each refit, compared "
                            "with the published interventional ranking, so "
                            "the comparison is a direction check"), rows


# ---------------------------------------------------------------------------
#  3. The learning curve
# ---------------------------------------------------------------------------
def learning_curve(p, rng, args):
    folds = args.curve_folds or CURVE_FOLDS
    print("\n" + "=" * 74)
    print("  3. THE LEARNING CURVE")
    print("=" * 74)
    print(f"  County-grouped cross-validation at {folds} folds rather than the")
    print(f"  paper's {C.N_FOLDS}. A learning curve is read for its shape and")
    print("  five points at eight folds costs hours to answer what four folds")
    print("  answer. The fold count is in the result file, and no number from")
    print("  this arm is comparable with the headline score.")
    print()
    X, y, groups = p["X"], p["y"], p["groups"]
    uniq = np.unique(groups)
    rows, out = [], {}
    for share in (args.curve_shares or CURVE_SHARES):
        k = max(int(round(len(uniq) * share)), 2 * folds)
        pick = set(rng.choice(uniq, min(k, len(uniq)), replace=False).tolist())
        m = np.fromiter((g in pick for g in groups), bool, len(groups))
        s = C.cv_score(X[m], y[m], groups[m],
                       f"curve_{int(100 * share):03d}_f{folds}",
                       n_folds=folds)
        s.update(arm="learning_curve", county_share=float(share),
                 n_counties=int(len(pick)), n_rows=int(m.sum()),
                 n_folds=int(folds))
        out[f"{int(100 * share):03d}pct"] = s
        rows.append(s)
        print(f"    {100 * share:5.0f} per cent of counties  "
              f"{len(pick):>5} counties  {int(m.sum()):>6} rows   "
              f"R2 {s['r2']:+.4f}   MAE {s['mae']:.4f}")
    if len(rows) >= 3:
        xs = np.log([r["n_rows"] for r in rows])
        ys = np.array([r["r2"] for r in rows])
        slope = float(np.polyfit(xs, ys, 1)[0])
        last = ys[-1] - ys[-2]
        print(f"\n    R2 rises {slope:+.4f} per e-fold of rows over the whole")
        print(f"    curve, and {last:+.4f} between the last two points.")
        print("    A last step that is still clearly positive means the model")
        print("    is data-limited and the score should not be presented as a")
        print("    ceiling. A last step near zero means it has flattened.")
        out["slope_r2_per_log_row"] = slope
        out["last_step_r2"] = float(last)
    out["n_folds"] = int(folds)
    out["note"] = (f"Fitted at {folds} folds, not the paper's {C.N_FOLDS}. "
                   f"Read for shape only.")
    return out, rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="comma separated: rank, reseed, curve")
    ap.add_argument("--boot-draws", type=int, default=0)
    ap.add_argument("--reseeds", type=int, nargs="*", default=None)
    ap.add_argument("--curve-folds", type=int, default=0)
    ap.add_argument("--curve-shares", type=float, nargs="*", default=None)
    a = ap.parse_args()
    want = (["rank", "reseed", "curve"] if a.only == "all"
            else [s.strip() for s in a.only.split(",")])

    C.banner("26  how much the ranking moves, and whether more data would help")
    C.seed_everything()
    C.DERIVED.mkdir(parents=True, exist_ok=True)
    rng = C.seed_everything(extra=29)
    p = C.load_panel()
    print(f"  panel {p['X'].shape[0]:,} rows x {p['X'].shape[1]} columns")

    blob, rows = {}, []
    if "rank" in want:
        import joblib
        model = joblib.load(C.MODEL_A)["model"]
        out, r = rank_bootstrap(p, model, rng, a)
        blob["topk_overlap"], rows = out, rows + r
        del model
        gc.collect()
    if "reseed" in want:
        out, r = reseed(p, rng, a)
        blob["reseed"], rows = out, rows + r
    if "curve" in want:
        out, r = learning_curve(p, rng, a)
        blob["learning_curve"], rows = out, rows + r

    if OUT_JSON.exists():
        try:
            old = json.load(open(OUT_JSON))
            for k, v in old.items():
                blob.setdefault(k, v)
        except Exception:
            pass
    json.dump(blob, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\n  [saved] {OUT_JSON}")
    if rows:
        df = pd.DataFrame(rows)
        if OUT_CSV.exists():
            try:
                prev = pd.read_csv(OUT_CSV)
                df = pd.concat([prev, df], ignore_index=True)
                keys = [k for k in ("arm", "feature", "seed_offset",
                                    "county_share") if k in df.columns]
                df = df.drop_duplicates(subset=keys, keep="last")
            except Exception:
                pass
        df.to_csv(OUT_CSV, index=False)
        print(f"  [saved] {OUT_CSV}")

    C.write_manifest("26_rank_stability_and_learning_curve.py",
                     inputs=[C.MODEL_A, C.FRAME, C.RANKING_A],
                     outputs=[OUT_JSON, OUT_CSV],
                     settings=dict(arms=want, ATTRIB_ROWS=ATTRIB_ROWS,
                                   BOOT_DRAWS=a.boot_draws or BOOT_DRAWS,
                                   TOPK=list(TOPK),
                                   RESEEDS=list(a.reseeds or RESEEDS),
                                   CURVE_FOLDS=a.curve_folds or CURVE_FOLDS,
                                   CURVE_SHARES=list(a.curve_shares
                                                     or CURVE_SHARES)),
                     notes="Rank stability under a county bootstrap and under "
                           "refits, and the learning curve over county "
                           "subsets.")


if __name__ == "__main__":
    main()
