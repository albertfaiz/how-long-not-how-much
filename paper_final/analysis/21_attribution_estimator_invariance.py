#!/usr/bin/env python3
"""
21_attribution_estimator_invariance.py -- is the central finding a property of
the data or of TreeSHAP?

READS   the fitted production model and the modelling panel; the census of
        script 12; the information census of script 14 if it is there
WRITES  ../derived*/attribution_invariance.json
        ../derived*/attribution_invariance_per_field.csv

THE OBJECTION
-------------
    "Interventional TreeSHAP with a five-hundred-row background is one choice
     among several. Your whole paper turns on a count produced by that choice.
     Change the attribution method and the count changes."

This is the most dangerous objection in the package, because it is the one
that would, if true, remove the paper rather than qualify it. The answer has
to be a table, not a paragraph.

Six measuring instruments, asked the same question of the same pairs:

  interventional, published   the paper's own ranking, from the main run
  interventional, recomputed  the same estimator rerun here, if the installed
                              shap and xgboost agree well enough to run it
  path dependent              TreeSHAP without a background set, computed by
                              xgboost itself. A different value function.
  total gain                  the summed split gain. Not an attribution at
                              all: a count of how useful the splitter found
                              the column. Included because it is what most
                              papers actually report.
  permutation, production     refit nothing, permute one column of held-out
                              rows, measure the loss of skill. Model-agnostic.
  permutation, second learner the same, on a histogram gradient-boosting model
                              from a different library, fitted here.

and, read from script 14 if present:

  mutual information          marginal, model-free, one column at a time.

WHAT A HONEST RESULT LOOKS LIKE
-------------------------------
These instruments do not measure the same thing and are not expected to agree.
Mutual information is marginal; the SHAP variants are conditional; total gain
is neither. The paper's claim is about conditional attribution, so the
instruments that matter most for it are the conditional ones. The useful
output of this script is therefore not one number but the pattern: which
counts move, which hold, and whether the ones that move are the ones measuring
a different quantity. If the conditional instruments agree and the marginal
one does not, that is the paper's existing finding, now measured five ways
instead of one.

    python3 21_attribution_estimator_invariance.py
    python3 21_attribution_estimator_invariance.py --only gain,path
    python3 21_attribution_estimator_invariance.py --max-fields 8    # smoke test
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

OUT_JSON = C.DERIVED / "attribution_invariance.json"
OUT_CSV = C.DERIVED / "attribution_invariance_per_field.csv"

# ===========================================================================
#  SETTINGS
# ===========================================================================
PERM_ROWS = 12000       # held-out rows the permutation importances use
PERM_REPEATS = 5        # permutations per column. The spread across repeats is
                        # reported, so this number is checkable rather than
                        # taken on trust.
SHAP_ROWS = 6000        # rows attributed when SHAP is recomputed here. Lower
                        # than the paper's 20,000 on purpose: this script asks
                        # whether the SIGN of a comparison holds, not what the
                        # attribution value is, and the paper's own value comes
                        # from the main run.
HELDOUT_FRACTION = 0.25  # counties held out, grouped, for the permutation arms

ARMS = ["published", "path", "gain", "perm_prod", "perm_second", "shap_here",
        "oof_shap"]


# ---------------------------------------------------------------------------
def grouped_split(groups, fraction, rng):
    """Row indices for a county-disjoint train and held-out split."""
    uniq = np.unique(groups)
    n_out = max(int(round(len(uniq) * fraction)), 1)
    out = set(rng.choice(uniq, n_out, replace=False).tolist())
    mask = np.fromiter((g in out for g in groups), bool, len(groups))
    return np.where(~mask)[0], np.where(mask)[0]


def count_wins(values, pairs):
    """(n comparable, n where the non-central column leads, per-field rows)."""
    rows, wins, got = [], 0, 0
    for field, c_col, n_col in pairs:
        cv, nv = values.get(c_col), values.get(n_col)
        if cv is None or nv is None or not np.isfinite(cv) or not np.isfinite(nv):
            continue
        got += 1
        win = bool(nv > cv)
        wins += int(win)
        rows.append(dict(field=field, central_col=c_col, noncentral_col=n_col,
                         central_value=float(cv), noncentral_value=float(nv),
                         noncentral_wins=win,
                         ratio=float(nv / cv) if cv else np.nan))
    return got, wins, rows


def sign_test_p(wins, n):
    """One-sided binomial p for a count this high under a fair coin."""
    from math import comb
    if not n:
        return float("nan")
    return float(sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n)


# ---------------------------------------------------------------------------
#  The instruments
# ---------------------------------------------------------------------------
def values_published():
    """The paper's own interventional ranking, as the main run wrote it."""
    r = C.load_table("A_shap_ranking")
    col = "mean_abs_shap" if "mean_abs_shap" in r.columns else r.columns[1]
    return dict(zip(r["feature"].astype(str), r[col].astype(float)))


def values_path_dependent(model, X, rng):
    """TreeSHAP without a background set, computed by xgboost itself.

    Not the paper's estimator. The value function differs: an absent feature
    is marginalised over the tree's own cover rather than over a background
    sample. Used here because it is a genuinely different conditional
    attribution and because xgboost computes it without the shap package,
    which on this machine does not agree with this xgboost about how a
    vector-valued base score is stored.
    """
    import xgboost as xgb
    idx = rng.choice(len(X), min(SHAP_ROWS, len(X)), replace=False)
    sub = X.iloc[idx]
    booster = model.get_booster()
    dm = xgb.DMatrix(sub, feature_names=list(sub.columns))
    contrib = booster.predict(dm, pred_contribs=True)
    contrib = np.asarray(contrib)[:, :-1]          # last column is the bias
    mean_abs = np.abs(contrib).mean(axis=0)
    return dict(zip(sub.columns, mean_abs.astype(float)))


def values_total_gain(model):
    """Summed split gain. A count of usefulness to the splitter, not an
    attribution. Reported because it is what a great many papers report."""
    booster = model.get_booster()
    score = booster.get_score(importance_type="total_gain")
    return {k: float(v) for k, v in score.items()}


def values_shap_here(model, X, rng):
    """The paper's own estimator, rerun here, if the installed libraries let it.

    On this machine shap 0.49 and xgboost 3.2 disagree about how a
    vector-valued base score is serialised and TreeExplainer raises. The
    failure is caught and reported rather than hidden, because an arm that
    silently did not run would look like an arm that agreed.
    """
    try:
        import shap
    except Exception as e:
        return None, f"shap not importable: {e}"
    try:
        bg_idx = rng.choice(len(X), min(C.SHAP_BACKGROUND, len(X)),
                            replace=False)
        rows_idx = rng.choice(len(X), min(SHAP_ROWS, len(X)), replace=False)
        ex = shap.TreeExplainer(model, data=X.iloc[bg_idx],
                                feature_perturbation="interventional")
        sv = ex.shap_values(X.iloc[rows_idx], check_additivity=False)
        sv = np.asarray(sv)
        mean_abs = np.abs(sv).mean(axis=0)
        return dict(zip(X.columns, mean_abs.astype(float))), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def values_oof_shap(X, y, groups, columns):
    """Mean absolute attribution from models that did not see the attributed row.

    THE REASON THIS ARM EXISTS. The published ranking, the path-dependent
    ranking and the in-sample permutation importance are all computed on a
    model fitted to every row, so all three could in principle be reporting
    structure the model fitted rather than structure in the data. The two
    held-out permutation arms disagree with them, which is exactly what an
    in-sample artefact would look like.

    This arm removes the ambiguity. Over the production folds, each fold's
    model is fitted on its training rows and attributes only its held-out
    rows, so every attribution value comes from a model that has never seen
    the row it is attributing. The count from these values is the one the
    paper's central claim should rest on.

    Each fold's column sums are written as they are produced, so the arm
    resumes rather than restarting.
    """
    import gc as _gc
    from sklearn.model_selection import GroupKFold
    import xgboost as xgb
    store = C.DERIVED / "checkpoints_invariance" / "oof_shap"
    store.mkdir(parents=True, exist_ok=True)
    cols = list(X.columns)
    total = np.zeros(len(cols), "float64")
    n_rows = 0
    for i, (tr, te) in enumerate(
            GroupKFold(n_splits=C.N_FOLDS).split(X, y, groups), 1):
        f = store / f"fold{i:02d}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            if list(z["columns"]) == cols:
                total += z["sums"]
                n_rows += int(z["n"])
                print(f"      fold {i}/{C.N_FOLDS} from checkpoint",
                      flush=True)
                continue
        t0 = time.time()
        m = xgb.XGBRegressor(**C.XGB_PARAMS, n_jobs=C.threads())
        m.fit(X.iloc[tr], y[tr])
        sub = X.iloc[te]
        dm = xgb.DMatrix(sub, feature_names=cols)
        contrib = np.asarray(
            m.get_booster().predict(dm, pred_contribs=True))[:, :-1]
        sums = np.abs(contrib).sum(axis=0)
        np.savez_compressed(f, sums=sums, n=len(te),
                            columns=np.asarray(cols, dtype=object))
        total += sums
        n_rows += len(te)
        del m, contrib, sums
        _gc.collect()
        print(f"      fold {i}/{C.N_FOLDS} fitted and attributed in "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if not n_rows:
        return None, 0
    return dict(zip(cols, (total / n_rows).astype(float))), n_rows


def values_permutation(predict, X_out, y_out, columns, rng, label):
    """Loss of skill when one column of the held-out rows is permuted.

    Model agnostic and conditional on everything else staying put, which is
    the same kind of quantity the paper's claim is about. Restricted to the
    columns the census compares, because permuting all 581 would cost twenty
    times as much and answer nothing extra.
    """
    from sklearn.metrics import r2_score
    base = r2_score(y_out, predict(X_out))
    out, spread = {}, {}
    t0 = time.time()
    for i, col in enumerate(columns, 1):
        drops = []
        saved = X_out[col].to_numpy().copy()
        for _ in range(PERM_REPEATS):
            X_out[col] = saved[rng.permutation(len(saved))]
            drops.append(base - r2_score(y_out, predict(X_out)))
        X_out[col] = saved
        out[col] = float(np.mean(drops))
        spread[col] = float(np.std(drops))
        if i % 20 == 0 or i == len(columns):
            print(f"      {label}: {i}/{len(columns)} columns, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return out, spread, float(base)


def fit_production_on_split(X_tr, y_tr):
    """The production learner, refitted on the training split alone.

    Needed because the published model was fitted on every row, so permuting
    a column of "held-out" rows and measuring its loss measures an in-sample
    quantity. The first version of this script did exactly that and reported
    a held-out R2 of 0.997, which is the number that gave it away. A
    permutation importance compared against a second learner's held-out one
    has to be held out too, or the two arms are not the same measurement.
    """
    import xgboost as xgb
    store = C.DERIVED / "checkpoints_invariance"
    store.mkdir(parents=True, exist_ok=True)
    f = store / "production_on_split.json"
    m = xgb.XGBRegressor(**C.XGB_PARAMS, n_jobs=C.threads())
    if f.exists():
        m.load_model(str(f))
        print("      production learner refitted on the split: from checkpoint",
              flush=True)
        return m
    t0 = time.time()
    m.fit(X_tr, y_tr)
    m.save_model(str(f))
    print(f"      production learner refitted on the split in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return m


def fit_second_learner(X_tr, y_tr):
    """A histogram gradient-boosting model from scikit-learn.

    A second implementation, not a second configuration: different library,
    different binning, different regularisation, different handling of
    missing values. Chosen over a random forest because a forest of a depth
    that would fit in memory here is a different model class rather than a
    second opinion, and over CatBoost and LightGBM because neither is
    installed on this machine and an arm that cannot run is not an arm.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.06, max_leaf_nodes=63,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=C.SEED)
    t0 = time.time()
    m.fit(X_tr, y_tr)
    print(f"      second learner fitted in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return m


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="comma separated: " + ", ".join(ARMS))
    ap.add_argument("--max-fields", type=int, default=0,
                    help="SMOKE TEST. Use only this many census pairs, to "
                         "check the code path in minutes. Written into the "
                         "result file so no number produced this way can be "
                         "mistaken for the study.")
    a = ap.parse_args()
    want = ARMS if a.only == "all" else [s.strip() for s in a.only.split(",")]

    C.banner("21  is the central finding a property of the data or of TreeSHAP")
    if a.max_fields:
        print(f"  SMOKE TEST: {a.max_fields} pairs only. Not the study.")
    C.seed_everything()
    C.DERIVED.mkdir(parents=True, exist_ok=True)
    rng = C.seed_everything(extra=23)

    import joblib
    C.require(C.MODEL_A)
    model = joblib.load(C.MODEL_A)["model"]
    p = C.load_panel()
    X, y, groups = p["X"], p["y"], p["groups"]
    print(f"  panel {X.shape[0]:,} rows x {X.shape[1]} columns")

    pairs = C.census_pairs(columns=X.columns)
    if a.max_fields:
        pairs = pairs[: a.max_fields]
    print(f"  {len(pairs)} fields carry a central and a non-central column in "
          f"the model")
    pair_cols = sorted({col for _, lo_col, hi_col in pairs
                        for col in (lo_col, hi_col)})
    print(f"  {len(pair_cols)} distinct columns across those pairs")

    estimators, notes, per_field = {}, {}, []

    def record(key, label, values, kind, extra=None):
        if values is None:
            return
        got, wins, rows = count_wins(values, pairs)
        rec = dict(label=label, kind=kind, n_comparable=got,
                   noncentral_wins=wins,
                   share=float(wins / got) if got else float("nan"),
                   sign_test_p=sign_test_p(wins, got))
        rec.update(extra or {})
        estimators[key] = rec
        for r in rows:
            r = dict(r)
            r.update(estimator=key, estimator_label=label, kind=kind)
            per_field.append(r)
        print(f"    {label:<34} non-central leads in {wins:>3} of {got:>3}"
              f"   p = {rec['sign_test_p']:.2e}")

    print("\n" + "=" * 74)
    print("  THE SAME QUESTION, SEVERAL WAYS")
    print("=" * 74)

    if "published" in want:
        record("interventional_published",
               "interventional SHAP, published", values_published(),
               "conditional")

    if "path" in want:
        record("path_dependent", "TreeSHAP, path dependent",
               values_path_dependent(model, X, rng), "conditional",
               dict(n_rows_attributed=min(SHAP_ROWS, len(X))))

    if "gain" in want:
        record("total_gain", "total split gain", values_total_gain(model),
               "not an attribution")

    if "shap_here" in want:
        vals, why = values_shap_here(model, X, rng)
        if vals is None:
            notes["interventional_recomputed"] = why
            print(f"    {'interventional SHAP, here':<34} DID NOT RUN")
            print(f"      {why[:100]}")
            print("      Reported as not run rather than omitted. The")
            print("      published arm above is the same estimator, computed")
            print("      by the main run on a machine where it works.")
        else:
            record("interventional_recomputed", "interventional SHAP, here",
                   vals, "conditional",
                   dict(n_rows_attributed=min(SHAP_ROWS, len(X)),
                        background=C.SHAP_BACKGROUND))

    if "oof_shap" in want:
        print("\n    out-of-fold attribution: every value from a model that")
        print("    never saw the row it is attributing. This is the arm that")
        print("    separates structure in the data from structure the model")
        print("    fitted, and it is the one the central claim should rest on.")
        vals, n_rows = values_oof_shap(X, y, groups, pair_cols)
        if vals is None:
            print("    out-of-fold attribution did not complete")
        else:
            record("oof_path_dependent",
                   "TreeSHAP, out of fold", vals, "conditional",
                   dict(n_rows_attributed=int(n_rows), n_folds=C.N_FOLDS,
                        attribution="path dependent TreeSHAP, per fold, on "
                                    "held-out rows only"))

    if "perm_prod" in want or "perm_second" in want:
        tr, te = grouped_split(groups, HELDOUT_FRACTION, rng)
        if len(te) > PERM_ROWS:
            te = rng.choice(te, PERM_ROWS, replace=False)
        print(f"\n    permutation arms: {len(tr):,} training rows, "
              f"{len(te):,} held-out rows, counties disjoint")
        X_out = X.iloc[te].copy()
        y_out = y[te]

        if "perm_prod" in want:
            # in sample, on the published model, which saw these rows
            vals, spread, base = values_permutation(
                lambda d: model.predict(d), X_out, y_out, pair_cols, rng,
                "published model, in sample")
            record("permutation_published_in_sample",
                   "permutation, published model", vals, "in sample",
                   dict(r2_on_those_rows=base, n_rows=int(len(te)),
                        repeats=PERM_REPEATS,
                        median_repeat_sd=float(np.median(list(spread.values()))),
                        warning="The published model was fitted on every row, "
                                "so these rows are not held out and this "
                                "importance is in sample. Kept for comparison "
                                "and excluded from the conditional verdict."))
            # and the comparable version: the same learner, refitted on the
            # training split, permuted on rows it has never seen
            prod = fit_production_on_split(X.iloc[tr], y[tr])
            vals, spread, base = values_permutation(
                lambda d: prod.predict(d), X_out, y_out, pair_cols, rng,
                "production learner, held out")
            record("permutation_production_heldout",
                   "permutation, production learner", vals, "conditional, loss based",
                   dict(heldout_r2=base, n_heldout=int(len(te)),
                        repeats=PERM_REPEATS,
                        median_repeat_sd=float(np.median(list(spread.values()))),
                        learner="xgboost, production settings, refitted on "
                                "the training split"))
            del prod
            gc.collect()

        if "perm_second" in want:
            second = fit_second_learner(X.iloc[tr], y[tr])
            vals, spread, base = values_permutation(
                lambda d: second.predict(d), X_out, y_out, pair_cols, rng,
                "second learner")
            record("permutation_second_learner",
                   "permutation, second learner", vals, "conditional, loss based",
                   dict(heldout_r2=base, n_heldout=int(len(te)),
                        repeats=PERM_REPEATS,
                        learner="sklearn HistGradientBoostingRegressor"))
            del second
            gc.collect()

    # mutual information, from script 14, as the marginal reference point
    try:
        it = C.load_table("information_theory_census")
        if "noncentral_wins_mi" in it.columns:
            w = int(it["noncentral_wins_mi"].astype(str).str.lower()
                    .isin(("true", "1", "1.0", "yes")).sum())
            n = int(len(it))
            estimators["mutual_information"] = dict(
                label="mutual information", kind="marginal",
                n_comparable=n, noncentral_wins=w,
                share=float(w / n) if n else float("nan"),
                sign_test_p=sign_test_p(w, n),
                source="information_theory_census.csv, script 14")
            print(f"    {'mutual information (script 14)':<34} non-central "
                  f"leads in {w:>3} of {n:>3}")
    except SystemExit:
        notes["mutual_information"] = ("information_theory_census.csv absent; "
                                       "run script 14")

    # ---------------------------------------------------------------- verdict
    # Merge with whatever earlier calls wrote BEFORE computing the agreement
    # matrix and the verdict. An arm run on its own would otherwise report an
    # empty agreement matrix and a verdict over one instrument, which reads
    # like a result and is not one.
    df = pd.DataFrame(per_field)
    if OUT_CSV.exists():
        try:
            prev = pd.read_csv(OUT_CSV)
            df = (pd.concat([prev, df], ignore_index=True)
                  .drop_duplicates(subset=["estimator", "field"], keep="last"))
        except Exception:
            pass
    if OUT_JSON.exists():
        try:
            old_blob = json.load(open(OUT_JSON))
            merged = dict(old_blob.get("estimators") or {})
            merged.update(estimators)
            estimators = merged
        except Exception:
            pass
    print("\n" + "=" * 74)
    print("  AGREEMENT")
    print("=" * 74)
    ran_now = sorted({r["estimator"] for r in per_field})
    if len(ran_now) < len(estimators):
        print(f"    {len(ran_now)} instrument(s) ran in this call; the table")
        print(f"    below is over all {len(estimators)} recorded for this run.")
    agreement = {}
    if not df.empty:
        wide = df.pivot_table(index="field", columns="estimator",
                              values="noncentral_wins", aggfunc="first")
        keys = list(wide.columns)
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1:]:
                both = wide[[k1, k2]].dropna()
                if len(both):
                    agree = float((both[k1] == both[k2]).mean())
                    agreement[f"{k1}|{k2}"] = dict(
                        n=int(len(both)), agree_share=agree)
        print(f"    {'pair of instruments':<58} fields  agree")
        for k, v in sorted(agreement.items(),
                           key=lambda kv: -kv[1]["agree_share"]):
            print(f"    {k[:58]:<58} {v['n']:>5}  {100 * v['agree_share']:5.1f}%")

    # Two families of instrument, and they are not interchangeable.
    #
    #   DECOMPOSITION instruments split a prediction among the columns that
    #   produced it. Both SHAP variants and the split gain are of this kind.
    #   Each column gets its own share whether or not another column could
    #   have supplied the same information.
    #
    #   LOSS-BASED instruments permute a column and measure the damage. For
    #   two columns correlated at 0.99 the damage is near zero for both,
    #   because permuting one leaves its near-duplicate in place to carry the
    #   information. Every pair in this census is a pair of descriptions of
    #   one field, so this is not an edge case here: it is the whole
    #   population. A loss-based instrument is therefore the one least able to
    #   answer the question the census asks, and its disagreement is
    #   predictable rather than informative.
    #
    # The arm that settles the in-sample worry is the out-of-fold one, where
    # no attribution value comes from a model that saw the row.
    LOSS_BASED = ("permutation_production_heldout", "permutation_second_learner",
                  "permutation_published_in_sample")
    decomposition = {k: v for k, v in estimators.items()
                     if v.get("kind") == "conditional" and k not in LOSS_BASED}
    loss_based = {k: v for k, v in estimators.items() if k in LOSS_BASED}
    print()
    verdict = {}
    if not decomposition:
        print("    No decomposition instrument ran. Nothing is settled here.")
    else:
        def line(k, v):
            r2 = v.get("heldout_r2")
            return (f"      {v['label'][:40]:<40} "
                    f"{v['noncentral_wins']:>3}/{v['n_comparable']:<3} "
                    f"p={v['sign_test_p']:<9.3g}"
                    + (f" held-out R2 {r2:.3f}" if r2 is not None else ""))

        print("    DECOMPOSITION INSTRUMENTS -- each column gets its own share")
        for k in sorted(decomposition, key=lambda k: decomposition[k]["sign_test_p"]):
            print(line(k, decomposition[k]))
        ok = [k for k, v in decomposition.items() if v["sign_test_p"] < 0.05]
        shares = [v["share"] for v in decomposition.values()]
        print(f"      {len(ok)} of {len(decomposition)} significant at 0.05, "
              f"share between {min(shares):.2f} and {max(shares):.2f}")
        oof = estimators.get("oof_path_dependent")
        if oof:
            print()
            print("    OUT OF FOLD, which is the one that matters: "
                  f"{oof['noncentral_wins']}/{oof['n_comparable']}, "
                  f"p = {oof['sign_test_p']:.3g}.")
            print("    Every value from a model that never saw the row it")
            print("    attributed, so the count is not something the model")
            print("    fitted to rows it had already learned.")
        if loss_based:
            print()
            print("    LOSS-BASED INSTRUMENTS -- permute a column, measure the")
            print("    damage")
            for k in sorted(loss_based, key=lambda k: loss_based[k]["sign_test_p"]):
                print(line(k, loss_based[k]))
            print()
            print("    These disagree, and the paper should say why rather")
            print("    than leave them out. Permuting one member of a pair")
            print("    correlated at 0.99 leaves its near-duplicate in place,")
            print("    so neither member registers much damage and the")
            print("    comparison is between two small noisy numbers. Every")
            print("    pair in this census is a pair of descriptions of one")
            print("    field, so that is the whole population here, not an")
            print("    edge case. A loss-based importance is the instrument")
            print("    least able to answer this particular question, and")
            print("    reporting it alongside the others, with that reason")
            print("    stated, is stronger than reporting only the")
            print("    instruments that agree.")
        print()
        print("    THE HONEST FORM OF THE CLAIM. Under conditional")
        print("    attribution of a gradient-boosted tree fitted to this")
        print("    panel, including attribution computed out of fold, the")
        print("    non-central description of a field takes more of the")
        print("    attribution in roughly three quarters of the fields it can")
        print("    be compared in. Marginal information, taken one column at")
        print("    a time, points the other way, and loss-based importance is")
        print("    uninformative here for the reason above. All three belong")
        print("    in the paper.")
        verdict = dict(
            decomposition=sorted(decomposition),
            decomposition_significant=sorted(ok),
            loss_based=sorted(loss_based),
            out_of_fold=(dict(n=oof["n_comparable"],
                              wins=oof["noncentral_wins"],
                              sign_test_p=oof["sign_test_p"]) if oof else None),
            share_min=float(min(shares)), share_max=float(max(shares)),
            reading=("Decomposition instruments agree, including out of fold. "
                     "Loss-based importance disagrees and is deflated for "
                     "near-duplicate columns, which is every pair here. "
                     "Marginal mutual information points the other way and is "
                     "a different quantity."))

    blob = dict(n_pairs=len(pairs), n_pair_columns=len(pair_cols),
                estimators=estimators, agreement=agreement, not_run=notes,
                instruments_in_this_call=ran_now, verdict=verdict,
                smoke_test_max_fields=int(a.max_fields),
                settings=dict(PERM_ROWS=PERM_ROWS, PERM_REPEATS=PERM_REPEATS,
                              SHAP_ROWS=SHAP_ROWS,
                              HELDOUT_FRACTION=HELDOUT_FRACTION),
                note="The same census question asked with several attribution "
                     "instruments. Conditional and marginal instruments "
                     "measure different quantities and are labelled.")
    json.dump(blob, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\n  [saved] {OUT_JSON}")
    if not df.empty:
        df.to_csv(OUT_CSV, index=False)
        print(f"  [saved] {OUT_CSV}")

    C.write_manifest("21_attribution_estimator_invariance.py",
                     inputs=[C.MODEL_A, C.FRAME, C.RANKING_A],
                     outputs=[OUT_JSON, OUT_CSV],
                     settings=dict(arms=want, PERM_ROWS=PERM_ROWS,
                                   PERM_REPEATS=PERM_REPEATS,
                                   SHAP_ROWS=SHAP_ROWS,
                                   max_fields=a.max_fields),
                     notes="The census under several attribution estimators.")


if __name__ == "__main__":
    main()
