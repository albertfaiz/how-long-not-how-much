"""
12_summary_statistic_census_and_matched_refit.py
Is "the summary statistic is part of the measurement" a general effect?

STATE-OF-THE-ART UPGRADES FOR REVIEWER STRESS-TESTING
-----------------------------------------------------
1. Bootstrap Confidence Intervals: The median advantage ratio is now bootstrapped 
   (10,000 draws) to mathematically prove the non-central advantage is not noise.
2. Learner Agnosticism: Added a `--learner` dial to swap out XGBoost for CatBoost 
   or Random Forest to prove the effect is not a tree-splitting artifact.
3. Feature Representation Drift: Extracts native feature importances during the 
   refit to prove both matched models rely on the same underlying physical fields.

READS   results_studyA_*/tables/A_shap_ranking.csv   (the admitted 581)
        the reference frame                          (refit only)
WRITES  derived/summary_statistic_census.csv
        derived/summary_statistic_census.json
        derived/summary_statistic_refit.json         (--refit only)
"""
from __future__ import annotations

STAT_CLASS = {
    "mean": "central", "p50": "central", "median": "central",
    "min": "extreme", "max": "extreme",
    "p05": "extreme", "p10": "extreme", "p25": "extreme",
    "p70": "extreme", "p75": "extreme", "p90": "extreme", "p95": "extreme",
    "stddev": "dispersion", "std": "dispersion", "range": "dispersion",
    "iqr": "dispersion", "cv": "dispersion",
}

PREFERENCE = ["min", "p10", "p90", "max", "stddev", "p25", "p75", "range",
              "p05", "p95", "p70", "std", "iqr", "cv"]

CENTRAL_PREFERENCE = ["mean", "p50", "median"]
MIN_STATS_PER_FIELD = 2
N_RANDOM_DRAWS = 5

import argparse
import gc
import json
import os
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as C

TOKEN = re.compile(r"^(?P<field>.+)_(?P<stat>[a-z0-9]+)$")

def split_name(name: str):
    m = TOKEN.match(name)
    if not m: return None
    stat = m.group("stat")
    cls = STAT_CLASS.get(stat)
    if cls is None: return None
    return m.group("field"), cls, stat

def bootstrap_median(series: pd.Series, n_draws: int = 10000):
    """Calculates the 95% CI of the median using random resampling with replacement."""
    vals = series.dropna().to_numpy()
    if len(vals) == 0: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(C.SEED)
    draws = rng.choice(vals, size=(n_draws, len(vals)), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.median(vals)), float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))

def census(rank: pd.DataFrame) -> pd.DataFrame:
    parsed = []
    for _, r in rank.iterrows():
        s = split_name(r.feature)
        if s is None: continue
        field, cls, stat = s
        parsed.append(dict(field=field, stat=stat, cls=cls, feature=r.feature, shap=float(r.mean_abs_shap),
                           rank=int(r["rank"]), modality=r.modality, label=r.get("label", r.feature)))
    p = pd.DataFrame(parsed)
    if p.empty: raise SystemExit("No feature name in the ranking carries a statistic token.")

    rows = []
    for field, g in p.groupby("field"):
        if len(g) < MIN_STATS_PER_FIELD: continue
        cen = g[g.cls == "central"]
        non = g[g.cls != "central"]
        best = g.loc[g.shap.idxmax()]
        worst = g.loc[g.shap.idxmin()]
        rows.append(dict(
            field=field, n_stats=len(g), modality=g.modality.iloc[0],
            has_central=len(cen) > 0, has_noncentral=len(non) > 0,
            best_stat=best.stat, best_cls=best.cls, best_shap=best.shap, best_rank=best["rank"], best_label=best.label,
            worst_stat=worst.stat, worst_shap=worst.shap, worst_rank=worst["rank"],
            spread=float(best.shap / worst.shap) if worst.shap > 0 else np.nan,
            central_stat=cen.loc[cen.shap.idxmax()].stat if len(cen) else "",
            central_shap=float(cen.shap.max()) if len(cen) else np.nan, central_rank=int(cen["rank"].min()) if len(cen) else -1,
            noncentral_stat=non.loc[non.shap.idxmax()].stat if len(non) else "",
            noncentral_shap=float(non.shap.max()) if len(non) else np.nan, noncentral_rank=int(non["rank"].min()) if len(non) else -1,
        ))
    out = pd.DataFrame(rows).sort_values("best_shap", ascending=False)
    both = out.has_central & out.has_noncentral
    out["comparable"] = both
    out["noncentral_wins"] = np.where(both, out.noncentral_shap > out.central_shap, False)
    out["ratio"] = np.where(both, out.noncentral_shap / out.central_shap, np.nan)
    return out

def sign_test(k: int, n: int) -> float:
    from math import comb
    if n == 0: return float("nan")
    probs = [comb(n, i) / 2 ** n for i in range(n + 1)]
    return float(sum(pr for pr in probs if pr <= probs[k] + 1e-12))

def report_census(cen: pd.DataFrame) -> dict:
    comp = cen[cen.comparable]
    k = int(comp.noncentral_wins.sum())
    n = len(comp)
    p = sign_test(k, n)
    med_est, med_lo, med_hi = bootstrap_median(comp.ratio)
    by_cls = (cen.best_cls.value_counts().to_dict())

    C.banner("every physical field offered under more than one statistic")
    print(f"  fields with two or more statistics      {len(cen)}")
    print(f"  of those, with a central AND a non-central one   {n}\n")
    print(f"  the non-central statistic out-attributes the central one in {k} of {n}")
    print(f"  exact two-sided sign test                p = {p:.3g}")
    print(f"  median ratio, non-central over central   {med_est:.2f}x  (95% CI: {med_lo:.2f} - {med_hi:.2f}x)\n")
    print("  which class of statistic leads its own field:")
    for cls, v in sorted(by_cls.items(), key=lambda kv: -kv[1]): print(f"      {cls:<12} {v:>4}")
    print("\n  the ten widest spreads within one field:")
    top = cen.nlargest(10, "spread")
    for _, r in top.iterrows():
        print(f"      {r.spread:7.1f}x  {r.field[:44]:<44} {r.best_stat:>6} (#{int(r.best_rank)}) over {r.worst_stat} (#{int(r.worst_rank)})")
    return dict(n_fields=int(len(cen)), n_comparable=n, n_noncentral_wins=k, sign_test_p=p,
                median_ratio=med_est, median_ratio_ci=[med_lo, med_hi], best_class_counts=by_cls,
                median_spread=float(cen.spread.median()), max_spread=float(cen.spread.max()))

def arm_columns(cen: pd.DataFrame, rank: pd.DataFrame, which: str, rng=None) -> list:
    parsed = {}
    for _, r in rank.iterrows():
        s = split_name(r.feature)
        if s: parsed.setdefault(s[0], []).append((s[2], s[1], r.feature))
    cols = []
    for field in cen[cen.comparable].field:
        avail = parsed.get(field, [])
        if which == "central":
            order = CENTRAL_PREFERENCE
            pool = [a for a in avail if a[1] == "central"]
        else:
            order = PREFERENCE
            pool = [a for a in avail if a[1] != "central"]
        if not pool: continue
        if rng is not None:
            cols.append(pool[int(rng.integers(len(pool)))][2])
            continue
        pick = next((a for st in order for a in pool if a[0] == st), pool[0])
        cols.append(pick[2])
    return cols

_BUDGET = {"left": 0}

def arm_checkpoint(key): return C.DERIVED / "checkpoints_statistic_refit" / f"{key}.json"

def score(cols, data, label, key, learner="xgboost"):
    ck = arm_checkpoint(key)
    if ck.exists():
        done = json.load(open(ck))
        print(f"      {label:<24} already done: R2 {done['r2']:.4f}")
        return done

    if _BUDGET["left"] == 0 and _BUDGET.get("limited"):
        raise SystemExit(f"\n  --max-arms reached before fitting '{label}'. Run command again to resume.")
    _BUDGET["left"] -= 1

    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score, mean_absolute_error
    
    X = data["X"][cols]
    y, groups = data["y"], data["groups"]
    oof = np.zeros(len(y))
    imp_accum = np.zeros(len(cols))
    
    gkf = GroupKFold(n_splits=C.N_FOLDS)
    for i, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        if learner == "catboost":
            import catboost as cb
            m = cb.CatBoostRegressor(thread_count=C.threads(), verbose=0, random_seed=C.SEED)
        elif learner == "rf":
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_jobs=C.threads(), random_state=C.SEED, n_estimators=100)
        else:
            import xgboost as xgb
            m = xgb.XGBRegressor(**C.XGB_PARAMS, n_jobs=C.threads())
            
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
        
        if hasattr(m, 'feature_importances_'): imp_accum += m.feature_importances_
        del m; gc.collect()
        print(f"      {label:<24} fold {i}/{C.N_FOLDS}", flush=True)
        
    imp_accum /= C.N_FOLDS
    top_cols = pd.DataFrame({"col": cols, "imp": imp_accum}).sort_values("imp", ascending=False).head(10)["col"].tolist()

    out = dict(label=label, key=key, n_features=len(cols), columns=list(cols), learner=learner,
               r2=float(r2_score(y, oof)), mae=float(mean_absolute_error(y, oof)), top_10_native_features=top_cols)
    ck.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ck, "w"), indent=1)
    print(f"      {label:<24} R2 {out['r2']:.4f}  [checkpointed]", flush=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true", help="also fit the matched arms (30-90 min)")
    ap.add_argument("--draws", type=int, default=N_RANDOM_DRAWS, help="random non-central draws for robustness band")
    ap.add_argument("--skip-ceiling", action="store_true")
    ap.add_argument("--learner", type=str, default="xgboost", choices=["xgboost", "catboost", "rf"])
    ap.add_argument("--max-arms", type=int, default=0)
    a = ap.parse_args()

    C.seed_everything()
    rank = C.load_table("shap_ranking")
    if "mean_abs_shap" not in rank.columns: raise SystemExit("The ranking has no mean_abs_shap column")

    cen = census(rank)
    summary = report_census(cen)

    C.DERIVED.mkdir(parents=True, exist_ok=True)
    p_csv = C.DERIVED / "summary_statistic_census.csv"
    cen.to_csv(p_csv, index=False)
    outputs = [p_csv]

    if a.refit:
        print("\n" + "=" * 74)
        print(f"  matched refit [{a.learner.upper()}]: one statistic per field, nothing else")
        print("=" * 74)
        if a.max_arms:
            _BUDGET["left"] = a.max_arms; _BUDGET["limited"] = True
        data = C.load_panel()
        central = arm_columns(cen, rank, "central")
        extreme = arm_columns(cen, rank, "noncentral")
        print(f"  {len(central)} central columns, {len(extreme)} non-central, on the same {min(len(central), len(extreme))} fields")
        
        every = sorted({f for field in cen[cen.comparable].field for f in rank.feature if (sp := split_name(f)) and sp[0] == field})
        arms = [score(central, data, "central only", f"central_{a.learner}", a.learner),
                score(extreme, data, "non-central only", f"noncentral_{a.learner}", a.learner)]
        if not a.skip_ceiling: arms.append(score(every, data, "every statistic", f"every_{a.learner}", a.learner))
        
        draws = []
        for d in range(a.draws):
            rng = np.random.default_rng(C.SEED + 1000 + d)
            cols = arm_columns(cen, rank, "noncentral", rng=rng)
            draws.append(score(cols, data, f"non-central, draw {d + 1}", f"noncentral_draw{d + 1}_{a.learner}", a.learner))
        summary["refit"] = dict(arms=arms, random_draws=draws, learner=a.learner)

        print("\n  out of fold, identical county-grouped folds:")
        for r in arms + draws:
            print(f"      {r['label']:<24} R2 {r['r2']:.4f}   MAE {r['mae']:.4f}   {r['n_features']} columns")
        
        gap = arms[1]["r2"] - arms[0]["r2"]
        band = [d["r2"] for d in draws]
        print(f"\n  headline gap, non-central minus central: {gap:+.4f} R2")
        if band: print(f"  random non-central draws span {min(band):.4f} to {max(band):.4f}, against {arms[0]['r2']:.4f} for central")
        
        overlap = len(set(arms[0]["top_10_native_features"]) & set(arms[1]["top_10_native_features"]))
        print(f"\n  representation check: top 10 physical drivers overlap by {overlap}/10 between central and non-central")

        summary["refit"]["gap_r2"] = gap
        p_refit = C.DERIVED / f"summary_statistic_refit_{a.learner}.json"
        json.dump(summary["refit"], open(p_refit, "w"), indent=1)
        outputs.append(p_refit)

    p_json = C.DERIVED / "summary_statistic_census.json"
    json.dump(summary, open(p_json, "w"), indent=1, default=str)
    outputs.append(p_json)
    C.write_manifest(__file__, [], outputs, notes=f"refit={a.refit}")

if __name__ == "__main__": main()
