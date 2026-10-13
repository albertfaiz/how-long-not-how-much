from __future__ import annotations
import argparse, json, time, numpy as np, pandas as pd
import _config as C

ARMS = ("FULL", "DROP_P", "DROP_M")   
TOP_N = 10                            

def run_arm(name, drop, p, a, ckdir):
    import shap, xgboost as xgb
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold

    cols = [f for f in p["features"] if f not in drop]
    X, y, g = p["X"][cols], p["y"], p["groups"]
    ck = ckdir / f"{name}_f{a.folds}.npz"
    oof = np.full(len(X), np.nan)
    done: set[int] = set()
    last = None

    if ck.exists() and not a.fresh:
        z = np.load(ck, allow_pickle=True)
        oof, done = z["oof"], set(z["done"].tolist())
        print(f"  [{name}] resuming, {len(done)}/{a.folds} folds already fitted")

    t0 = time.time()
    splits = list(GroupKFold(n_splits=a.folds).split(X, y, g))
    for k, (tr, te) in enumerate(splits, 1):
        if k in done and k != a.folds: continue
        m = xgb.XGBRegressor(n_jobs=C.threads(), **C.XGB_PARAMS)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
        last = m
        done.add(k)
        np.savez(ck, oof=oof, done=np.array(sorted(done)))
        print(f"  [{name}] fold {k}/{a.folds}  {time.time()-t0:7.0f}s  (checkpointed)", flush=True)

    if last is None:
        tr, _ = splits[-1]
        last = xgb.XGBRegressor(n_jobs=C.threads(), **C.XGB_PARAMS).fit(X.iloc[tr], y[tr])

    r2, mae = r2_score(y, oof), mean_absolute_error(y, oof)
    rng = C.seed_everything()
    samp = rng.choice(len(X), size=min(a.shap, len(X)), replace=False)
    bg = rng.choice(len(X), size=C.SHAP_BACKGROUND, replace=False)
    
    ts = time.time()
    ex = shap.TreeExplainer(last, data=X.iloc[bg], feature_perturbation="interventional")
    
    # FIX: Removed the silent=True argument causing the TypeError
    sv = np.abs(ex.shap_values(X.iloc[samp], check_additivity=False)).mean(0)
    print(f"  [{name}] interventional TreeSHAP on {len(samp):,} rows: {time.time()-ts:.0f}s", flush=True)

    rank = pd.Series(sv, index=cols).sort_values(ascending=False)
    order = {f: i + 1 for i, f in enumerate(rank.index)}
    return dict(
        r2=float(r2), mae=float(mae), n_features=len(cols), seconds=round(time.time() - t0),
        pct_rank=order.get(C.NCE_TAIL), pct_shap=None if C.NCE_TAIL not in order else float(rank[C.NCE_TAIL]),
        mean_rank=order.get(C.NCE_MEAN), mean_shap=None if C.NCE_MEAN not in order else float(rank[C.NCE_MEAN]),
        top10=[(f, round(float(v), 4)) for f, v in rank.head(TOP_N).items()])

def report(res):
    f, dp, dm = res.get("FULL"), res.get("DROP_P"), res.get("DROP_M")
    if not (f and dp): return
    rise = dp["mean_shap"] / f["mean_shap"]
    rec = 100 * (dp["mean_shap"] - f["mean_shap"]) / f["pct_shap"]
    print("\n" + "=" * 74)
    print("  WHAT THE MANUSCRIPT PRINTS")
    print("=" * 74)
    print(f"  tail construction, both present  rank {f['pct_rank']:>4}  mean|SHAP| {f['pct_shap']:.4f}")
    print(f"  central, both present            rank {f['mean_rank']:>4}  mean|SHAP| {f['mean_shap']:.5f}")
    print(f"  central, twin removed            rank {dp['mean_rank']:>4}  mean|SHAP| {dp['mean_shap']:.5f}")
    print(f"\n  rise {rise:.1f}-fold, recovering {rec:.1f} % of what the tail construction carried")
    print(f"  R2  both {f['r2']:.4f} | tail removed {dp['r2']:.4f} ({dp['r2']-f['r2']:+.4f})", end="")
    if dm:
        print(f" | central removed {dm['r2']:.4f} ({dm['r2']-f['r2']:+.4f})")
        print(f"  tail with the field to itself: rank {dm['pct_rank']}, mean|SHAP| {dm['pct_shap']:.4f}")
        if dm["r2"] >= f["r2"]: print("  Removing the weaker description did not cost accuracy.")
    print("=" * 74)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=C.N_FOLDS)
    ap.add_argument("--shap", type=int, default=C.SHAP_SAMPLE)
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoints and refit everything")
    a = ap.parse_args()

    C.banner("04  refit with one night-cooling construction removed")
    print(f"  arms {ARMS}   shap sample {a.shap:,}\n")
    C.require(C.FRAME, C.MODEL_A)
    p = C.load_panel()
    
    ckdir = C.DERIVED / "checkpoints_refit"
    ckdir.mkdir(parents=True, exist_ok=True)
    out = C.DERIVED / "nce_refit.json"
    
    blob = json.load(open(out)) if (out.exists() and not a.fresh) else {}
    res = blob.get("conditions", {})
    if res: print(f"resuming; already complete: {', '.join(sorted(res))}\n")

    drop_for = {"FULL": [], "DROP_P": [C.NCE_TAIL], "DROP_M": [C.NCE_MEAN]}
    protocol = dict(folds=a.folds, nshap=a.shap, nbg=C.SHAP_BACKGROUND, n_rows=int(len(p["X"])), n_features=len(p["features"]), window=list(p["window"]), learner="xgboost", threads=C.threads(), **C.XGB_PARAMS)

    for name in ARMS:
        if name in res: continue
        what = "both present" if not drop_for[name] else f"without {drop_for[name][0]}"
        print(f"--- {name} ({what}) ---")
        res[name] = run_arm(name, drop_for[name], p, a, ckdir)
        json.dump(dict(protocol=protocol, conditions=res), open(out, "w"), indent=1)
        print(f"  [saved] {out.name}\n")

    report(res)
    json.dump(dict(protocol=protocol, conditions=res), open(out, "w"), indent=1)
    C.write_manifest(__file__, [C.FRAME, C.MODEL_A], [out], settings=dict(folds=a.folds, shap_sample=a.shap, arms=list(ARMS)))

if __name__ == "__main__": main()
