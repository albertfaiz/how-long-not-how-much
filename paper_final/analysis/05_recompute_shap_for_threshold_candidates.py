from __future__ import annotations
import argparse, json, time, numpy as np, pandas as pd
import _config as C

CANDIDATES = [
    "prism_tdmean_min", "eng2_night_cooling_efficiency", "eng_night_cooling_eff",
    "lst_night_1km_mean", "eng2_wetbulb_night", "eng2_wetbulb_hot",
    "eng2_hcho_no2_ratio", "pm25_mean", "prism_tmin_min",
    "cams_fot_formaldehyde_above75th_percentile", "usda_cropland_usda_corn_pct",
]

def _binned(x, y, nb=40):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < nb * 5: return np.full(nb, np.nan)
    qs = np.quantile(x, np.linspace(0, 1, nb + 1)); qs[-1] += 1e-9
    out = np.full(nb, np.nan)
    for i in range(nb):
        s = (x >= qs[i]) & (x < qs[i + 1])
        if s.sum() > 2: out[i] = y[s].mean()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=C.SHAP_SAMPLE)
    a = ap.parse_args()

    import joblib, shap
    C.banner("05  recompute the attribution surface for the gate candidates")
    C.require(C.FRAME, C.MODEL_A)
    d = joblib.load(C.MODEL_A)
    model, feats = d["model"], d["features"]

    missing = [c for c in CANDIDATES if c not in feats]
    keep = [c for c in CANDIDATES if c in feats]
    
    p = C.load_panel()
    X, df = p["X"], p["df"]
    rng = C.seed_everything()
    samp = rng.choice(len(X), size=min(a.sample, len(X)), replace=False)
    bg = rng.choice(len(X), size=C.SHAP_BACKGROUND, replace=False)

    t0 = time.time()
    ex = shap.TreeExplainer(model, data=X.iloc[bg], feature_perturbation="interventional")
    
    # FIX: Removed the silent=True argument
    sv = ex.shap_values(X.iloc[samp], check_additivity=False)
    print(f"  interventional TreeSHAP: {time.time()-t0:.0f}s")

    idx = {f: i for i, f in enumerate(feats)}
    out = df.iloc[samp][[C.GROUP, C.YEAR, C.TARGET]].reset_index(drop=True)
    for f in keep:
        out[f] = X.iloc[samp][f].to_numpy()
        out["shap_" + f] = sv[:, idx[f]]

    checks = {}
    if C.DEPENDENCE_A.exists():
        st = pd.read_csv(C.DEPENDENCE_A)
        for f in keep:
            if f not in st.columns or ("shap_" + f) not in st.columns: continue
            a1 = _binned(out[f].to_numpy(), out["shap_" + f].to_numpy())
            a2 = _binned(st[f].to_numpy(), st["shap_" + f].to_numpy())
            m = np.isfinite(a1) & np.isfinite(a2)
            if m.sum() > 5:
                r = float(np.corrcoef(a1[m], a2[m])[0, 1])
                rng_ = float(np.nanmax(a2[m]) - np.nanmin(a2[m]))
                rms = float(np.sqrt(np.nanmean((a1[m] - a2[m]) ** 2)))
                checks[f] = dict(r=r, rms_over_range=rms / rng_ if rng_ > 0 else float("nan"), curve_range=rng_)

    C.DERIVED.mkdir(parents=True, exist_ok=True)
    dest = C.DERIVED / "shap_dependence_candidates.csv"
    out.to_csv(dest, index=False)
    chk = C.DERIVED / "shap_dependence_check.json"
    json.dump(dict(n_rows=int(len(out)), sample=int(a.sample), background=C.SHAP_BACKGROUND, seed=C.SEED, candidates=keep, missing=missing, binned_correlation_vs_stored=checks), open(chk, "w"), indent=1)
    C.write_manifest(__file__, [C.FRAME, C.MODEL_A], [dest, chk], settings=dict(sample=a.sample, candidates=keep))

if __name__ == "__main__": main()
