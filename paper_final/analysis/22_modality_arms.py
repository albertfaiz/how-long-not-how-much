#!/usr/bin/env python3
"""
22_modality_arms.py -- how much of the skill is satellite-borne, and does the
central finding hold inside the satellite-only panel?

READS   ../derived*/panel_manifest.json  (script 16, which froze the columns)
        the modelling panel, through _config.load_panel
        the census of script 12
WRITES  ../derived*/modality_arms.json
        ../derived*/modality_arms.csv

THE OBJECTION
-------------
    "This is a socioeconomic model with satellite decoration. You have not
     separated the two, and the reader cannot tell which does the work."

The first thing this script does is answer that objection in a way that was
not anticipated when it was written. Script 16's export found that the model's
581 columns contain NO socioeconomic, census or health-outcome column at all:
the arm built as a control turned out to be the model. Whatever else is true
of this paper, the predictors are environmental measurements and nothing else,
and the count is in the manifest rather than in anyone's recollection.

That makes the interesting split a different one, and it is the one this
script measures:

    all admitted       every column the production model saw
    satellite only     columns from an instrument in orbit
    non satellite      reanalysis, soil survey, agricultural census,
                       livestock inventory, and the engineered combinations
                       built from them

The three scores say how much of the skill needs an instrument in orbit and
how much a reanalysis product already carries. Both answers are publishable
and they are different papers, so the number has to be measured rather than
assumed.

THE SECOND QUESTION
-------------------
Within the satellite-only panel, refitted and re-attributed from scratch, does
the non-central summary of a field still lead? If the central finding is a
property of the data rather than of one particular column set, it should
survive being asked inside a third of the columns.

    python3 22_modality_arms.py
    python3 22_modality_arms.py --only arms
    python3 22_modality_arms.py --folds 3        # smoke test, says so
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

MANIFEST = C.DERIVED / "panel_manifest.json"
OUT_JSON = C.DERIVED / "modality_arms.json"
OUT_CSV = C.DERIVED / "modality_arms.csv"

SHAP_ROWS = 6000          # rows attributed inside each arm
_SMOKE = {"folds": 0}


def folds():
    return _SMOKE["folds"] or C.N_FOLDS


def tag(name):
    return f"modality_{name}" + (f"_f{_SMOKE['folds']}" if _SMOKE["folds"]
                                 else "")


def sign_test_p(wins, n):
    from math import comb
    if not n:
        return float("nan")
    return float(sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n)


def path_dependent(model, X, rng, n_rows=SHAP_ROWS):
    """Mean absolute path-dependent TreeSHAP, computed by xgboost itself."""
    import xgboost as xgb
    idx = rng.choice(len(X), min(n_rows, len(X)), replace=False)
    sub = X.iloc[idx]
    dm = xgb.DMatrix(sub, feature_names=list(sub.columns))
    contrib = np.asarray(model.get_booster().predict(dm, pred_contribs=True))
    return dict(zip(sub.columns,
                    np.abs(contrib[:, :-1]).mean(axis=0).astype(float)))


def fit_full(X, y, name):
    """One model on all rows, for attribution inside an arm. Not a score."""
    import xgboost as xgb
    store = C.DERIVED / "checkpoints_modality"
    store.mkdir(parents=True, exist_ok=True)
    path = store / f"{name}_full.json"
    m = xgb.XGBRegressor(**C.XGB_PARAMS, n_jobs=C.threads())
    if path.exists():
        m.load_model(str(path))
        print(f"      {name}: full-data model from checkpoint", flush=True)
        return m
    t0 = time.time()
    m.fit(X, y)
    m.save_model(str(path))
    print(f"      {name}: full-data model fitted in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return m


# ---------------------------------------------------------------------------
def read_manifest():
    if not MANIFEST.exists():
        raise SystemExit(
            f"{MANIFEST.name} is not there.\n"
            f"    It is written by 16_export_frozen_panels_and_gate_audit.py,\n"
            f"    which is where the column lists for these arms come from.\n"
            f"    Run:  python3 16_export_frozen_panels_and_gate_audit.py "
            f"--only export")
    blob = json.load(open(MANIFEST))
    need = ("columns_all_admitted", "columns_satellite_only",
            "n_modelled", "n_satellite_only")
    gone = [k for k in need if k not in blob]
    if gone:
        raise SystemExit(f"{MANIFEST.name} is missing {', '.join(gone)}. "
                         f"Rerun script 16's export arm.")
    return blob


def arms_from_manifest(man, present):
    allc = [c for c in man["columns_all_admitted"] if c in present]
    sat = [c for c in man["columns_satellite_only"] if c in present]
    non = [c for c in allc if c not in set(sat)]
    return {"all_admitted": allc, "satellite_only": sat,
            "non_satellite": non}


# ---------------------------------------------------------------------------
def run_arms(p, cols):
    print("\n" + "=" * 74)
    print("  1. THREE COLUMN SETS, THE SAME PROTOCOL")
    print("=" * 74)
    print("  County-grouped folds, the production learner, the production")
    print("  settings. The only thing that changes between arms is which")
    print("  columns are available, so the difference between two scores is")
    print("  the value of those columns and nothing else.")
    if _SMOKE["folds"]:
        print(f"\n  SMOKE TEST at {_SMOKE['folds']} folds instead of "
              f"{C.N_FOLDS}. Not comparable with the paper.")
    print()
    X, y, groups = p["X"], p["y"], p["groups"]
    rows = {}
    for name in ("all_admitted", "satellite_only", "non_satellite"):
        use = cols[name]
        if not use:
            print(f"    {name}: no columns, skipped")
            continue
        s = C.cv_score(X[use], y, groups, tag(name), n_folds=folds())
        s["arm"] = name
        rows[name] = s
        print(f"    {name:<18} {len(use):>4} columns   R2 {s['r2']:+.4f}   "
              f"MAE {s['mae']:.4f}")
    if "all_admitted" in rows and "satellite_only" in rows:
        base = rows["all_admitted"]
        print()
        for name in ("satellite_only", "non_satellite"):
            if name in rows:
                d = rows[name]["r2"] - base["r2"]
                share = (rows[name]["r2"] / base["r2"]
                         if base["r2"] else float("nan"))
                print(f"    {name:<18} keeps {100 * share:5.1f} per cent of "
                      f"the full model's R2   ({d:+.4f})")
        print()
        print("    Two column sets that overlap in nothing and each keep most")
        print("    of the skill is the signature of a well covered signal")
        print("    measured twice, not of one indispensable instrument. The")
        print("    paper should say which of those it is claiming.")
    return rows


def census_within(p, cols, rng):
    print("\n" + "=" * 74)
    print("  2. THE CENSUS QUESTION, ASKED INSIDE THE SATELLITE-ONLY PANEL")
    print("=" * 74)
    print("  A model refitted on the satellite columns alone, attributed from")
    print("  scratch, and asked the paper's own question: of the two")
    print("  descriptions of a field, which takes the attribution?")
    print()
    X, y = p["X"], p["y"]
    sat = cols["satellite_only"]
    pairs_all = C.census_pairs(columns=X.columns)
    pairs = [(f, c, n) for f, c, n in pairs_all
             if c in set(sat) and n in set(sat)]
    print(f"    {len(pairs)} of the census's {len(pairs_all)} pairs have both "
          f"members inside the satellite panel")
    if not pairs:
        print("    Nothing to compare. The satellite panel does not contain")
        print("    both descriptions of any field, which is itself a fact")
        print("    about the instrument coverage and is recorded as one.")
        return dict(n_pairs=0,
                    note="no census pair lies wholly inside the satellite "
                         "panel"), []
    m = fit_full(X[sat], y, "satellite_only")
    vals = path_dependent(m, X[sat], rng)
    del m
    gc.collect()
    rows, wins = [], 0
    for field, c_col, n_col in pairs:
        cv, nv = vals.get(c_col), vals.get(n_col)
        if cv is None or nv is None:
            continue
        win = bool(nv > cv)
        wins += int(win)
        rows.append(dict(arm="census_within_satellite", field=field,
                         central_col=c_col, noncentral_col=n_col,
                         central_value=float(cv), noncentral_value=float(nv),
                         noncentral_wins=win))
    n = len(rows)
    pv = sign_test_p(wins, n)
    print(f"    non-central leads in {wins} of {n}   sign test p = {pv:.4g}")
    # the same pairs, as the published full-model attribution ranked them
    pub = C.load_table("A_shap_ranking")
    pubv = dict(zip(pub["feature"].astype(str),
                    pub["mean_abs_shap"].astype(float)))
    agree = sum(1 for r in rows
                if (pubv.get(r["noncentral_col"], np.nan) >
                    pubv.get(r["central_col"], np.nan)) == r["noncentral_wins"])
    print(f"    the full model agreed on the direction for {agree} of {n} of "
          f"the same pairs")
    print()
    print("    This arm changes two things at once, the column set and the")
    print("    attribution estimator, so a disagreement here does not say")
    print("    which of the two caused it. Script 21 separates the estimator;")
    print("    this arm separates the column set.")
    return dict(n_pairs=n, noncentral_wins=wins, sign_test_p=pv,
                agreement_with_full_model=int(agree),
                attribution="path dependent TreeSHAP on a satellite-only "
                            "refit",
                note="Both the column set and the attribution estimator "
                     "differ from the published ranking."), rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="comma separated: arms, census")
    ap.add_argument("--folds", type=int, default=0,
                    help="SMOKE TEST ONLY. Fewer folds than the study uses. "
                         "Results are written under a separate tag and the "
                         "file says so.")
    a = ap.parse_args()
    want = (["arms", "census"] if a.only == "all"
            else [s.strip() for s in a.only.split(",")])
    if a.folds:
        _SMOKE["folds"] = a.folds

    C.banner("22  how much of the skill needs an instrument in orbit")
    C.seed_everything()
    rng = C.seed_everything(extra=24)
    C.DERIVED.mkdir(parents=True, exist_ok=True)

    man = read_manifest()
    p = C.load_panel()
    cols = arms_from_manifest(man, set(p["X"].columns))
    print(f"  panel {p['X'].shape[0]:,} rows x {p['X'].shape[1]} columns")
    print(f"  candidates {man['n_candidates']}, gate admitted "
          f"{man['n_gate_admitted']}, modelled {man['n_modelled']}")
    for k, v in cols.items():
        print(f"    {k:<18} {len(v):>4} columns")
    nonphys = man["n_modelled"] - man.get("n_environment_only",
                                          man["n_modelled"])
    print(f"\n  socioeconomic, census and health-outcome columns in the "
          f"model: {nonphys}")
    if nonphys == 0:
        print("  The model has none. The predictors are environmental")
        print("  measurements only, which answers the objection this script")
        print("  was written for before any arm is fitted, and is the first")
        print("  thing the paper should say about the panel.")

    blob, rows = {}, []
    if "arms" in want:
        blob["arms"] = run_arms(p, cols)
    if "census" in want:
        out, r = census_within(p, cols, rng)
        blob["census_within_satellite"], rows = out, rows + r

    if OUT_JSON.exists():
        try:
            old = json.load(open(OUT_JSON))
            for k, v in old.items():
                blob.setdefault(k, v)
        except Exception:
            pass
    blob["column_counts"] = {k: len(v) for k, v in cols.items()}
    blob["n_non_physical_columns_in_model"] = int(nonphys)
    blob["n_folds"] = int(folds())
    blob["smoke_test"] = bool(_SMOKE["folds"])
    json.dump(blob, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\n  [saved] {OUT_JSON}")

    table = [dict(arm=k, **{kk: vv for kk, vv in v.items() if kk != "arm"})
             for k, v in (blob.get("arms") or {}).items()]
    out = pd.DataFrame(table + rows)
    if not out.empty:
        out.to_csv(OUT_CSV, index=False)
        print(f"  [saved] {OUT_CSV}")

    C.write_manifest("22_modality_arms.py",
                     inputs=[MANIFEST, C.FRAME, C.MODEL_A],
                     outputs=[OUT_JSON, OUT_CSV],
                     settings=dict(arms=want, n_folds=folds(),
                                   SHAP_ROWS=SHAP_ROWS,
                                   smoke_test=bool(_SMOKE["folds"])),
                     notes="Satellite-borne against non-satellite column sets, "
                           "and the census question asked inside the "
                           "satellite-only panel.")


if __name__ == "__main__":
    main()
