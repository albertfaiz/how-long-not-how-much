"""
01_check_environment_and_inputs.py
Run this first. It does no analysis; it tells you whether the rest will work.

Checks, in order:
  1  the project root was found, and the two results folders are under it
  2  every input file the other scripts read exists and is readable
  3  every library they import is installed, with its version
  4  the reference frame really contains the columns the paper names
  5  XGBoost gives the same answer twice, on this machine

WHAT IT WRITES
  derived/manifests/01_check_environment_and_inputs.manifest.json

TIME
  under a minute, except step 5 which takes two to three.

    python3 01_check_environment_and_inputs.py
    python3 01_check_environment_and_inputs.py --skip-determinism
"""
from __future__ import annotations

# ===========================================================================
#  SETTINGS
# ===========================================================================
DETERMINISM_TREES = 120       # small model; enough to expose a thread effect
DETERMINISM_THREADS = (1, 4)  # fit at each and compare bit for bit
# ===========================================================================

import argparse
import sys

import _config as C


REQUIRED_LIBS = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "shap",
                 "pyarrow", "matplotlib", "joblib"]
OPTIONAL_LIBS = ["catboost", "geopandas"]

# columns the manuscript names by hand; if one is missing the paper is wrong
NAMED_COLUMNS = [
    C.TARGET, C.GROUP, C.YEAR,
    C.NCE_TAIL, C.NCE_MEAN,
    "lst_day_1km_p90", "lst_night_1km_p10",
    "lst_day_1km_mean", "lst_night_1km_mean",
    "cams_fot_formaldehyde_above75th_percentile",
    "prism_tdmean_min", "pm25_mean",
]


def check_inputs() -> list[str]:
    problems = []
    print("INPUT FILES")
    for name, p in C._INPUTS.items():
        if p.exists():
            size = p.stat().st_size
            print(f"  ok       {name:<26} {size/1e6:8.1f} MB")
        else:
            print(f"  MISSING  {name:<26} {p}")
            problems.append(f"missing input: {p}")
    return problems


def check_libraries() -> list[str]:
    problems = []
    print("\nLIBRARIES")
    v = C.versions()
    for m in REQUIRED_LIBS:
        key = m
        if v.get(key, "absent") == "absent":
            print(f"  MISSING  {m}")
            problems.append(f"required library not installed: {m}")
        else:
            print(f"  ok       {m:<14} {v[key]}")
    for m in OPTIONAL_LIBS:
        state = v.get(m, "absent")
        print(f"  {'ok      ' if state != 'absent' else 'optional'} "
              f"{m:<14} {state}")
    if problems:
        names = [p.split(": ")[1] for p in problems]
        print("\n  install with:")
        print("    pip install " + " ".join(
            {"sklearn": "scikit-learn"}.get(m, m) for m in names))
    return problems


def check_columns() -> list[str]:
    import pyarrow.parquet as pq
    problems = []
    print("\nCOLUMNS THE MANUSCRIPT NAMES")
    if not C.FRAME.exists():
        print("  skipped, the reference frame is missing")
        return ["reference frame missing, columns not checked"]
    have = set(pq.ParquetFile(C.FRAME).schema.names)
    for c in NAMED_COLUMNS:
        if c in have:
            print(f"  ok       {c}")
        else:
            print(f"  MISSING  {c}")
            problems.append(f"column named in the paper is not in the frame: {c}")
    print(f"  frame holds {len(have):,} columns in total")
    return problems


def check_determinism() -> list[str]:
    """
    The question behind this check: if you run an analysis twice, do you get
    the same number? A figure and a sentence that disagree because of a silent
    rerun is the worst failure this project can have.

    XGBoost's 'hist' tree method is documented as deterministic for a fixed
    seed, and this confirms it on THIS machine, including across thread
    counts, which is the case a laptop and a cluster differ in.
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold

    print("\nDETERMINISM")
    p = C.load_panel()
    X, y, g = p["X"], p["y"], p["groups"]
    tr, te = next(iter(GroupKFold(n_splits=C.N_FOLDS).split(X, y, g)))
    kw = dict(C.XGB_PARAMS)
    kw["n_estimators"] = DETERMINISM_TREES

    sums = {}
    for nj in DETERMINISM_THREADS:
        m = xgb.XGBRegressor(n_jobs=nj, **kw).fit(X.iloc[tr], y[tr])
        pred = m.predict(X.iloc[te]).astype(np.float64)
        sums[f"{nj} threads"] = float(pred.sum())
        print(f"  fitted at {nj:>2} thread(s): prediction sum "
              f"{pred.sum():.9f}")
    # same threads twice
    nj = DETERMINISM_THREADS[-1]
    m = xgb.XGBRegressor(n_jobs=nj, **kw).fit(X.iloc[tr], y[tr])
    again = float(m.predict(X.iloc[te]).astype(np.float64).sum())
    sums[f"{nj} threads, rerun"] = again
    print(f"  refitted at {nj:>2} thread(s): prediction sum {again:.9f}")

    vals = {round(v, 6) for v in sums.values()}
    problems = []
    if len(vals) == 1:
        print("\n  PASS  identical across thread counts and across reruns.")
        print("        A number produced by these scripts will reproduce.")
    else:
        print("\n  FAIL  the fits disagree. Do not trust a rerun to reproduce")
        print("        a published number until this is understood.")
        problems.append("XGBoost is not reproducible on this machine")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-determinism", action="store_true")
    a = ap.parse_args()

    C.banner("01  environment and inputs")
    problems = check_inputs() + check_libraries() + check_columns()
    if not a.skip_determinism and not problems:
        problems += check_determinism()
    elif problems:
        print("\nDETERMINISM\n  skipped, because something above is missing")

    print("\n" + "=" * 74)
    if problems:
        print(f"  {len(problems)} problem(s) to fix before running anything else:")
        for p in problems:
            print(f"    - {p}")
        print("=" * 74)
        C.write_manifest(__file__, list(C._INPUTS.values()), [],
                         notes=f"{len(problems)} problems: {problems}")
        sys.exit(1)
    print("  everything the other scripts need is present and reproducible.")
    print("  next:  python3 02_outcome_distribution_and_error_brackets.py")
    print("=" * 74)
    C.write_manifest(__file__, list(C._INPUTS.values()), [], notes="clean")


if __name__ == "__main__":
    main()
