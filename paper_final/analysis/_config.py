"""
_config.py -- the ONLY file in this folder that knows where anything lives.

Every other script imports from here. If a path is wrong, it is wrong in one
place and you fix it once. Nothing else in this folder contains an absolute
path, and nothing contains a /mnt/ path from the machine the code was drafted
on.

    python3 _config.py          prints every path and whether it exists

HOW THE ROOT IS FOUND
---------------------
Walking up from this file until a directory containing BOTH a
results_studyA_* and a results_studyB_* folder appears. That is the shared
project folder, wherever it has been copied to: a laptop, a cluster scratch
directory, an external drive. Override it if you need to:

    export SATSDOH_ROOT=/some/other/place

RUN SETTINGS
------------
The block below is what makes a number in the paper the number it is. Every
script records these values in its manifest, so if a figure ever disagrees
with the text you can tell which settings each was produced under. Change them
here and the change propagates; change them in a script and only that script
moves.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ===========================================================================
#  RUN SETTINGS -- the tunables. Everything below this block is machinery.
# ===========================================================================

# -- reproducibility --------------------------------------------------------
SEED = 0                  # every rng in every script is seeded from this
N_THREADS = 0             # 0 = use all cores. Set a number to pin it.
                          # XGBoost 'hist' gives bit-identical answers at any
                          # thread count (tested), so this is for speed only.

# -- cross-validation -------------------------------------------------------
N_FOLDS = 8               # county-grouped. The production protocol.

# -- attribution ------------------------------------------------------------
SHAP_SAMPLE = 20000       # rows attributed. Raised from 12,000: a smoother
                          # dependence curve makes every turning point in
                          # script 06 better determined.
SHAP_BACKGROUND = 500     # background rows for the interventional value
                          # function. Raised from 100: the background set is
                          # what 'absent' means, and 100 draws of it is a
                          # noisy definition.

# -- bootstrap for turning points -------------------------------------------
N_BOOT = 5000             # 400 in the first submission, which estimates a
                          # 2.5th percentile from the 10th order statistic.
                          # 5,000 puts 125 draws below the lower endpoint and
                          # takes the Monte Carlo error on every interval
                          # endpoint below one per cent of its width. Script 06
                          # reports that error so the choice stays checkable.
N_BINS = 45               # equal-count quantile bins for the response curve

# -- the stability gate -----------------------------------------------------
# These three are OUR convention, declared in advance. They are not taken from
# any published standard, and script 06 sweeps them so the paper can show the
# verdict does not depend on where they were placed.
GATE_MIN_FOUND = 0.60     # a crossing found in at least this share of draws
GATE_MAX_WIDTH = 0.40     # interval narrower than this share of the range
GATE_INSIDE = True        # point estimate must lie inside its own interval

# -- the learner ------------------------------------------------------------
# Matches the production fit. Changing anything here invalidates every number
# the paper quotes, which is why it is written out once and imported.
XGB_PARAMS = dict(
    n_estimators=1200, learning_rate=0.03, max_depth=8, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
    random_state=SEED, tree_method="hist", enable_categorical=False,
)

# -- figures ----------------------------------------------------------------
FIG_DPI = 600
FIG_FORMATS = ("png", "pdf")

# ===========================================================================
#  PATHS
# ===========================================================================

def _find_root() -> Path:
    env = os.environ.get("SATSDOH_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
        raise SystemExit(f"SATSDOH_ROOT is set to {p}, which does not exist")
    here = Path(__file__).resolve()
    for q in [here.parent, *here.parents]:
        # Study A is the only run this paper reads. Study B (environment plus
        # socioeconomic data) is optional: no script here needs it.
        if list(q.glob("results_studyA_*")):
            return q
    # last resort: the folder whose name starts with "satellite"
    for q in here.parents:
        if q.name.lower().startswith("satellite"):
            return q
    raise SystemExit(
        "Could not find the project root. It is the folder that contains\n"
        "results_studyA_*. Either run this from inside\n"
        "that folder's tree, or set SATSDOH_ROOT to point at it:\n"
        "    export SATSDOH_ROOT=/Users/you/Desktop/satellite_sdoh")


def _one(pattern: str, parent: Path) -> Path:
    """The single directory matching a glob, newest if several."""
    hits = sorted(parent.glob(pattern))
    if not hits:
        return parent / pattern.replace("*", "MISSING")
    return hits[-1]


ROOT = _find_root()

# the two folders the main ML run produced
STUDY_A = _one("results_studyA_*", ROOT)
STUDY_B = _one("results_studyB_*", ROOT)

# where this analysis package lives, and where it writes
ANALYSIS = Path(__file__).resolve().parent
PAPER = ANALYSIS.parent                    # .../paper4_merged_sensors

# A rerun of the analysis should be able to land beside the first run rather
# than on top of it, so the two can be compared instead of one quietly
# replacing the other. SATSDOH_RUN adds a suffix to the folder that holds
# recomputed results:
#
#     export SATSDOH_RUN=independent_run      ->  derived_independent_run/
#
# which is the naming already in the working tree. Figures/ and tables_v3/ do
# NOT move with it, deliberately: those are parts of one document, and main.tex
# includes them by name, so a build that wrote them somewhere else would produce
# a paper that does not compile. To put them somewhere else anyway, name the
# folder outright, which wins over the suffix:
#
#     export SATSDOH_DERIVED=/path/to/another/derived
#     export SATSDOH_FIGURES=/path/to/another/Figures
_RUN = (os.environ.get("SATSDOH_RUN") or "").strip().strip("_")
_SUFFIX = f"_{_RUN}" if _RUN else ""


def _outdir(env_name: str, stem: str, suffixed: bool = False) -> Path:
    """The folder a run writes to: named outright, or the stem beside PAPER."""
    named = os.environ.get(env_name)
    if named:
        return Path(named).expanduser().resolve()
    return PAPER / f"{stem}{_SUFFIX if suffixed else ''}"


FIGURES = _outdir("SATSDOH_FIGURES", "Figures")
TABLES = _outdir("SATSDOH_TABLES", "tables_v3")
# Some tables ship with the paper and are read rather than written: the master
# reference table is one. Pointing SATSDOH_TABLES at a rerun's folder must not
# make those unreadable, so their location is separate and always canonical.
TABLES_SOURCE = PAPER / "tables_v3"
DERIVED = _outdir("SATSDOH_DERIVED", "derived", suffixed=True)
MANIFESTS = DERIVED / "manifests"
RUN_LABEL = _RUN or "default"

# inputs the scripts read, each named once
FRAME = STUDY_A / "studyA_reference_frame.parquet"
MODEL_A = STUDY_A / "studyA_final_model.joblib"
OOF_A = STUDY_A / "studyA_oof_predictions.csv"
RANKING_A = STUDY_A / "studyA_shap_ranking.csv"
DEPENDENCE_A = STUDY_A / "studyA_dependence_data.csv"
FEATURE_GATE_A = STUDY_A / "studyA_feature_gate.csv"
TOURNAMENT_A = STUDY_A / "studyA_model_tournament.csv"
CVFOLDS_A = STUDY_A / "studyA_cv_folds.csv"
SUMMARY_A = STUDY_A / "studyA_metrics.json"
STUDY_A_TABLES = STUDY_A / "tables"      # where the main ML run wrote its
                                         # own summary json files
RUN_LOG = ROOT / "superset_311981.log"
RUNNER = ROOT / "run_ml_satallite_and_SDOH.py"

TARGET = "meanlifeexpectency"
GROUP = "fips"
YEAR = "year"

# the two night-cooling variables the paper compares
NCE_TAIL = "eng_night_cooling_eff"
NCE_MEAN = "eng2_night_cooling_efficiency"

_INPUTS = {
    "reference frame (parquet)": FRAME,
    "fitted model (joblib)": MODEL_A,
    "out-of-fold predictions": OOF_A,
    "SHAP ranking": RANKING_A,
    "dependence sample": DEPENDENCE_A,
    "admission record": FEATURE_GATE_A,
    "learner tournament": TOURNAMENT_A,
    "fold scores": CVFOLDS_A,
    "run log": RUN_LOG,
}

# ===========================================================================
#  MACHINERY -- seeding, manifests, small helpers
# ===========================================================================


def threads() -> int:
    return N_THREADS if N_THREADS else (os.cpu_count() or 1)


def seed_everything(extra: int = 0) -> "object":
    """One seeded generator, and the global seeds set for libraries that use
    them. `extra` lets a script take an independent stream without changing
    SEED, which would move every other number in the paper."""
    import random
    import numpy as np
    s = SEED + extra
    random.seed(s)
    np.random.seed(s)
    return np.random.default_rng(s)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def versions() -> dict:
    out = {"python": sys.version.split()[0],
           "platform": f"{platform.system()} {platform.machine()}"}
    for m in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "catboost",
              "shap", "pyarrow", "matplotlib", "joblib"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = "absent"
    return out


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=5).stdout.strip() or "none"
    except Exception:
        return "none"


def write_manifest(script: str, inputs, outputs, settings=None, notes=None):
    """
    Every script calls this last. The manifest is what makes a rerun checkable:
    it records the settings that produced the numbers, the versions of the
    libraries that computed them, and a hash of every file written. Script 10
    reruns a script and compares manifests.
    """
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    inputs = [Path(p) for p in inputs]
    outputs = [Path(p) for p in outputs]
    man = dict(
        script=script,
        written_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        root=str(ROOT),
        settings=dict(SEED=SEED, N_FOLDS=N_FOLDS, SHAP_SAMPLE=SHAP_SAMPLE,
                      SHAP_BACKGROUND=SHAP_BACKGROUND, N_BOOT=N_BOOT,
                      N_BINS=N_BINS, GATE_MIN_FOUND=GATE_MIN_FOUND,
                      GATE_MAX_WIDTH=GATE_MAX_WIDTH,
                      XGB_PARAMS=XGB_PARAMS, threads=threads(),
                      **(settings or {})),
        versions=versions(),
        git=git_commit(),
        inputs={p.name: (sha256(p) if p.exists() else "MISSING")
                for p in inputs},
        outputs={p.name: (sha256(p) if p.exists() else "NOT WRITTEN")
                 for p in outputs},
        notes=notes or "",
    )
    dest = MANIFESTS / (Path(script).stem + ".manifest.json")
    json.dump(man, open(dest, "w"), indent=1, default=str)
    print(f"  [manifest] {dest.relative_to(PAPER)}")
    return man


def require(*paths) -> None:
    """Fail before doing work, with a message that says what to do."""
    missing = [Path(p) for p in paths if not Path(p).exists()]
    if missing:
        lines = "\n".join(f"    {p}" for p in missing)
        raise SystemExit(
            f"Missing input file(s):\n{lines}\n\n"
            f"Project root is currently:\n    {ROOT}\n"
            f"If that is wrong, set SATSDOH_ROOT and run again. If it is "
            f"right, the file was produced by the main ML run and should be "
            f"in {STUDY_A.name}/.")


# ---------------------------------------------------------------------------
#  Loading a result file, with a refusal
# ---------------------------------------------------------------------------
#
#  A real danger on this project, and one that has already happened twice: a
#  script crashes for want of a file, somebody writes a placeholder to get past
#  it, and the placeholder is later read as a result. Two such files exist in
#  the working tree right now --
#
#      derived/A_lifetime_summary.json                 {}
#      derived_independent_run/A_lifetime_summary.json {"short_wins": 10,
#                                                       "n_short": 20, ...}
#
#  -- against the real one in results_studyA_*/tables/, which records
#  short_wins 7 of 8 and long_wins 3 of 12 with a Mann-Whitney p of 0.0062.
#  The placeholder says ten of twenty against ten of twenty, which is exactly
#  "no effect", and it would have gone into the manuscript as the lifetime
#  result had anything read it.
#
#  So nothing in this package loads a result file by opening it. Everything
#  goes through load_result(), which is told the fields a real file must carry.
#
#  A file that fails the test is passed over in favour of the authoritative
#  copy in results_studyA_*/tables/, and the substitution is printed with both
#  paths on every occurrence. It raises only when no complete copy exists in
#  any searched location.

def _incomplete(path, required):
    """(blob, missing) for a readable JSON file; (None, reason) otherwise."""
    try:
        blob = json.load(open(path))
    except Exception as e:
        return None, [f"unreadable JSON: {e}"]
    if not blob:
        return None, ["the file is empty"]
    return blob, [k for k in required if k not in blob]


def load_result(path, required=(), search_tables=True):
    """
    Read a JSON result, after checking it is one.

    `required` lists the keys a genuine file must have. A file that is empty,
    unreadable, or missing any of them is not used.

    `search_tables` also looks in results_studyA_*/tables/, which is where the
    main ML run writes its own summaries and therefore where the authoritative
    copy of several of these files lives. That copy is used both when the
    expected file is absent and when it is present but fails the test.
    """
    path = Path(path)
    alt = STUDY_A_TABLES / path.name if search_tables else None

    if path.exists():
        blob, missing = _incomplete(path, required)
        if not missing:
            return blob
        why = (missing[0] if blob is None
               else "missing field(s) a real result must have: "
                    + ", ".join(missing))
    else:
        blob, why = None, "the file does not exist"

    # the expected file is unusable. Is the authoritative copy usable?
    if alt is not None and alt != path and alt.exists():
        other, other_missing = _incomplete(alt, required)
        if not other_missing:
            print("  !! NOT USING " + str(path))
            print("     " + why)
            print("     That looks like a placeholder written to get past a "
                  "crash, not a result.")
            print("     Reading the authoritative copy instead:")
            print("     " + str(alt))
            return other

    raise SystemExit(
        f"Cannot read a real result for {path.name}.\n"
        f"    {path}\n"
        f"        {why}\n"
        + (f"    {alt}\n        is not usable either\n"
           if alt is not None and alt != path else "")
        + f"    Rerun the script that produces it. Do not write a stand-in: a\n"
          f"    placeholder here becomes a number in the paper.")


# ---------------------------------------------------------------------------
#  Finding a table, wherever this project put it
# ---------------------------------------------------------------------------
#
#  The same quantity is written under three names in three places in this
#  project, for historical reasons that are not worth unpicking now:
#
#      derived/bracket_error.csv
#      results_studyA_*/tables/A_bracket_error.csv
#      results_studyA_*/studyA_bracket_error.csv
#
#  A script that opens one of those by name works in the folder it was written
#  in and nowhere else. load_table takes the bare stem and looks for every
#  spelling in every place, in a fixed order, and says where it looked when it
#  cannot find any of them.

def table_candidates(stem: str):
    """
    Every path this project might have written the table `stem` to.

    A name like "A_cv_folds", "studyA_cv_folds" or "cv_folds" all mean the same
    table, and the study it belongs to is carried by the prefix. A B_ prefix
    sends the search to the Study B folders first, an A_ prefix or no prefix to
    Study A's.
    """
    stem = str(stem)
    if stem.endswith(".csv"):
        stem = stem[: -len(".csv")]
    study = "A"
    for pre in ("studyA_", "studyB_", "A_", "B_"):
        if stem.startswith(pre):
            study = "B" if "B" in pre else "A"
            stem = stem[len(pre):]
            break
    names = [f"{stem}.csv", f"{study}_{stem}.csv", f"study{study}_{stem}.csv"]
    home = STUDY_B if study == "B" else STUDY_A
    folders = [DERIVED, home / "tables", home, PAPER / "derived"]
    # the other study, last, because a handful of tables were written once into
    # whichever folder the script that made them happened to be run from
    other = STUDY_A if study == "B" else STUDY_B
    folders += [other / "tables", other]
    for folder in folders:
        for n in names:
            yield folder / n


def load_table(stem: str, **kw):
    """
    Read a CSV by its bare name, from wherever in this project it lives.

        load_table("bracket_error")        # any of the three spellings above

    Raises with the full list of places tried, so a missing table is a question
    about which script has not been run rather than a stack trace.
    """
    import pandas as pd
    tried = []
    for p in table_candidates(stem):
        if p in tried:
            continue
        tried.append(p)
        if p.exists():
            return pd.read_csv(p, **kw)
    raise SystemExit(
        f"No table found for {stem!r}. Looked in:\n"
        + "\n".join(f"    {p}" for p in tried)
        + "\n    If the first of those is the one you expect, the script that\n"
          "    writes it has not been run yet; RUN_ORDER.md says which.")


# ---------------------------------------------------------------------------
#  Cross-validated predictions, checkpointed per fold
# ---------------------------------------------------------------------------
#  Several scripts fit the production model once per fold, which is minutes per
#  fold. Writing each fold's out-of-fold predictions as they are produced lets
#  an interrupted run resume instead of restarting, and lets a long battery be
#  spread over several sessions. The tag names the experiment and must be
#  unique: two experiments sharing a tag would read each other's folds.

def cv_oof(X, y, groups, tag, n_folds=None, params=None, verbose=True):
    """Out-of-fold predictions over grouped folds, resuming from checkpoints."""
    import gc
    import numpy as _np
    from sklearn.model_selection import GroupKFold
    import xgboost as xgb

    n_folds = n_folds or N_FOLDS
    # The fold count is part of the checkpoint's identity. Without it, a smoke
    # test at three folds writes fold01 and fold02 and a later production run
    # at eight folds reads them as its own first two folds, which are
    # different splits entirely. The production fold count keeps the plain
    # path so checkpoints already on disk are still found.
    store = DERIVED / "checkpoints_cv" / (
        tag if n_folds == N_FOLDS else f"{tag}_f{n_folds}")
    store.mkdir(parents=True, exist_ok=True)
    oof = _np.full(len(y), _np.nan)
    for i, (tr, te) in enumerate(
            GroupKFold(n_splits=n_folds).split(X, y, groups), 1):
        f = store / f"fold{i:02d}.npy"
        if f.exists():
            oof[te] = _np.load(f)
            if verbose:
                print(f"      {tag:<28} fold {i}/{n_folds}  from checkpoint",
                      flush=True)
            continue
        m = xgb.XGBRegressor(**(params or XGB_PARAMS), n_jobs=threads())
        m.fit(X.iloc[tr], y[tr])
        pred = m.predict(X.iloc[te])
        _np.save(f, pred)
        oof[te] = pred
        del m
        gc.collect()
        if verbose:
            print(f"      {tag:<28} fold {i}/{n_folds}  fitted", flush=True)
    if _np.isnan(oof).any():
        raise SystemExit(f"{tag}: some rows have no out-of-fold prediction")
    return oof


def cv_score(X, y, groups, tag, **kw):
    """R2 and MAE from cv_oof, with the column count."""
    from sklearn.metrics import r2_score, mean_absolute_error
    oof = cv_oof(X, y, groups, tag, **kw)
    return dict(label=tag, n_features=int(X.shape[1]),
                r2=float(r2_score(y, oof)),
                mae=float(mean_absolute_error(y, oof)))


def dependence_table():
    """
    (table, feature names) for script 05's attribution surface.

    The table is WIDE: one column per candidate holding the predictor value
    and one more named `shap_<candidate>` holding that row's attribution. Two
    later scripts assumed a long table with feature/x/shap columns and failed
    on the first read; the idiom is written down once here so a third does not.

        d, feats = C.dependence_table()
        x = d[f].to_numpy(float)
        v = d["shap_" + f].to_numpy(float)
    """
    d = load_table("shap_dependence_candidates")
    feats = [c for c in d.columns if "shap_" + c in d.columns]
    if not feats:
        raise SystemExit(
            "shap_dependence_candidates.csv carries no candidate paired with "
            "a shap_ column.\n    Columns: " + ", ".join(map(str, d.columns)))
    return d, feats


def census_pairs(columns=None, census=None):
    """
    The (field, central column, non-central column) triples the census compares.

    Several studies ask the census's question with a different measuring
    instrument -- mutual information, a second learner, an era at a time -- and
    every one of them needs the same list of pairs. Built once here so two
    scripts cannot disagree about which pairs they are, which has happened:
    an earlier version of one check used each field's OVERALL best statistic as
    its non-central one, so for the fields whose best statistic IS central it
    compared a column with itself and recorded a tie.

    `columns` restricts the result to pairs whose both members are present.
    """
    census = load_table("summary_statistic_census") if census is None else census
    have = set(columns) if columns is not None else None
    out = []
    for _, r in census.iterrows():
        c_stat = str(r.get("central_stat") or "").strip()
        n_stat = str(r.get("noncentral_stat") or "").strip()
        if not c_stat or not n_stat or "nan" in (c_stat.lower(), n_stat.lower()):
            continue
        field = str(r["field"])
        c_col, n_col = f"{field}_{c_stat}", f"{field}_{n_stat}"
        if c_col == n_col:
            continue
        if have is not None and (c_col not in have or n_col not in have):
            continue
        out.append((field, c_col, n_col))
    return out


def banner(title: str) -> None:
    print("=" * 74)
    print(f"  {title}")
    print("=" * 74)
    print(f"  root     {ROOT}")
    print(f"  study A  {STUDY_A.name}")
    print(f"  seed {SEED}   folds {N_FOLDS}   threads {threads()}")
    print(f"  run      {RUN_LABEL}   "
          f"(results to {DERIVED.name}/, figures to {FIGURES.name}/)")
    print("-" * 74)


def load_panel(columns=None):
    """
    The modelling panel, exactly as every script needs it: the model's own
    feature list, its median fill values, its year window, and the rows with an
    outcome. Written once here so no two scripts can build it differently.
    """
    import joblib
    import pyarrow.parquet as pq
    require(FRAME, MODEL_A)
    d = joblib.load(MODEL_A)
    feats, fill = d["features"], d["median_fill"]
    lo, hi = d["window"]
    cols = (columns or feats) + [GROUP, YEAR, TARGET]
    df = pq.read_table(FRAME, columns=sorted(set(cols))).to_pandas()
    df = df[(df[YEAR] >= lo) & (df[YEAR] <= hi)
            & df[TARGET].notna()].reset_index(drop=True)
    X = df[feats].copy() if columns is None else None
    if X is not None:
        for c, v in fill.items():
            if c in X.columns:
                X[c] = X[c].fillna(v)
        X = X.astype("float32")
    return dict(df=df, X=X, features=feats, window=(lo, hi),
                y=df[TARGET].to_numpy(), groups=df[GROUP].to_numpy())


if __name__ == "__main__":
    banner("paths and inputs")
    for name, p in _INPUTS.items():
        mark = "ok     " if Path(p).exists() else "MISSING"
        print(f"  {mark}  {name:<26} {p}")
    print("-" * 74)
    for name, p in (("FIGURES", FIGURES), ("TABLES", TABLES),
                    ("DERIVED", DERIVED)):
        print(f"  {'ok     ' if p.exists() else 'will be created'}  "
              f"{name:<26} {p}")
    print("-" * 74)
    v = versions()
    missing = [k for k, w in v.items() if w == "absent"]
    print("  libraries:", ", ".join(f"{k} {w}" for k, w in v.items()
                                    if w != "absent"))
    if missing:
        print("  NOT INSTALLED:", ", ".join(missing))
        print("\n  install what you need with:")
        print("    pip install " + " ".join(
            {"sklearn": "scikit-learn"}.get(m, m) for m in missing))
