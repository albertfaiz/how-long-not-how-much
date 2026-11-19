#!/usr/bin/env python3
"""
paper4_analysis_all_in_one.py -- every analysis behind the final Paper 4
(version 28), in one file, and nothing the final paper does not use.

This file carries, verbatim, the code of every script that produced a number,
a table or a result file of the final paper, as kept in analysis_final/:

    analysis/_config.py              paths and run settings (seed, folds, ...)
    analysis/01_ ... 34_*.py         the 23 analysis steps the final paper
                                     uses, in run order (steps 12, 14 and 20
                                     are the final-paper copies, without the
                                     parts it no longer reports)
    analysis/build_paper_constants.py   every number the manuscript prints
    analysis/paper_figures.py        figure code; its chemical-family rules
                                     are imported by build_paper_constants
    pipeline/*.py                    the main model run, as run on the Juno
                                     cluster (kept for reference)

The code is stored at the end of this file as comment lines that begin with
'#|', so it can be read and searched here, and it is run exactly as stored:
a step executes its script's code in this process, with _config supplied
from the same store, so its results are the results of the script itself.

    python3 paper4_analysis_all_in_one.py --list
    python3 paper4_analysis_all_in_one.py --run 02
    python3 paper4_analysis_all_in_one.py --run 14 -- --all-candidates   # options after --
    python3 paper4_analysis_all_in_one.py --run constants
    python3 paper4_analysis_all_in_one.py --run all-reads       # every "reads" step
    python3 paper4_analysis_all_in_one.py --show 28             # print a script
    python3 paper4_analysis_all_in_one.py --extract ../my_copy  # write them all out
    python3 paper4_analysis_all_in_one.py --verify              # store = files here?

Keep this file in paper_final/analysis_final/ (where the scripts live): the paths
are found from there, exactly as the scripts find them. Elsewhere, set
SATSDOH_ROOT to the project folder that holds results_studyA_*.

Steps marked "refits" fit XGBoost again, recompute SHAP values or draw
bootstrap replicates, and take minutes to hours; steps marked "reads" work
from result files already written and take seconds to a minute. Nothing here
uses synthetic or mock data: every step reads the project's own files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path

THIS = Path(__file__).resolve()
HERE = THIS.parent

STEPS = [
    ('01', '01_check_environment_and_inputs.py',
     'Check the Python environment and that every input file is present', 'refits'),
    ('02', '02_outcome_distribution_and_error_brackets.py',
     'Distribution of the outcome and model error by outcome bracket', 'reads'),
    ('03', '03_baselines_five_reference_models.py',
     'Five reference baselines on the stored county-grouped folds', 'refits'),
    ('04', '04_refit_drop_one_nce_construction.py',
     'Drop-one refit of the two night-cooling constructions', 'refits'),
    ('05', '05_recompute_shap_for_threshold_candidates.py',
     'Recompute SHAP values for the turning-point candidates', 'refits'),
    ('06', '06_bootstrap_turning_points_and_gate.py',
     'Bootstrap the turning points and apply the stability rule', 'refits'),
    ('07', '07_learner_and_hyperparameter_table.py',
     'The learner and its hyperparameters (Appendix Table A1)', 'reads'),
    ('10', '10_verify_reproducibility_by_rerunning.py',
     'Rerun a script and compare its outputs with the earlier run', 'reads'),
    ('11', '11_verify_nce_identity_by_search.py',
     'Is the night-cooling pair one field described twice?', 'reads'),
    ('12', '12_fields_with_several_summaries.py',
     'Fields with several admitted summaries and the spread of their attribution', 'reads'),
    ('13', '13_spatial_holdout_and_attribution_structure.py',
     'State hold-out and the structure of attribution', 'refits'),
    ('14', '14_regional_invariance_of_turning_points.py',
     'Regional invariance of the turning points (median longitude and 100th meridian)', 'reads'),
    ('16', '16_export_frozen_panels_and_gate_audit.py',
     'Frozen modelling panels and the coverage-gate audit', 'refits'),
    ('17', '17_drop_formaldehyde_durations.py',
     'Refits without the formaldehyde durations, singly and together', 'refits'),
    ('19', '19_temporal_validation.py',
     'Temporal validation: predicting a year not seen', 'refits'),
    ('20', '20_null_skill_and_null_crossing.py',
     'Null skill and null turning points', 'refits'),
    ('22', '22_modality_arms.py',
     'Satellite-only and non-satellite modality arms', 'refits'),
    ('23', '23_influence_and_perturbation.py',
     'Influence of single counties and predictor perturbation', 'refits'),
    ('25', '25_lifetime_literature_test.py',
     'Duration advantage against atmospheric lifetime', 'reads'),
    ('26', '26_rank_stability_and_learning_curve.py',
     'Rank stability and the learning curve', 'refits'),
    ('28', '28_residual_and_social_determinants.py',
     'Residual audit against social determinants of health', 'refits'),
    ('30', '30_residual_audit_sensitivity.py',
     'Sensitivity of the residual audit', 'reads'),
    ('33', '33_export_figure10c_data.py',
     'The data behind Figure 8C', 'reads'),
    ('34', '34_library_version_sensitivity.py',
     'Sensitivity of the numbers to library versions', 'reads'),
    ('35', '35_check_method_statements.py',
     'Check the statements about methods and data against the code', 'reads'),
    ('constants', 'build_paper_constants.py',
     'Write every number and generated table the manuscript prints', 'reads'),
]


# ---------------------------------------------------------------- the store
def store() -> dict:
    """{relative path: source text} read from the end of this file."""
    out, cur, buf = {}, None, []
    for ln in THIS.read_text().split("\n"):
        if ln.startswith("#@@ FILE "):
            cur, buf = ln[len("#@@ FILE "):], []
        elif ln.startswith("#@@ END ") and cur:
            out[cur] = "\n".join(buf)
            cur = None
        elif cur is not None and ln.startswith("#|"):
            buf.append(ln[3:] if ln.startswith("#| ") else "")
    return out


def _by_name(src: dict) -> dict:
    return {Path(k).name: v for k, v in src.items()}


# ---------------------------------------------------------------- running
class _StoredLoader:
    """Loads a script from the store when another script asks for it by path
    (scripts 14, 20, 23, 24 and 29 reuse functions of scripts 06, 12 and the
    main run this way)."""

    def __init__(self, source, path):
        self.source, self.path = source, path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = self.path
        exec(compile(self.source, self.path, "exec"), module.__dict__)


class _StoredFinder:
    """Lets a script import a stored module by name (build_paper_constants
    imports the chemical-family rules from paper_figures)."""

    def __init__(self, by_name):
        self.by_name = by_name

    def find_spec(self, name, path=None, target=None):
        fn = name + ".py"
        if name != "_config" and fn in self.by_name:
            loc = str(HERE / fn)
            return importlib.util.spec_from_loader(
                name, _StoredLoader(self.by_name[fn], loc), origin=loc)
        return None


def _install(src: dict):
    by_name = _by_name(src)
    sys.meta_path.insert(0, _StoredFinder(by_name))
    cfg = types.ModuleType("_config")
    cfg.__file__ = str(HERE / "_config.py")
    sys.modules["_config"] = cfg
    exec(compile(src["analysis/_config.py"], cfg.__file__, "exec"), cfg.__dict__)

    real = importlib.util.spec_from_file_location

    def from_store(name, location=None, *a, **k):
        base = Path(str(location)).name if location else ""
        if base in by_name:
            return importlib.util.spec_from_loader(
                name, _StoredLoader(by_name[base], str(location)),
                origin=str(location))
        return real(name, location, *a, **k)
    importlib.util.spec_from_file_location = from_store


def run_step(key: str, extra, src: dict):
    hit = [s for s in STEPS if s[0] == key]
    if not hit:
        raise SystemExit(f"no step {key!r}; --list shows them")
    _key, name, purpose, _kind = hit[0]
    if key == "10" and not (HERE / "01_check_environment_and_inputs.py").exists():
        raise SystemExit("Step 10 reruns the scripts as separate programs, so it "
                         "needs them as files:\n    python3 " + THIS.name +
                         " --extract <folder>  and run 10 from there.")
    path = str(HERE / name)
    print(f"\n=== step {key}: {purpose}  [{name}]")
    sys.argv = [path, *extra]
    g = {"__name__": "__main__", "__file__": path, "__builtins__": __builtins__}
    exec(compile(src["analysis/" + name], path, "exec"), g)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Every analysis of Paper 4.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", metavar="STEP",
                    help="a step number, 'constants', 'all-reads' or 'all'")
    ap.add_argument("--show", metavar="STEP")
    ap.add_argument("--extract", metavar="FOLDER")
    ap.add_argument("--verify", action="store_true",
                    help="compare the stored code with the script files here")
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    src = store()

    if a.list or not (a.run or a.show or a.extract or a.verify):
        print(f"{'step':<10} {'kind':<8} what it does")
        for k, n, d, c in STEPS:
            print(f"{k:<10} {c:<8} {d}")
        print(f"\n{len(src)} files stored, "
              f"{sum(s.count(chr(10)) + 1 for s in src.values()):,} lines of code")
        return
    if a.show:
        hit = [s for s in STEPS if s[0] == a.show] or [
            (None, a.show, "", "")]
        rel = next((r for r in src if Path(r).name in (hit[0][1], a.show)), None)
        if rel is None:
            raise SystemExit(f"nothing stored as {a.show!r}")
        print(src[rel])
        return
    if a.extract:
        dest = Path(a.extract).expanduser().resolve()
        for rel, text in src.items():
            p = dest / rel
            if p.exists():
                raise SystemExit(f"{p} exists; choose an empty folder")
        for rel, text in src.items():
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        print(f"{len(src)} files written under {dest}")
        return
    if a.verify:
        bad = 0
        for rel, text in src.items():
            cand = [HERE / Path(rel).name, HERE.parent / rel]
            p = next((c for c in cand if c.exists()), None)
            if p is None:
                print(f"  not here      {rel}")
                continue
            same = hashlib.sha256(p.read_bytes()).hexdigest() == \
                hashlib.sha256(text.encode()).hexdigest()
            bad += not same
            print(f"  {'identical' if same else 'DIFFERS  '}     {rel}")
        if bad:
            raise SystemExit(f"{bad} stored file(s) differ from the files here")
        return

    _install(src)
    if a.run in ("all", "all-reads"):
        keys = [k for k, _n, _d, c in STEPS
                if k != "10" and (a.run == "all" or c == "reads")]
    else:
        keys = [a.run]
    for k in keys:
        run_step(k, extra, src)


if __name__ == "__main__":
    main()

# ===========================================================================
#  STORED SOURCES  (verbatim; every line below is '#|' plus the original)
# ===========================================================================
#@@ FILE analysis/_config.py
#| """
#| _config.py -- the ONLY file in this folder that knows where anything lives.
#|
#| Every other script imports from here. If a path is wrong, it is wrong in one
#| place and you fix it once. Nothing else in this folder contains an absolute
#| path, and nothing contains a /mnt/ path from the machine the code was drafted
#| on.
#|
#|     python3 _config.py          prints every path and whether it exists
#|
#| HOW THE ROOT IS FOUND
#| ---------------------
#| Walking up from this file until a directory containing BOTH a
#| results_studyA_* and a results_studyB_* folder appears. That is the shared
#| project folder, wherever it has been copied to: a laptop, a cluster scratch
#| directory, an external drive. Override it if you need to:
#|
#|     export SATSDOH_ROOT=/some/other/place
#|
#| RUN SETTINGS
#| ------------
#| The block below is what makes a number in the paper the number it is. Every
#| script records these values in its manifest, so if a figure ever disagrees
#| with the text you can tell which settings each was produced under. Change them
#| here and the change propagates; change them in a script and only that script
#| moves.
#| """
#| from __future__ import annotations
#|
#| import hashlib
#| import json
#| import os
#| import platform
#| import subprocess
#| import sys
#| from datetime import datetime, timezone
#| from pathlib import Path
#|
#| # ===========================================================================
#| #  RUN SETTINGS -- the tunables. Everything below this block is machinery.
#| # ===========================================================================
#|
#| # -- reproducibility --------------------------------------------------------
#| SEED = 0                  # every rng in every script is seeded from this
#| N_THREADS = 0             # 0 = use all cores. Set a number to pin it.
#|                           # XGBoost 'hist' gives bit-identical answers at any
#|                           # thread count (tested), so this is for speed only.
#|
#| # -- cross-validation -------------------------------------------------------
#| N_FOLDS = 8               # county-grouped. The production protocol.
#|
#| # -- attribution ------------------------------------------------------------
#| SHAP_SAMPLE = 20000       # rows attributed. Raised from 12,000: a smoother
#|                           # dependence curve makes every turning point in
#|                           # script 06 better determined.
#| SHAP_BACKGROUND = 500     # background rows for the interventional value
#|                           # function. Raised from 100: the background set is
#|                           # what 'absent' means, and 100 draws of it is a
#|                           # noisy definition.
#|
#| # -- bootstrap for turning points -------------------------------------------
#| N_BOOT = 5000             # 400 in the first submission, which estimates a
#|                           # 2.5th percentile from the 10th order statistic.
#|                           # 5,000 puts 125 draws below the lower endpoint and
#|                           # takes the Monte Carlo error on every interval
#|                           # endpoint below one per cent of its width. Script 06
#|                           # reports that error so the choice stays checkable.
#| N_BINS = 45               # equal-count quantile bins for the response curve
#|
#| # -- the stability gate -----------------------------------------------------
#| # These three are OUR convention, declared in advance. They are not taken from
#| # any published standard, and script 06 sweeps them so the paper can show the
#| # verdict does not depend on where they were placed.
#| GATE_MIN_FOUND = 0.60     # a crossing found in at least this share of draws
#| GATE_MAX_WIDTH = 0.40     # interval narrower than this share of the range
#| GATE_INSIDE = True        # point estimate must lie inside its own interval
#|
#| # -- the learner ------------------------------------------------------------
#| # Matches the production fit. Changing anything here invalidates every number
#| # the paper quotes, which is why it is written out once and imported.
#| XGB_PARAMS = dict(
#|     n_estimators=1200, learning_rate=0.03, max_depth=8, min_child_weight=10,
#|     subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
#|     random_state=SEED, tree_method="hist", enable_categorical=False,
#| )
#|
#| # -- figures ----------------------------------------------------------------
#| FIG_DPI = 600
#| FIG_FORMATS = ("png", "pdf")
#|
#| # ===========================================================================
#| #  PATHS
#| # ===========================================================================
#|
#| def _find_root() -> Path:
#|     env = os.environ.get("SATSDOH_ROOT")
#|     if env:
#|         p = Path(env).expanduser().resolve()
#|         if p.exists():
#|             return p
#|         raise SystemExit(f"SATSDOH_ROOT is set to {p}, which does not exist")
#|     here = Path(__file__).resolve()
#|     for q in [here.parent, *here.parents]:
#|         # Study A is the only run this paper reads. Study B (environment plus
#|         # socioeconomic data) is optional: no script here needs it.
#|         if list(q.glob("results_studyA_*")):
#|             return q
#|     # last resort: the folder whose name starts with "satellite"
#|     for q in here.parents:
#|         if q.name.lower().startswith("satellite"):
#|             return q
#|     raise SystemExit(
#|         "Could not find the project root. It is the folder that contains\n"
#|         "results_studyA_*. Either run this from inside\n"
#|         "that folder's tree, or set SATSDOH_ROOT to point at it:\n"
#|         "    export SATSDOH_ROOT=/Users/you/Desktop/satellite_sdoh")
#|
#|
#| def _one(pattern: str, parent: Path) -> Path:
#|     """The single directory matching a glob, newest if several."""
#|     hits = sorted(parent.glob(pattern))
#|     if not hits:
#|         return parent / pattern.replace("*", "MISSING")
#|     return hits[-1]
#|
#|
#| ROOT = _find_root()
#|
#| # the two folders the main ML run produced
#| STUDY_A = _one("results_studyA_*", ROOT)
#| STUDY_B = _one("results_studyB_*", ROOT)
#|
#| # where this analysis package lives, and where it writes
#| ANALYSIS = Path(__file__).resolve().parent
#| PAPER = ANALYSIS.parent                    # .../paper4_merged_sensors
#|
#| # A rerun of the analysis should be able to land beside the first run rather
#| # than on top of it, so the two can be compared instead of one quietly
#| # replacing the other. SATSDOH_RUN adds a suffix to the folder that holds
#| # recomputed results:
#| #
#| #     export SATSDOH_RUN=independent_run      ->  derived_independent_run/
#| #
#| # which is the naming already in the working tree. Figures/ and tables_v3/ do
#| # NOT move with it, deliberately: those are parts of one document, and main.tex
#| # includes them by name, so a build that wrote them somewhere else would produce
#| # a paper that does not compile. To put them somewhere else anyway, name the
#| # folder outright, which wins over the suffix:
#| #
#| #     export SATSDOH_DERIVED=/path/to/another/derived
#| #     export SATSDOH_FIGURES=/path/to/another/Figures
#| _RUN = (os.environ.get("SATSDOH_RUN") or "").strip().strip("_")
#| _SUFFIX = f"_{_RUN}" if _RUN else ""
#|
#|
#| def _outdir(env_name: str, stem: str, suffixed: bool = False) -> Path:
#|     """The folder a run writes to: named outright, or the stem beside PAPER."""
#|     named = os.environ.get(env_name)
#|     if named:
#|         return Path(named).expanduser().resolve()
#|     return PAPER / f"{stem}{_SUFFIX if suffixed else ''}"
#|
#|
#| FIGURES = _outdir("SATSDOH_FIGURES", "Figures")
#| TABLES = _outdir("SATSDOH_TABLES", "tables_v3")
#| # Some tables ship with the paper and are read rather than written: the master
#| # reference table is one. Pointing SATSDOH_TABLES at a rerun's folder must not
#| # make those unreadable, so their location is separate and always canonical.
#| TABLES_SOURCE = PAPER / "tables_v3"
#| DERIVED = _outdir("SATSDOH_DERIVED", "derived", suffixed=True)
#| MANIFESTS = DERIVED / "manifests"
#| RUN_LABEL = _RUN or "default"
#|
#| # inputs the scripts read, each named once
#| FRAME = STUDY_A / "studyA_reference_frame.parquet"
#| MODEL_A = STUDY_A / "studyA_final_model.joblib"
#| OOF_A = STUDY_A / "studyA_oof_predictions.csv"
#| RANKING_A = STUDY_A / "studyA_shap_ranking.csv"
#| DEPENDENCE_A = STUDY_A / "studyA_dependence_data.csv"
#| FEATURE_GATE_A = STUDY_A / "studyA_feature_gate.csv"
#| TOURNAMENT_A = STUDY_A / "studyA_model_tournament.csv"
#| CVFOLDS_A = STUDY_A / "studyA_cv_folds.csv"
#| SUMMARY_A = STUDY_A / "studyA_metrics.json"
#| STUDY_A_TABLES = STUDY_A / "tables"      # where the main ML run wrote its
#|                                          # own summary json files
#| RUN_LOG = ROOT / "superset_311981.log"
#| RUNNER = ROOT / "run_ml_satallite_and_SDOH.py"
#|
#| TARGET = "meanlifeexpectency"
#| GROUP = "fips"
#| YEAR = "year"
#|
#| # the two night-cooling variables the paper compares
#| NCE_TAIL = "eng_night_cooling_eff"
#| NCE_MEAN = "eng2_night_cooling_efficiency"
#|
#| _INPUTS = {
#|     "reference frame (parquet)": FRAME,
#|     "fitted model (joblib)": MODEL_A,
#|     "out-of-fold predictions": OOF_A,
#|     "SHAP ranking": RANKING_A,
#|     "dependence sample": DEPENDENCE_A,
#|     "admission record": FEATURE_GATE_A,
#|     "learner tournament": TOURNAMENT_A,
#|     "fold scores": CVFOLDS_A,
#|     "run log": RUN_LOG,
#| }
#|
#| # ===========================================================================
#| #  MACHINERY -- seeding, manifests, small helpers
#| # ===========================================================================
#|
#|
#| def threads() -> int:
#|     return N_THREADS if N_THREADS else (os.cpu_count() or 1)
#|
#|
#| def seed_everything(extra: int = 0) -> "object":
#|     """One seeded generator, and the global seeds set for libraries that use
#|     them. `extra` lets a script take an independent stream without changing
#|     SEED, which would move every other number in the paper."""
#|     import random
#|     import numpy as np
#|     s = SEED + extra
#|     random.seed(s)
#|     np.random.seed(s)
#|     return np.random.default_rng(s)
#|
#|
#| def sha256(path) -> str:
#|     h = hashlib.sha256()
#|     with open(path, "rb") as fh:
#|         for chunk in iter(lambda: fh.read(1 << 20), b""):
#|             h.update(chunk)
#|     return h.hexdigest()[:16]
#|
#|
#| def versions() -> dict:
#|     out = {"python": sys.version.split()[0],
#|            "platform": f"{platform.system()} {platform.machine()}"}
#|     for m in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "catboost",
#|               "shap", "pyarrow", "matplotlib", "joblib"):
#|         try:
#|             out[m] = __import__(m).__version__
#|         except Exception:
#|             out[m] = "absent"
#|     return out
#|
#|
#| def git_commit() -> str:
#|     try:
#|         return subprocess.run(
#|             ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
#|             capture_output=True, text=True, timeout=5).stdout.strip() or "none"
#|     except Exception:
#|         return "none"
#|
#|
#| def write_manifest(script: str, inputs, outputs, settings=None, notes=None):
#|     """
#|     Every script calls this last. The manifest is what makes a rerun checkable:
#|     it records the settings that produced the numbers, the versions of the
#|     libraries that computed them, and a hash of every file written. Script 10
#|     reruns a script and compares manifests.
#|     """
#|     MANIFESTS.mkdir(parents=True, exist_ok=True)
#|     inputs = [Path(p) for p in inputs]
#|     outputs = [Path(p) for p in outputs]
#|     man = dict(
#|         script=script,
#|         written_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
#|         root=str(ROOT),
#|         settings=dict(SEED=SEED, N_FOLDS=N_FOLDS, SHAP_SAMPLE=SHAP_SAMPLE,
#|                       SHAP_BACKGROUND=SHAP_BACKGROUND, N_BOOT=N_BOOT,
#|                       N_BINS=N_BINS, GATE_MIN_FOUND=GATE_MIN_FOUND,
#|                       GATE_MAX_WIDTH=GATE_MAX_WIDTH,
#|                       XGB_PARAMS=XGB_PARAMS, threads=threads(),
#|                       **(settings or {})),
#|         versions=versions(),
#|         git=git_commit(),
#|         inputs={p.name: (sha256(p) if p.exists() else "MISSING")
#|                 for p in inputs},
#|         outputs={p.name: (sha256(p) if p.exists() else "NOT WRITTEN")
#|                  for p in outputs},
#|         notes=notes or "",
#|     )
#|     dest = MANIFESTS / (Path(script).stem + ".manifest.json")
#|     json.dump(man, open(dest, "w"), indent=1, default=str)
#|     try:
#|         shown = dest.relative_to(PAPER)
#|     except ValueError:                 # results sent outside the paper folder
#|         shown = dest
#|     print(f"  [manifest] {shown}")
#|     return man
#|
#|
#| def require(*paths) -> None:
#|     """Fail before doing work, with a message that says what to do."""
#|     missing = [Path(p) for p in paths if not Path(p).exists()]
#|     if missing:
#|         lines = "\n".join(f"    {p}" for p in missing)
#|         raise SystemExit(
#|             f"Missing input file(s):\n{lines}\n\n"
#|             f"Project root is currently:\n    {ROOT}\n"
#|             f"If that is wrong, set SATSDOH_ROOT and run again. If it is "
#|             f"right, the file was produced by the main ML run and should be "
#|             f"in {STUDY_A.name}/.")
#|
#|
#| # ---------------------------------------------------------------------------
#| #  Loading a result file, with a refusal
#| # ---------------------------------------------------------------------------
#| #
#| #  A real danger on this project, and one that has already happened twice: a
#| #  script crashes for want of a file, somebody writes a placeholder to get past
#| #  it, and the placeholder is later read as a result. Two such files exist in
#| #  the working tree right now --
#| #
#| #      derived/A_lifetime_summary.json                 {}
#| #      derived_independent_run/A_lifetime_summary.json {"short_wins": 10,
#| #                                                       "n_short": 20, ...}
#| #
#| #  -- against the real one in results_studyA_*/tables/, which records
#| #  short_wins 7 of 8 and long_wins 3 of 12 with a Mann-Whitney p of 0.0062.
#| #  The placeholder says ten of twenty against ten of twenty, which is exactly
#| #  "no effect", and it would have gone into the manuscript as the lifetime
#| #  result had anything read it.
#| #
#| #  So nothing in this package loads a result file by opening it. Everything
#| #  goes through load_result(), which is told the fields a real file must carry.
#| #
#| #  A file that fails the test is passed over in favour of the authoritative
#| #  copy in results_studyA_*/tables/, and the substitution is printed with both
#| #  paths on every occurrence. It raises only when no complete copy exists in
#| #  any searched location.
#|
#| def _incomplete(path, required):
#|     """(blob, missing) for a readable JSON file; (None, reason) otherwise."""
#|     try:
#|         blob = json.load(open(path))
#|     except Exception as e:
#|         return None, [f"unreadable JSON: {e}"]
#|     if not blob:
#|         return None, ["the file is empty"]
#|     return blob, [k for k in required if k not in blob]
#|
#|
#| def load_result(path, required=(), search_tables=True):
#|     """
#|     Read a JSON result, after checking it is one.
#|
#|     `required` lists the keys a genuine file must have. A file that is empty,
#|     unreadable, or missing any of them is not used.
#|
#|     `search_tables` also looks in results_studyA_*/tables/, which is where the
#|     main ML run writes its own summaries and therefore where the authoritative
#|     copy of several of these files lives. That copy is used both when the
#|     expected file is absent and when it is present but fails the test.
#|     """
#|     path = Path(path)
#|     alt = STUDY_A_TABLES / path.name if search_tables else None
#|
#|     if path.exists():
#|         blob, missing = _incomplete(path, required)
#|         if not missing:
#|             return blob
#|         why = (missing[0] if blob is None
#|                else "missing field(s) a real result must have: "
#|                     + ", ".join(missing))
#|     else:
#|         blob, why = None, "the file does not exist"
#|
#|     # the expected file is unusable. Is the authoritative copy usable?
#|     if alt is not None and alt != path and alt.exists():
#|         other, other_missing = _incomplete(alt, required)
#|         if not other_missing:
#|             print("  !! NOT USING " + str(path))
#|             print("     " + why)
#|             print("     That looks like a placeholder written to get past a "
#|                   "crash, not a result.")
#|             print("     Reading the authoritative copy instead:")
#|             print("     " + str(alt))
#|             return other
#|
#|     raise SystemExit(
#|         f"Cannot read a real result for {path.name}.\n"
#|         f"    {path}\n"
#|         f"        {why}\n"
#|         + (f"    {alt}\n        is not usable either\n"
#|            if alt is not None and alt != path else "")
#|         + f"    Rerun the script that produces it. Do not write a stand-in: a\n"
#|           f"    placeholder here becomes a number in the paper.")
#|
#|
#| # ---------------------------------------------------------------------------
#| #  Finding a table, wherever this project put it
#| # ---------------------------------------------------------------------------
#| #
#| #  The same quantity is written under three names in three places in this
#| #  project, for historical reasons that are not worth unpicking now:
#| #
#| #      derived/bracket_error.csv
#| #      results_studyA_*/tables/A_bracket_error.csv
#| #      results_studyA_*/studyA_bracket_error.csv
#| #
#| #  A script that opens one of those by name works in the folder it was written
#| #  in and nowhere else. load_table takes the bare stem and looks for every
#| #  spelling in every place, in a fixed order, and says where it looked when it
#| #  cannot find any of them.
#|
#| def table_candidates(stem: str):
#|     """
#|     Every path this project might have written the table `stem` to.
#|
#|     A name like "A_cv_folds", "studyA_cv_folds" or "cv_folds" all mean the same
#|     table, and the study it belongs to is carried by the prefix. A B_ prefix
#|     sends the search to the Study B folders first, an A_ prefix or no prefix to
#|     Study A's.
#|     """
#|     stem = str(stem)
#|     if stem.endswith(".csv"):
#|         stem = stem[: -len(".csv")]
#|     study = "A"
#|     for pre in ("studyA_", "studyB_", "A_", "B_"):
#|         if stem.startswith(pre):
#|             study = "B" if "B" in pre else "A"
#|             stem = stem[len(pre):]
#|             break
#|     names = [f"{stem}.csv", f"{study}_{stem}.csv", f"study{study}_{stem}.csv"]
#|     home = STUDY_B if study == "B" else STUDY_A
#|     folders = [DERIVED, home / "tables", home, PAPER / "derived"]
#|     # the other study, last, because a handful of tables were written once into
#|     # whichever folder the script that made them happened to be run from
#|     other = STUDY_A if study == "B" else STUDY_B
#|     folders += [other / "tables", other]
#|     for folder in folders:
#|         for n in names:
#|             yield folder / n
#|
#|
#| def load_table(stem: str, **kw):
#|     """
#|     Read a CSV by its bare name, from wherever in this project it lives.
#|
#|         load_table("bracket_error")        # any of the three spellings above
#|
#|     Raises with the full list of places tried, so a missing table is a question
#|     about which script has not been run rather than a stack trace.
#|     """
#|     import pandas as pd
#|     tried = []
#|     for p in table_candidates(stem):
#|         if p in tried:
#|             continue
#|         tried.append(p)
#|         if p.exists():
#|             return pd.read_csv(p, **kw)
#|     raise SystemExit(
#|         f"No table found for {stem!r}. Looked in:\n"
#|         + "\n".join(f"    {p}" for p in tried)
#|         + "\n    If the first of those is the one you expect, the script that\n"
#|           "    writes it has not been run yet; RUN_ORDER.md says which.")
#|
#|
#| # ---------------------------------------------------------------------------
#| #  Cross-validated predictions, checkpointed per fold
#| # ---------------------------------------------------------------------------
#| #  Several scripts fit the production model once per fold, which is minutes per
#| #  fold. Writing each fold's out-of-fold predictions as they are produced lets
#| #  an interrupted run resume instead of restarting, and lets a long battery be
#| #  spread over several sessions. The tag names the experiment and must be
#| #  unique: two experiments sharing a tag would read each other's folds.
#|
#| # ---------------------------------------------------------------------------
#| #  Grouped folds that are the same on every machine
#| # ---------------------------------------------------------------------------
#| #  scikit-learn's GroupKFold fills its folds by taking the groups in order of
#| #  size. This panel is balanced (every county has 17 rows), so every group
#| #  ties, and the order among ties comes from an unstable sort: it changes with
#| #  the scikit-learn and NumPy versions and with the processor. Every such
#| #  partition is a valid one, but they are not the same one. A fold prediction
#| #  stored under one partition and read back under another meets the wrong
#| #  rows, and the score computed from it is wrong without any error.
#| #
#| #  The fold of every row is therefore written once to derived/fold_maps/ and
#| #  read from there. Every script takes its grouped folds from grouped_splits,
#| #  so stored predictions always meet the rows they were made for, and scripts
#| #  that compare models "on the same folds" share one partition by construction.
#| #  The folder is always the paper's own derived/fold_maps, also when results
#| #  are sent elsewhere with SATSDOH_DERIVED or SATSDOH_RUN.
#| FOLD_MAPS = PAPER / "derived" / "fold_maps"
#|
#|
#| def fold_ids(groups, n_folds=None):
#|     """Fold number, 0 to n_folds - 1, of every row; read from the fold map of
#|     this grouping, or made once with GroupKFold and stored."""
#|     import hashlib
#|     import numpy as _np
#|     n_folds = n_folds or N_FOLDS
#|     _, codes = _np.unique(_np.asarray(groups), return_inverse=True)
#|     codes = _np.ascontiguousarray(codes.astype("int64"))
#|     key = hashlib.sha256(codes.tobytes()).hexdigest()[:12]
#|     f = FOLD_MAPS / (f"folds{n_folds}_rows{len(codes)}_"
#|                      f"groups{int(codes.max()) + 1}_{key}.npy")
#|     if f.exists():
#|         return _np.load(f)
#|     from sklearn.model_selection import GroupKFold
#|     ids = _np.full(len(codes), -1, dtype="int16")
#|     for k, (_, te) in enumerate(GroupKFold(n_splits=n_folds).split(codes, groups=codes)):
#|         ids[te] = k
#|     FOLD_MAPS.mkdir(parents=True, exist_ok=True)
#|     _np.save(f, ids)
#|     print(f"      [fold map written] {f.name}", flush=True)
#|     return ids
#|
#|
#| def grouped_splits(groups, n_folds=None):
#|     """The (train rows, test rows) of grouped K-fold cross-validation, in fold
#|     order, from the stored fold map of this grouping."""
#|     import numpy as _np
#|     n_folds = n_folds or N_FOLDS
#|     ids = fold_ids(groups, n_folds)
#|     return [(_np.flatnonzero(ids != k), _np.flatnonzero(ids == k))
#|             for k in range(n_folds)]
#|
#|
#| def load_xgb(model, path):
#|     """Read a saved model into `model` (an XGBRegressor) and return it.
#|
#|     XGBoost 3.0 cannot load a saved regressor under scikit-learn 1.8 or later:
#|     its type check raises "TypeError: `_estimator_type` undefined". The trees
#|     are then read through the booster, which gives the same model."""
#|     import xgboost as xgb
#|     try:
#|         model.load_model(str(path))
#|     except TypeError:
#|         booster = xgb.Booster()
#|         booster.load_model(str(path))
#|         booster.set_attr(scikit_learn=None)
#|         model._Booster = booster
#|         try:
#|             model._load_model_attributes(json.loads(booster.save_config()))
#|         except Exception:
#|             pass
#|     return model
#|
#|
#| def fitted_with(store):
#|     """The library versions that fitted the checkpoints in `store`, or {}."""
#|     f = Path(store) / "fitted_with.json"
#|     return json.load(open(f)) if f.exists() else {}
#|
#|
#| def cv_oof(X, y, groups, tag, n_folds=None, params=None, verbose=True):
#|     """Out-of-fold predictions over grouped folds, resuming from checkpoints."""
#|     import gc
#|     import numpy as _np
#|     import xgboost as xgb
#|
#|     n_folds = n_folds or N_FOLDS
#|     # The fold count is part of the checkpoint's identity. Without it, a smoke
#|     # test at three folds writes fold01 and fold02 and a later production run
#|     # at eight folds reads them as its own first two folds, which are
#|     # different splits entirely. The production fold count keeps the plain
#|     # path so checkpoints already on disk are still found.
#|     store = DERIVED / "checkpoints_cv" / (
#|         tag if n_folds == N_FOLDS else f"{tag}_f{n_folds}")
#|     store.mkdir(parents=True, exist_ok=True)
#|     oof = _np.full(len(y), _np.nan)
#|     # The store remembers which rows it was fitted for. Reading it against
#|     # other rows (another subset of counties, say) stops here instead of
#|     # returning a wrong score.
#|     import hashlib
#|     mark = hashlib.sha256(_np.ascontiguousarray(
#|         _np.asarray(groups).astype("int64")).tobytes()).hexdigest()[:16]
#|     rows_file = store / "rows.json"
#|     if rows_file.exists():
#|         if json.load(open(rows_file)).get("groups_sha") != mark:
#|             raise SystemExit(
#|                 f"{tag}: the stored folds were fitted for a different set of "
#|                 f"rows. Run the step whole and in its usual order, or move "
#|                 f"{store} aside to refit.")
#|     elif (not any(store.glob("fold*.npy"))
#|           or os.environ.get("SATSDOH_RECORD_ROWS") == "1"):
#|         json.dump(dict(rows=int(len(y)), groups_sha=mark),
#|                   open(rows_file, "w"), indent=1)
#|     for i, (tr, te) in enumerate(grouped_splits(groups, n_folds), 1):
#|         f = store / f"fold{i:02d}.npy"
#|         if f.exists():
#|             got = _np.load(f)
#|             if len(got) != len(te):
#|                 raise SystemExit(
#|                     f"{tag}: the stored fold {i} holds {len(got)} predictions "
#|                     f"but the fold has {len(te)} rows. The checkpoint was "
#|                     f"made for another partition; move {store} aside to refit.")
#|             oof[te] = got
#|             if verbose:
#|                 print(f"      {tag:<28} fold {i}/{n_folds}  from checkpoint",
#|                       flush=True)
#|             continue
#|         m = xgb.XGBRegressor(**(params or XGB_PARAMS), n_jobs=threads())
#|         m.fit(X.iloc[tr], y[tr])
#|         pred = m.predict(X.iloc[te])
#|         _np.save(f, pred)
#|         if not (store / "fitted_with.json").exists():
#|             json.dump(versions(), open(store / "fitted_with.json", "w"), indent=1)
#|         oof[te] = pred
#|         del m
#|         gc.collect()
#|         if verbose:
#|             print(f"      {tag:<28} fold {i}/{n_folds}  fitted", flush=True)
#|     if _np.isnan(oof).any():
#|         raise SystemExit(f"{tag}: some rows have no out-of-fold prediction")
#|     return oof
#|
#|
#| def cv_score(X, y, groups, tag, **kw):
#|     """R2 and MAE from cv_oof, with the column count."""
#|     from sklearn.metrics import r2_score, mean_absolute_error
#|     oof = cv_oof(X, y, groups, tag, **kw)
#|     return dict(label=tag, n_features=int(X.shape[1]),
#|                 r2=float(r2_score(y, oof)),
#|                 mae=float(mean_absolute_error(y, oof)))
#|
#|
#| def dependence_table():
#|     """
#|     (table, feature names) for script 05's attribution surface.
#|
#|     The table is WIDE: one column per candidate holding the predictor value
#|     and one more named `shap_<candidate>` holding that row's attribution. Two
#|     later scripts assumed a long table with feature/x/shap columns and failed
#|     on the first read; the idiom is written down once here so a third does not.
#|
#|         d, feats = C.dependence_table()
#|         x = d[f].to_numpy(float)
#|         v = d["shap_" + f].to_numpy(float)
#|     """
#|     d = load_table("shap_dependence_candidates")
#|     feats = [c for c in d.columns if "shap_" + c in d.columns]
#|     if not feats:
#|         raise SystemExit(
#|             "shap_dependence_candidates.csv carries no candidate paired with "
#|             "a shap_ column.\n    Columns: " + ", ".join(map(str, d.columns)))
#|     return d, feats
#|
#|
#| def census_pairs(columns=None, census=None):
#|     """
#|     The (field, central column, non-central column) triples the census compares.
#|
#|     Several studies ask the census's question with a different measuring
#|     instrument -- mutual information, a second learner, an era at a time -- and
#|     every one of them needs the same list of pairs. Built once here so two
#|     scripts cannot disagree about which pairs they are, which has happened:
#|     an earlier version of one check used each field's OVERALL best statistic as
#|     its non-central one, so for the fields whose best statistic IS central it
#|     compared a column with itself and recorded a tie.
#|
#|     `columns` restricts the result to pairs whose both members are present.
#|     """
#|     census = load_table("summary_statistic_census") if census is None else census
#|     have = set(columns) if columns is not None else None
#|     out = []
#|     for _, r in census.iterrows():
#|         c_stat = str(r.get("central_stat") or "").strip()
#|         n_stat = str(r.get("noncentral_stat") or "").strip()
#|         if not c_stat or not n_stat or "nan" in (c_stat.lower(), n_stat.lower()):
#|             continue
#|         field = str(r["field"])
#|         c_col, n_col = f"{field}_{c_stat}", f"{field}_{n_stat}"
#|         if c_col == n_col:
#|             continue
#|         if have is not None and (c_col not in have or n_col not in have):
#|             continue
#|         out.append((field, c_col, n_col))
#|     return out
#|
#|
#| # The library versions that produced the numbers in the paper (11 to 13
#| # October 2026; environment.yml pins them). Another version can change a refit
#| # in its last digits, and scikit-learn's own grouped folds from 1.8 on.
#| PAPER_VERSIONS = {"numpy": "2.2.6", "pandas": "2.2.3", "scipy": "1.16.1",
#|                   "sklearn": "1.7.1", "xgboost": "3.0.2", "shap": "0.48.0",
#|                   "catboost": "1.2.8"}
#|
#|
#| def version_notes() -> list:
#|     """One line for every installed library that differs from the paper's."""
#|     have = versions()
#|     return [f"{k} {have.get(k)} installed, the paper used {v}"
#|             for k, v in PAPER_VERSIONS.items()
#|             if have.get(k) not in (v, "absent", None)]
#|
#|
#| def banner(title: str) -> None:
#|     print("=" * 74)
#|     print(f"  {title}")
#|     print("=" * 74)
#|     print(f"  root     {ROOT}")
#|     print(f"  study A  {STUDY_A.name}")
#|     print(f"  seed {SEED}   folds {N_FOLDS}   threads {threads()}")
#|     print(f"  run      {RUN_LABEL}   "
#|           f"(results to {DERIVED.name}/, figures to {FIGURES.name}/)")
#|     for line in version_notes():
#|         print(f"  note     {line}")
#|     print("-" * 74)
#|
#|
#| def load_panel(columns=None):
#|     """
#|     The modelling panel, exactly as every script needs it: the model's own
#|     feature list, its median fill values, its year window, and the rows with an
#|     outcome. Written once here so no two scripts can build it differently.
#|     """
#|     import joblib
#|     import pyarrow.parquet as pq
#|     require(FRAME, MODEL_A)
#|     d = joblib.load(MODEL_A)
#|     feats, fill = d["features"], d["median_fill"]
#|     lo, hi = d["window"]
#|     cols = (columns or feats) + [GROUP, YEAR, TARGET]
#|     df = pq.read_table(FRAME, columns=sorted(set(cols))).to_pandas()
#|     df = df[(df[YEAR] >= lo) & (df[YEAR] <= hi)
#|             & df[TARGET].notna()].reset_index(drop=True)
#|     X = df[feats].copy() if columns is None else None
#|     if X is not None:
#|         for c, v in fill.items():
#|             if c in X.columns:
#|                 X[c] = X[c].fillna(v)
#|         X = X.astype("float32")
#|     return dict(df=df, X=X, features=feats, window=(lo, hi),
#|                 y=df[TARGET].to_numpy(), groups=df[GROUP].to_numpy())
#|
#|
#| if __name__ == "__main__":
#|     banner("paths and inputs")
#|     for name, p in _INPUTS.items():
#|         mark = "ok     " if Path(p).exists() else "MISSING"
#|         print(f"  {mark}  {name:<26} {p}")
#|     print("-" * 74)
#|     for name, p in (("FIGURES", FIGURES), ("TABLES", TABLES),
#|                     ("DERIVED", DERIVED)):
#|         print(f"  {'ok     ' if p.exists() else 'will be created'}  "
#|               f"{name:<26} {p}")
#|     print("-" * 74)
#|     v = versions()
#|     missing = [k for k, w in v.items() if w == "absent"]
#|     print("  libraries:", ", ".join(f"{k} {w}" for k, w in v.items()
#|                                     if w != "absent"))
#|     if missing:
#|         print("  NOT INSTALLED:", ", ".join(missing))
#|         print("\n  install what you need with:")
#|         print("    pip install " + " ".join(
#|             {"sklearn": "scikit-learn"}.get(m, m) for m in missing))
#|
#@@ END analysis/_config.py
#@@ FILE analysis/01_check_environment_and_inputs.py
#| """
#| 01_check_environment_and_inputs.py
#| Run this first. It does no analysis; it tells you whether the rest will work.
#|
#| Checks, in order:
#|   1  the project root was found, and the two results folders are under it
#|   2  every input file the other scripts read exists and is readable
#|   3  every library they import is installed, with its version
#|   4  the reference frame really contains the columns the paper names
#|   5  XGBoost gives the same answer twice, on this machine
#|
#| WHAT IT WRITES
#|   derived/manifests/01_check_environment_and_inputs.manifest.json
#|
#| TIME
#|   under a minute, except step 5 which takes two to three.
#|
#|     python3 01_check_environment_and_inputs.py
#|     python3 01_check_environment_and_inputs.py --skip-determinism
#| """
#| from __future__ import annotations
#|
#| # ===========================================================================
#| #  SETTINGS
#| # ===========================================================================
#| DETERMINISM_TREES = 120       # small model; enough to expose a thread effect
#| DETERMINISM_THREADS = (1, 4)  # fit at each and compare bit for bit
#| # ===========================================================================
#|
#| import argparse
#| import sys
#|
#| import _config as C
#|
#|
#| REQUIRED_LIBS = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "shap",
#|                  "pyarrow", "matplotlib", "joblib"]
#| OPTIONAL_LIBS = ["catboost", "geopandas"]
#|
#| # columns the manuscript names by hand; if one is missing the paper is wrong
#| NAMED_COLUMNS = [
#|     C.TARGET, C.GROUP, C.YEAR,
#|     C.NCE_TAIL, C.NCE_MEAN,
#|     "lst_day_1km_p90", "lst_night_1km_p10",
#|     "lst_day_1km_mean", "lst_night_1km_mean",
#|     "cams_fot_formaldehyde_above75th_percentile",
#|     "prism_tdmean_min", "pm25_mean",
#| ]
#|
#|
#| def check_inputs() -> list[str]:
#|     problems = []
#|     print("INPUT FILES")
#|     for name, p in C._INPUTS.items():
#|         if p.exists():
#|             size = p.stat().st_size
#|             print(f"  ok       {name:<26} {size/1e6:8.1f} MB")
#|         else:
#|             print(f"  MISSING  {name:<26} {p}")
#|             problems.append(f"missing input: {p}")
#|     return problems
#|
#|
#| def check_libraries() -> list[str]:
#|     problems = []
#|     print("\nLIBRARIES")
#|     v = C.versions()
#|     for m in REQUIRED_LIBS:
#|         key = m
#|         if v.get(key, "absent") == "absent":
#|             print(f"  MISSING  {m}")
#|             problems.append(f"required library not installed: {m}")
#|         else:
#|             print(f"  ok       {m:<14} {v[key]}")
#|     for m in OPTIONAL_LIBS:
#|         state = v.get(m, "absent")
#|         print(f"  {'ok      ' if state != 'absent' else 'optional'} "
#|               f"{m:<14} {state}")
#|     if problems:
#|         names = [p.split(": ")[1] for p in problems]
#|         print("\n  install with:")
#|         print("    pip install " + " ".join(
#|             {"sklearn": "scikit-learn"}.get(m, m) for m in names))
#|     return problems
#|
#|
#| def check_columns() -> list[str]:
#|     import pyarrow.parquet as pq
#|     problems = []
#|     print("\nCOLUMNS THE MANUSCRIPT NAMES")
#|     if not C.FRAME.exists():
#|         print("  skipped, the reference frame is missing")
#|         return ["reference frame missing, columns not checked"]
#|     have = set(pq.ParquetFile(C.FRAME).schema.names)
#|     for c in NAMED_COLUMNS:
#|         if c in have:
#|             print(f"  ok       {c}")
#|         else:
#|             print(f"  MISSING  {c}")
#|             problems.append(f"column named in the paper is not in the frame: {c}")
#|     print(f"  frame holds {len(have):,} columns in total")
#|     return problems
#|
#|
#| def check_determinism() -> list[str]:
#|     """
#|     The question behind this check: if you run an analysis twice, do you get
#|     the same number? A figure and a sentence that disagree because of a silent
#|     rerun is the worst failure this project can have.
#|
#|     XGBoost's 'hist' tree method is documented as deterministic for a fixed
#|     seed, and this confirms it on THIS machine, including across thread
#|     counts, which is the case a laptop and a cluster differ in.
#|     """
#|     import numpy as np
#|     import xgboost as xgb
#|     from sklearn.model_selection import GroupKFold
#|
#|     print("\nDETERMINISM")
#|     p = C.load_panel()
#|     X, y, g = p["X"], p["y"], p["groups"]
#|     tr, te = C.grouped_splits(g)[0]
#|     kw = dict(C.XGB_PARAMS)
#|     kw["n_estimators"] = DETERMINISM_TREES
#|
#|     sums = {}
#|     for nj in DETERMINISM_THREADS:
#|         m = xgb.XGBRegressor(n_jobs=nj, **kw).fit(X.iloc[tr], y[tr])
#|         pred = m.predict(X.iloc[te]).astype(np.float64)
#|         sums[f"{nj} threads"] = float(pred.sum())
#|         print(f"  fitted at {nj:>2} thread(s): prediction sum "
#|               f"{pred.sum():.9f}")
#|     # same threads twice
#|     nj = DETERMINISM_THREADS[-1]
#|     m = xgb.XGBRegressor(n_jobs=nj, **kw).fit(X.iloc[tr], y[tr])
#|     again = float(m.predict(X.iloc[te]).astype(np.float64).sum())
#|     sums[f"{nj} threads, rerun"] = again
#|     print(f"  refitted at {nj:>2} thread(s): prediction sum {again:.9f}")
#|
#|     vals = {round(v, 6) for v in sums.values()}
#|     problems = []
#|     if len(vals) == 1:
#|         print("\n  PASS  identical across thread counts and across reruns.")
#|         print("        A number produced by these scripts will reproduce.")
#|     else:
#|         print("\n  FAIL  the fits disagree. Do not trust a rerun to reproduce")
#|         print("        a published number until this is understood.")
#|         problems.append("XGBoost is not reproducible on this machine")
#|     return problems
#|
#|
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--skip-determinism", action="store_true")
#|     a = ap.parse_args()
#|
#|     C.banner("01  environment and inputs")
#|     problems = check_inputs() + check_libraries() + check_columns()
#|     if not a.skip_determinism and not problems:
#|         problems += check_determinism()
#|     elif problems:
#|         print("\nDETERMINISM\n  skipped, because something above is missing")
#|
#|     print("\n" + "=" * 74)
#|     if problems:
#|         print(f"  {len(problems)} problem(s) to fix before running anything else:")
#|         for p in problems:
#|             print(f"    - {p}")
#|         print("=" * 74)
#|         C.write_manifest(__file__, list(C._INPUTS.values()), [],
#|                          notes=f"{len(problems)} problems: {problems}")
#|         sys.exit(1)
#|     print("  everything the other scripts need is present and reproducible.")
#|     print("  next:  python3 02_outcome_distribution_and_error_brackets.py")
#|     print("=" * 74)
#|     C.write_manifest(__file__, list(C._INPUTS.values()), [], notes="clean")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/01_check_environment_and_inputs.py
#@@ FILE analysis/02_outcome_distribution_and_error_brackets.py
#| """
#| 02_outcome_distribution_and_error_brackets.py
#| Where the outcome sits, and where the model's error sits inside it.
#|
#| WHY THIS EXISTS
#| ---------------
#| Figure 2C reports a mean absolute error of about 6.6 years in the lowest
#| life-expectancy bracket, on n = 153 county-years, and 153 looks too small to
#| be right. This script recomputes the brackets from the out-of-fold predictions
#| so the number can be checked rather than believed, and prints the whole
#| outcome distribution around it.
#|
#| The short answer, which the output below will show you: the bracket is
#| labelled in YEARS OF LIFE EXPECTANCY AT BIRTH, not in age. The lowest observed
#| county-year in the panel is about 65 years, so the interval [60, 70) contains
#| only the small number of county-years whose life expectancy is under seventy.
#| There are 22 such counties. They are the reservation and central-Appalachian
#| counties that also dominate the residual map, which is a useful consistency
#| check rather than a coincidence.
#|
#| The script also warns when a bracket is too thin to support a stable mean, so
#| that the bracket edges can be reconsidered rather than defended.
#|
#| READS   studyA_oof_predictions.csv
#| WRITES  derived/outcome_distribution.csv
#|         derived/bracket_error.csv        (the figure's input, recomputed)
#|         derived/bracket_error_check.json
#| TIME    seconds
#|
#|     python3 02_outcome_distribution_and_error_brackets.py
#| """
#| from __future__ import annotations
#|
#| # ===========================================================================
#| #  SETTINGS
#| # ===========================================================================
#| # The bracket edges the manuscript uses. Left-closed, right-open.
#| BRACKET_EDGES = [60, 70, 74, 76.5, 79, 95]
#|
#| # A bracket holding fewer than this many county-years gets a warning printed,
#| # because a mean absolute error on a handful of rows is not a stable quantity.
#| THIN_BRACKET = 500
#|
#| # Also report equal-count brackets, as an alternative the paper could use.
#| ALSO_REPORT_QUINTILES = True
#| # ===========================================================================
#|
#| import json
#|
#| import numpy as np
#| import pandas as pd
#|
#| import _config as C
#|
#| FIPS_NAMES = {
#|     46102: "Oglala Lakota, SD", 46121: "Todd, SD", 46017: "Buffalo, SD",
#|     38085: "Sioux, ND", 38079: "Rolette, ND", 30085: "Roosevelt, MT",
#|     30035: "Glacier, MT", 54047: "McDowell, WV", 21189: "Owsley, KY",
#|     12125: "Union, FL", 51690: "Martinsville, VA", 51730: "Petersburg, VA",
#| }
#|
#|
#| def main():
#|     C.banner("02  outcome distribution and error brackets")
#|     C.require(C.OOF_A)
#|     d = pd.read_csv(C.OOF_A)
#|     y = d[C.TARGET].to_numpy(float)
#|     err = np.abs(d["resid"].to_numpy(float))
#|
#|     print(f"county-years            {len(d):,}")
#|     print(f"counties                {d[C.GROUP].nunique():,}")
#|     print(f"years                   {d[C.YEAR].min()}-{d[C.YEAR].max()}")
#|     print()
#|     print("LIFE EXPECTANCY AT BIRTH, YEARS  (this is the outcome, not an age)")
#|     q = np.percentile(y, [0, 1, 5, 25, 50, 75, 95, 99, 100])
#|     for lab, v in zip(["min", "p1", "p5", "p25", "median", "p75", "p95",
#|                        "p99", "max"], q):
#|         print(f"  {lab:<8}{v:8.2f}")
#|     print()
#|     print(f"  county-years below 70 years : {int((y < 70).sum()):>6,}"
#|           f"   ({100*(y<70).mean():.2f} % of the panel)")
#|     print(f"  county-years below 72 years : {int((y < 72).sum()):>6,}")
#|     print(f"  county-years below 74 years : {int((y < 74).sum()):>6,}")
#|     print(f"  distinct counties below 70  : "
#|           f"{d.loc[y < 70, C.GROUP].nunique():>6,}")
#|
#|     sub = d[y < 70]
#|     if len(sub):
#|         print("\n  the counties that make up the lowest bracket:")
#|         cnt = sub.groupby(C.GROUP).size().sort_values(ascending=False)
#|         for fips, n in cnt.items():
#|             nm = FIPS_NAMES.get(int(fips), "")
#|             print(f"    {int(fips):>5}  {n:>3} year(s)   {nm}")
#|
#|     # -- the brackets the figure uses, recomputed -----------------------
#|     print("\nERROR BY BRACKET  (recomputed from the out-of-fold predictions)")
#|     lab = []
#|     for lo, hi in zip(BRACKET_EDGES[:-1], BRACKET_EDGES[1:]):
#|         lab.append(f"{lo:g}-{hi:g}")
#|     cut = pd.cut(y, BRACKET_EDGES, right=False, labels=lab)
#|     g = pd.DataFrame({"bracket": cut, "err": err, "y": y,
#|                       "resid": d["resid"]}).groupby("bracket", observed=False)
#|     br = pd.DataFrame({
#|         "bracket": lab,
#|         "n": g.size().reindex(lab).to_numpy(),
#|         "mae": g.err.mean().reindex(lab).to_numpy(),
#|         "mean_resid": g.resid.mean().reindex(lab).to_numpy(),
#|     })
#|     # a standard error on the bracket mean, so a thin bracket declares itself
#|     br["mae_se"] = (g.err.std().reindex(lab).to_numpy()
#|                     / np.sqrt(br.n.to_numpy()))
#|     print(br.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
#|
#|     thin = br[br.n < THIN_BRACKET]
#|     warnings = []
#|     if len(thin):
#|         for _, r in thin.iterrows():
#|             msg = (f"bracket {r.bracket} holds {int(r.n)} county-years; its "
#|                    f"MAE of {r.mae:.2f} carries a standard error of "
#|                    f"{r.mae_se:.2f}")
#|             warnings.append(msg)
#|             print(f"\n  NOTE  {msg}")
#|         print("        The value is correct, and it rests on few observations.")
#|         print("        Figure 2C prints each bracket's count for this reason.")
#|
#|     # -- compare against the stored file the figure actually reads ------
#|     stored_path = C.STUDY_A / "studyA_bracket_error.csv"
#|     agree = None
#|     if stored_path.exists():
#|         st = pd.read_csv(stored_path)
#|         merged = br.merge(st, on="bracket", suffixes=("_new", "_stored"),
#|                           how="outer")
#|         dn = np.abs(merged.n_new - merged.n_stored).max()
#|         dm = np.abs(merged.mae_new - merged.mae_stored).max()
#|         agree = bool(dn == 0 and dm < 1e-6)
#|         print(f"\nAGAINST THE STORED FILE ({stored_path.name})")
#|         print(f"  max difference in count  {dn:.0f}")
#|         print(f"  max difference in MAE    {dm:.2e}")
#|         print(f"  {'reproduces exactly' if agree else 'DOES NOT REPRODUCE'}")
#|
#|     # -- an equal-count alternative, for comparison ---------------------
#|     if ALSO_REPORT_QUINTILES:
#|         print("\nEQUAL-COUNT ALTERNATIVE  (five brackets of equal size)")
#|         qc = pd.qcut(y, 5)
#|         qg = pd.DataFrame({"b": qc, "err": err}).groupby("b", observed=True)
#|         alt = pd.DataFrame({"bracket": [str(i) for i in qg.size().index],
#|                             "n": qg.size().to_numpy(),
#|                             "mae": qg.err.mean().to_numpy()})
#|         print(alt.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
#|         print("  Equal-count brackets remove the thin-bracket problem but hide")
#|         print("  the extreme tail, which is the point Figure 2C is making.")
#|
#|     C.DERIVED.mkdir(parents=True, exist_ok=True)
#|     out1 = C.DERIVED / "outcome_distribution.csv"
#|     pd.DataFrame({"percentile": [0, 1, 5, 25, 50, 75, 95, 99, 100],
#|                   "life_expectancy_years": q}).to_csv(out1, index=False)
#|     out2 = C.DERIVED / "bracket_error.csv"
#|     br.to_csv(out2, index=False)
#|     out3 = C.DERIVED / "bracket_error_check.json"
#|     json.dump(dict(n_rows=int(len(d)),
#|                    min_life_expectancy=float(y.min()),
#|                    max_life_expectancy=float(y.max()),
#|                    n_below_70=int((y < 70).sum()),
#|                    counties_below_70=int(d.loc[y < 70, C.GROUP].nunique()),
#|                    brackets=br.to_dict("records"),
#|                    reproduces_stored_file=agree,
#|                    thin_bracket_warnings=warnings),
#|               open(out3, "w"), indent=1)
#|     print(f"\n[saved] {out1.name}, {out2.name}, {out3.name}  in derived/")
#|     C.write_manifest(__file__, [C.OOF_A], [out1, out2, out3],
#|                      settings=dict(BRACKET_EDGES=BRACKET_EDGES),
#|                      notes="; ".join(warnings))
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/02_outcome_distribution_and_error_brackets.py
#@@ FILE analysis/03_baselines_five_reference_models.py
#| """
#| 03_baselines_five_reference_models.py
#| The floor the environment-only model has to clear, computed from scratch.
#|
#| WHY THIS EXISTS
#| ---------------
#| An R-squared means nothing on its own. Figure 2D reports five reference
#| predictors scored on the identical county-grouped folds, and the one that
#| matters is the fourth: an interpolator that is handed the measured life
#| expectancy of the four nearest TRAINING counties, which the model never sees.
#| If the model cannot beat that, it is a slow way of drawing a map.
#|
#| THE FIVE
#|   1  training-fold mean                 no features at all
#|   2  training-fold mean for that year   no features, but allowed the trend
#|   3  night-time lights only             the usual single-product stand-in
#|   4  latitude, longitude and year       position, measuring nothing
#|   5  four-nearest-neighbour             the outcome itself, next door
#|
#| Every one uses the same GroupKFold(n_splits=8) partition on FIPS as the main
#| fit, so the comparison is paired rather than approximate.
#|
#| COUNTY CENTROIDS
#|   Baselines 4 and 5 need a latitude and longitude per county. The script looks
#|   for a cached copy first, and only then tries the Census gazetteer over the
#|   network. If it has no cache and no network it says so and skips those two
#|   rather than failing the whole run; run it again once you are online and the
#|   cache will be written for good.
#|
#| READS   the reference frame, the fitted model
#| WRITES  derived/baselines.json
#|         derived/baselines.csv
#|         derived/county_centroids.csv   (the cache, written once)
#| TIME    15 to 40 minutes on a laptop. Baselines 3 and 4 fit a model each; the
#|         rest are arithmetic. Pass --only to run one at a time.
#|
#|     python3 03_baselines_five_reference_models.py
#|     python3 03_baselines_five_reference_models.py --only knn,coords
#| """
#| from __future__ import annotations
#|
#| # ===========================================================================
#| #  SETTINGS
#| # ===========================================================================
#| KNN_NEIGHBOURS = 4        # how many training counties the interpolator sees
#| KNN_SAME_YEAR = True      # borrow only from the same year, as the paper says
#| NTL_PREFIX = "ntl_"       # how night-time-lights columns are named
#| CENTROID_URL = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
#|                 "2020_Gazetteer/2020_Gaz_counties_national.zip")
#| # ===========================================================================
#|
#| import argparse
#| import io
#| import json
#| import time
#| import urllib.request
#| import zipfile
#|
#| import numpy as np
#| import pandas as pd
#|
#| import _config as C
#|
#| ORDER = ["naive_mean", "naive_year_mean", "ntl_only", "coords_only",
#|          "knn4_spatial"]
#| LABEL = {
#|     "naive_mean": ("Training-fold mean", "no features"),
#|     "naive_year_mean": ("Training-fold mean for that year", "no features"),
#|     "ntl_only": ("Night-time lights only", "{n} features"),
#|     "coords_only": ("Latitude, longitude, year", "3 features, not observations"),
#|     "knn4_spatial": ("Four-nearest-neighbour interpolation",
#|                      "needs the outcome next door"),
#| }
#|
#|
#| def folds(X, y, g):
#|     from sklearn.model_selection import GroupKFold
#|     return C.grouped_splits(g)
#|
#|
#| def cv_fit_score(X, y, g, tag):
#|     """Fit the production learner on the production folds and score OOF."""
#|     import xgboost as xgb
#|     from sklearn.metrics import mean_absolute_error, r2_score
#|     oof = np.full(len(y), np.nan)
#|     t0 = time.time()
#|     for k, (tr, te) in enumerate(folds(X, y, g), 1):
#|         m = xgb.XGBRegressor(n_jobs=C.threads(), **C.XGB_PARAMS)
#|         m.fit(X.iloc[tr], y[tr])
#|         oof[te] = m.predict(X.iloc[te])
#|         print(f"    [{tag}] fold {k}/{C.N_FOLDS}  {time.time()-t0:6.0f}s",
#|               flush=True)
#|     return dict(r2=float(r2_score(y, oof)),
#|                 mae=float(mean_absolute_error(y, oof)),
#|                 n_features=int(X.shape[1]), seconds=round(time.time() - t0))
#|
#|
#| def centroids():
#|     """Cached first, network second, honest failure third."""
#|     cache = C.DERIVED / "county_centroids.csv"
#|     if cache.exists():
#|         print(f"  centroids from cache: {cache.name}")
#|         return pd.read_csv(cache)
#|     print("  no centroid cache; fetching the Census gazetteer ...")
#|     try:
#|         with urllib.request.urlopen(CENTROID_URL, timeout=60) as r:
#|             blob = r.read()
#|         with zipfile.ZipFile(io.BytesIO(blob)) as z:
#|             name = [n for n in z.namelist() if n.endswith(".txt")][0]
#|             raw = pd.read_csv(z.open(name), sep="\t", dtype={"GEOID": str})
#|     except Exception as e:
#|         print(f"  could not fetch centroids: {type(e).__name__}: {e}")
#|         return None
#|     raw.columns = [c.strip() for c in raw.columns]
#|     df = pd.DataFrame({"fips": raw["GEOID"].astype(int),
#|                        "lat": raw["INTPTLAT"].astype(float),
#|                        "lon": raw["INTPTLONG"].astype(float)})
#|     C.DERIVED.mkdir(parents=True, exist_ok=True)
#|     df.to_csv(cache, index=False)
#|     print(f"  centroids cached to {cache.name}; this will not download again")
#|     return df
#|
#|
#| # ---------------------------------------------------------------- baselines
#| def b_naive(p, R):
#|     from sklearn.metrics import mean_absolute_error, r2_score
#|     X, y, g, df = p["X"], p["y"], p["groups"], p["df"]
#|     oof = np.full(len(y), np.nan)
#|     for tr, te in folds(X, y, g):
#|         oof[te] = y[tr].mean()
#|     R["naive_mean"] = dict(r2=float(r2_score(y, oof)),
#|                            mae=float(mean_absolute_error(y, oof)),
#|                            n_features=0)
#|     oof2 = np.full(len(y), np.nan)
#|     yr = df[C.YEAR].to_numpy()
#|     for tr, te in folds(X, y, g):
#|         mu = pd.Series(y[tr]).groupby(yr[tr]).mean()
#|         oof2[te] = [mu.get(v, y[tr].mean()) for v in yr[te]]
#|     R["naive_year_mean"] = dict(r2=float(r2_score(y, oof2)),
#|                                 mae=float(mean_absolute_error(y, oof2)),
#|                                 n_features=0)
#|
#|
#| def b_ntl(p, R):
#|     cols = [c for c in p["features"] if c.startswith(NTL_PREFIX)]
#|     if not cols:
#|         print(f"  no columns start with {NTL_PREFIX!r}; skipping")
#|         return
#|     print(f"  night-time lights: {len(cols)} columns")
#|     R["ntl_only"] = cv_fit_score(p["X"][cols], p["y"], p["groups"], "ntl")
#|
#|
#| def b_coords(p, R):
#|     cen = centroids()
#|     if cen is None:
#|         return
#|     d = p["df"][[C.GROUP, C.YEAR, C.TARGET]].merge(cen, on="fips", how="left")
#|     ok = d.lat.notna().to_numpy()
#|     X = d.loc[ok, ["lat", "lon", C.YEAR]].astype("float32").reset_index(drop=True)
#|     R["coords_only"] = cv_fit_score(X, d.loc[ok, C.TARGET].to_numpy(),
#|                                     d.loc[ok, C.GROUP].to_numpy(), "coords")
#|
#|
#| def b_knn(p, R):
#|     from sklearn.metrics import mean_absolute_error, r2_score
#|     from sklearn.neighbors import NearestNeighbors
#|     cen = centroids()
#|     if cen is None:
#|         return
#|     d = p["df"][[C.GROUP, C.YEAR, C.TARGET]].merge(cen, on="fips", how="left")
#|     d = d[d.lat.notna()].reset_index(drop=True)
#|     y, g = d[C.TARGET].to_numpy(), d[C.GROUP].to_numpy()
#|     yr = d[C.YEAR].to_numpy()
#|     P = np.radians(d[["lat", "lon"]].to_numpy())
#|     oof = np.full(len(d), np.nan)
#|     t0 = time.time()
#|     for k, (tr, te) in enumerate(folds(P, y, g), 1):
#|         years = np.unique(yr[te]) if KNN_SAME_YEAR else [None]
#|         for v in years:
#|             itr = tr[yr[tr] == v] if KNN_SAME_YEAR else tr
#|             ite = te[yr[te] == v] if KNN_SAME_YEAR else te
#|             if len(itr) < KNN_NEIGHBOURS + 1 or not len(ite):
#|                 continue
#|             nn = NearestNeighbors(n_neighbors=KNN_NEIGHBOURS,
#|                                   metric="haversine").fit(P[itr])
#|             dist, idx = nn.kneighbors(P[ite])
#|             w = 1.0 / np.maximum(dist, 1e-6)
#|             oof[ite] = (y[itr][idx] * w).sum(1) / w.sum(1)
#|         print(f"    [knn] fold {k}/{C.N_FOLDS}  {time.time()-t0:5.0f}s",
#|               flush=True)
#|     m = np.isfinite(oof)
#|     R["knn4_spatial"] = dict(r2=float(r2_score(y[m], oof[m])),
#|                              mae=float(mean_absolute_error(y[m], oof[m])),
#|                              n=int(m.sum()), n_features=0)
#|
#|
#| RUNNERS = {"naive": b_naive, "ntl": b_ntl, "coords": b_coords, "knn": b_knn}
#|
#|
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--only", default="all",
#|                     help="comma-separated: naive,ntl,coords,knn")
#|     a = ap.parse_args()
#|
#|     C.banner("03  five reference baselines")
#|     C.seed_everything()
#|     p = C.load_panel()
#|     print(f"panel: {len(p['df']):,} county-years, {p['X'].shape[1]} features\n")
#|
#|     out = C.DERIVED / "baselines.json"
#|     R = json.load(open(out)) if out.exists() else {}
#|     if R:
#|         print(f"resuming; already have: {', '.join(sorted(R))}\n")
#|
#|     picked = list(RUNNERS) if a.only == "all" else [s.strip() for s in
#|                                                     a.only.split(",")]
#|     for name in picked:
#|         print(f"[{name}]")
#|         RUNNERS[name](p, R)
#|         C.DERIVED.mkdir(parents=True, exist_ok=True)
#|         json.dump(R, open(out, "w"), indent=1)
#|         print()
#|
#|     # the model itself, for the same table
#|     try:
#|         met = json.load(open(C.SUMMARY_A))
#|         R["satellite_model"] = dict(r2=float(met.get("r2")),
#|                                     mae=float(met.get("mae")),
#|                                     n_features=int(met.get("n_features", 0)))
#|     except Exception:
#|         pass
#|
#|     rows = []
#|     for k in ORDER + ["satellite_model"]:
#|         if k not in R:
#|             continue
#|         nm, note = LABEL.get(k, ("Satellite model", "{n} features"))
#|         rows.append(dict(key=k, label=nm,
#|                          note=note.format(n=R[k].get("n_features", 0)),
#|                          r2=R[k]["r2"], mae=R[k]["mae"]))
#|     tab = pd.DataFrame(rows)
#|     print("=" * 74)
#|     print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
#|     print("=" * 74)
#|     if "knn4_spatial" in R and "satellite_model" in R:
#|         d = R["satellite_model"]["r2"] - R["knn4_spatial"]["r2"]
#|         print(f"  model minus interpolator: {d:+.4f} in R-squared")
#|         print("  The interpolator is given the outcome in neighbouring")
#|         print("  counties. The model is given no outcome anywhere.")
#|
#|     csv = C.DERIVED / "baselines.csv"
#|     tab.to_csv(csv, index=False)
#|     json.dump(R, open(out, "w"), indent=1)
#|     print(f"\n[saved] {out.name}, {csv.name}  in derived/")
#|     C.write_manifest(__file__, [C.FRAME, C.MODEL_A], [out, csv],
#|                      settings=dict(KNN_NEIGHBOURS=KNN_NEIGHBOURS,
#|                                    KNN_SAME_YEAR=KNN_SAME_YEAR),
#|                      notes=f"ran: {picked}")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/03_baselines_five_reference_models.py
#@@ FILE analysis/04_refit_drop_one_nce_construction.py
#| from __future__ import annotations
#| import argparse, json, time, numpy as np, pandas as pd
#| import _config as C
#|
#| ARMS = ("FULL", "DROP_P", "DROP_M")   
#| TOP_N = 10                            
#|
#| def run_arm(name, drop, p, a, ckdir):
#|     import shap, xgboost as xgb
#|     from sklearn.metrics import mean_absolute_error, r2_score
#|     from sklearn.model_selection import GroupKFold
#|
#|     cols = [f for f in p["features"] if f not in drop]
#|     X, y, g = p["X"][cols], p["y"], p["groups"]
#|     ck = ckdir / f"{name}_f{a.folds}.npz"
#|     oof = np.full(len(X), np.nan)
#|     done: set[int] = set()
#|     last = None
#|
#|     if ck.exists() and not a.fresh:
#|         z = np.load(ck, allow_pickle=True)
#|         oof, done = z["oof"], set(z["done"].tolist())
#|         print(f"  [{name}] resuming, {len(done)}/{a.folds} folds already fitted")
#|
#|     t0 = time.time()
#|     splits = C.grouped_splits(g, a.folds)
#|     for k, (tr, te) in enumerate(splits, 1):
#|         if k in done and k != a.folds: continue
#|         m = xgb.XGBRegressor(n_jobs=C.threads(), **C.XGB_PARAMS)
#|         m.fit(X.iloc[tr], y[tr])
#|         oof[te] = m.predict(X.iloc[te])
#|         last = m
#|         done.add(k)
#|         np.savez(ck, oof=oof, done=np.array(sorted(done)))
#|         print(f"  [{name}] fold {k}/{a.folds}  {time.time()-t0:7.0f}s  (checkpointed)", flush=True)
#|
#|     if last is None:
#|         tr, _ = splits[-1]
#|         last = xgb.XGBRegressor(n_jobs=C.threads(), **C.XGB_PARAMS).fit(X.iloc[tr], y[tr])
#|
#|     r2, mae = r2_score(y, oof), mean_absolute_error(y, oof)
#|     rng = C.seed_everything()
#|     samp = rng.choice(len(X), size=min(a.shap, len(X)), replace=False)
#|     bg = rng.choice(len(X), size=C.SHAP_BACKGROUND, replace=False)
#|     
#|     ts = time.time()
#|     ex = shap.TreeExplainer(last, data=X.iloc[bg], feature_perturbation="interventional")
#|     
#|     # FIX: Removed the silent=True argument causing the TypeError
#|     sv = np.abs(ex.shap_values(X.iloc[samp], check_additivity=False)).mean(0)
#|     print(f"  [{name}] interventional TreeSHAP on {len(samp):,} rows: {time.time()-ts:.0f}s", flush=True)
#|
#|     rank = pd.Series(sv, index=cols).sort_values(ascending=False)
#|     order = {f: i + 1 for i, f in enumerate(rank.index)}
#|     return dict(
#|         r2=float(r2), mae=float(mae), n_features=len(cols), seconds=round(time.time() - t0),
#|         pct_rank=order.get(C.NCE_TAIL), pct_shap=None if C.NCE_TAIL not in order else float(rank[C.NCE_TAIL]),
#|         mean_rank=order.get(C.NCE_MEAN), mean_shap=None if C.NCE_MEAN not in order else float(rank[C.NCE_MEAN]),
#|         top10=[(f, round(float(v), 4)) for f, v in rank.head(TOP_N).items()])
#|
#| def report(res):
#|     f, dp, dm = res.get("FULL"), res.get("DROP_P"), res.get("DROP_M")
#|     if not (f and dp): return
#|     rise = dp["mean_shap"] / f["mean_shap"]
#|     rec = 100 * (dp["mean_shap"] - f["mean_shap"]) / f["pct_shap"]
#|     print("\n" + "=" * 74)
#|     print("  WHAT THE MANUSCRIPT PRINTS")
#|     print("=" * 74)
#|     print(f"  tail construction, both present  rank {f['pct_rank']:>4}  mean|SHAP| {f['pct_shap']:.4f}")
#|     print(f"  central, both present            rank {f['mean_rank']:>4}  mean|SHAP| {f['mean_shap']:.5f}")
#|     print(f"  central, twin removed            rank {dp['mean_rank']:>4}  mean|SHAP| {dp['mean_shap']:.5f}")
#|     print(f"\n  rise {rise:.1f}-fold, recovering {rec:.1f} % of what the tail construction carried")
#|     print(f"  R2  both {f['r2']:.4f} | tail removed {dp['r2']:.4f} ({dp['r2']-f['r2']:+.4f})", end="")
#|     if dm:
#|         print(f" | central removed {dm['r2']:.4f} ({dm['r2']-f['r2']:+.4f})")
#|         print(f"  tail with the field to itself: rank {dm['pct_rank']}, mean|SHAP| {dm['pct_shap']:.4f}")
#|         if dm["r2"] >= f["r2"]: print("  Removing the weaker description did not cost accuracy.")
#|     print("=" * 74)
#|
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--folds", type=int, default=C.N_FOLDS)
#|     ap.add_argument("--shap", type=int, default=C.SHAP_SAMPLE)
#|     ap.add_argument("--fresh", action="store_true", help="ignore checkpoints and refit everything")
#|     a = ap.parse_args()
#|
#|     C.banner("04  refit with one night-cooling construction removed")
#|     print(f"  arms {ARMS}   shap sample {a.shap:,}\n")
#|     C.require(C.FRAME, C.MODEL_A)
#|     p = C.load_panel()
#|     
#|     ckdir = C.DERIVED / "checkpoints_refit"
#|     ckdir.mkdir(parents=True, exist_ok=True)
#|     out = C.DERIVED / "nce_refit.json"
#|     
#|     blob = json.load(open(out)) if (out.exists() and not a.fresh) else {}
#|     res = blob.get("conditions", {})
#|     if res: print(f"resuming; already complete: {', '.join(sorted(res))}\n")
#|
#|     drop_for = {"FULL": [], "DROP_P": [C.NCE_TAIL], "DROP_M": [C.NCE_MEAN]}
#|     protocol = dict(folds=a.folds, nshap=a.shap, nbg=C.SHAP_BACKGROUND, n_rows=int(len(p["X"])), n_features=len(p["features"]), window=list(p["window"]), learner="xgboost", threads=C.threads(), **C.XGB_PARAMS)
#|
#|     for name in ARMS:
#|         if name in res: continue
#|         what = "both present" if not drop_for[name] else f"without {drop_for[name][0]}"
#|         print(f"--- {name} ({what}) ---")
#|         res[name] = run_arm(name, drop_for[name], p, a, ckdir)
#|         json.dump(dict(protocol=protocol, conditions=res), open(out, "w"), indent=1)
#|         print(f"  [saved] {out.name}\n")
#|
#|     report(res)
#|     json.dump(dict(protocol=protocol, conditions=res), open(out, "w"), indent=1)
#|     C.write_manifest(__file__, [C.FRAME, C.MODEL_A], [out], settings=dict(folds=a.folds, shap_sample=a.shap, arms=list(ARMS)))
#|
#| if __name__ == "__main__": main()
#|
#@@ END analysis/04_refit_drop_one_nce_construction.py
#@@ FILE analysis/05_recompute_shap_for_threshold_candidates.py
#| from __future__ import annotations
#| import argparse, json, time, numpy as np, pandas as pd
#| import _config as C
#|
#| CANDIDATES = [
#|     "prism_tdmean_min", "eng2_night_cooling_efficiency", "eng_night_cooling_eff",
#|     "lst_night_1km_mean", "eng2_wetbulb_night", "eng2_wetbulb_hot",
#|     "eng2_hcho_no2_ratio", "pm25_mean", "prism_tmin_min",
#|     "cams_fot_formaldehyde_above75th_percentile", "usda_cropland_usda_corn_pct",
#| ]
#|
#| def _binned(x, y, nb=40):
#|     m = np.isfinite(x) & np.isfinite(y)
#|     x, y = x[m], y[m]
#|     if len(x) < nb * 5: return np.full(nb, np.nan)
#|     qs = np.quantile(x, np.linspace(0, 1, nb + 1)); qs[-1] += 1e-9
#|     out = np.full(nb, np.nan)
#|     for i in range(nb):
#|         s = (x >= qs[i]) & (x < qs[i + 1])
#|         if s.sum() > 2: out[i] = y[s].mean()
#|     return out
#|
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--sample", type=int, default=C.SHAP_SAMPLE)
#|     a = ap.parse_args()
#|
#|     import joblib, shap
#|     C.banner("05  recompute the attribution surface for the gate candidates")
#|     C.require(C.FRAME, C.MODEL_A)
#|     d = joblib.load(C.MODEL_A)
#|     model, feats = d["model"], d["features"]
#|
#|     missing = [c for c in CANDIDATES if c not in feats]
#|     keep = [c for c in CANDIDATES if c in feats]
#|     
#|     p = C.load_panel()
#|     X, df = p["X"], p["df"]
#|     rng = C.seed_everything()
#|     samp = rng.choice(len(X), size=min(a.sample, len(X)), replace=False)
#|     bg = rng.choice(len(X), size=C.SHAP_BACKGROUND, replace=False)
#|
#|     t0 = time.time()
#|     ex = shap.TreeExplainer(model, data=X.iloc[bg], feature_perturbation="interventional")
#|     
#|     # FIX: Removed the silent=True argument
#|     sv = ex.shap_values(X.iloc[samp], check_additivity=False)
#|     print(f"  interventional TreeSHAP: {time.time()-t0:.0f}s")
#|
#|     idx = {f: i for i, f in enumerate(feats)}
#|     out = df.iloc[samp][[C.GROUP, C.YEAR, C.TARGET]].reset_index(drop=True)
#|     for f in keep:
#|         out[f] = X.iloc[samp][f].to_numpy()
#|         out["shap_" + f] = sv[:, idx[f]]
#|
#|     checks = {}
#|     if C.DEPENDENCE_A.exists():
#|         st = pd.read_csv(C.DEPENDENCE_A)
#|         for f in keep:
#|             if f not in st.columns or ("shap_" + f) not in st.columns: continue
#|             a1 = _binned(out[f].to_numpy(), out["shap_" + f].to_numpy())
#|             a2 = _binned(st[f].to_numpy(), st["shap_" + f].to_numpy())
#|             m = np.isfinite(a1) & np.isfinite(a2)
#|             if m.sum() > 5:
#|                 r = float(np.corrcoef(a1[m], a2[m])[0, 1])
#|                 rng_ = float(np.nanmax(a2[m]) - np.nanmin(a2[m]))
#|                 rms = float(np.sqrt(np.nanmean((a1[m] - a2[m]) ** 2)))
#|                 checks[f] = dict(r=r, rms_over_range=rms / rng_ if rng_ > 0 else float("nan"), curve_range=rng_)
#|
#|     C.DERIVED.mkdir(parents=True, exist_ok=True)
#|     dest = C.DERIVED / "shap_dependence_candidates.csv"
#|     out.to_csv(dest, index=False)
#|     chk = C.DERIVED / "shap_dependence_check.json"
#|     json.dump(dict(n_rows=int(len(out)), sample=int(a.sample), background=C.SHAP_BACKGROUND, seed=C.SEED, candidates=keep, missing=missing, binned_correlation_vs_stored=checks), open(chk, "w"), indent=1)
#|     C.write_manifest(__file__, [C.FRAME, C.MODEL_A], [dest, chk], settings=dict(sample=a.sample, candidates=keep))
#|
#| if __name__ == "__main__": main()
#|
#@@ END analysis/05_recompute_shap_for_threshold_candidates.py
#@@ FILE analysis/06_bootstrap_turning_points_and_gate.py
#| """
#| 06_bootstrap_turning_points_and_gate.py
#| Where each response turns, how far that value moves under resampling, and
#| whether the answer depends on where we drew the line.
#|
#| WHAT A TURNING POINT IS HERE
#| ----------------------------
#| The value of a variable at which its contribution to the prediction changes
#| sign. Observations go into equal-count quantile bins of the variable, the mean
#| interventional attribution is taken in each bin, a smoothing spline is fitted
#| to those means, and the located value is where the spline crosses zero.
#|
#| THE GATE, AND WHO DECIDED IT
#| ----------------------------
#| Nobody outside this project. The three conditions
#|
#|     a crossing is found in at least GATE_MIN_FOUND of the resamples
#|     the interval is narrower than GATE_MAX_WIDTH of the observed range
#|     the point estimate lies inside its own interval
#|
#| are our convention, fixed before any estimate was seen. The percentile
#| bootstrap underneath them is standard; the cut values are not taken from any
#| published standard, and a paper that leans on them should not pretend
#| otherwise.
#|
#| What makes the result defensible is therefore not the cuts but their
#| irrelevance, and this script measures that directly. It sweeps both cuts over
#| a wide grid and reports how many candidates pass at every combination. If the
#| verdict is the same across most of the grid, the convention is not doing the
#| work. If it flips near the values we chose, the paper has a problem and should
#| say so. Either way you will see it rather than guess.
#|
#| ON THE NUMBER OF RESAMPLES
#| --------------------------
#| The first submission used 400. A 2.5th percentile estimated from 400 draws is
#| the tenth order statistic, which moves noticeably from one run to the next.
#| The default here is 2,000, and the script reports the Monte Carlo standard
#| error of each interval endpoint so you can judge whether to go higher. The
#| rule of thumb it prints is based on the spread of the draws themselves, not on
#| a table.
#|
#| READS   derived/shap_dependence_candidates.csv   (from script 05)
#| WRITES  derived/turning_points.json
#|         derived/turning_points.csv
#|         derived/gate_sensitivity.csv
#| TIME    N_BOOT x variables spline fits. At 2,000 draws and eleven variables,
#|         roughly 10 to 25 minutes. Scales linearly, and --boot lets you try a
#|         small number first.
#|
#|     python3 06_bootstrap_turning_points_and_gate.py
#|     python3 06_bootstrap_turning_points_and_gate.py --boot 200      # quick look
#|     python3 06_bootstrap_turning_points_and_gate.py --boot 5000     # publish
#| """
#| from __future__ import annotations
#|
#| # ===========================================================================
#| #  SETTINGS  (N_BOOT, N_BINS and the gate live in _config.py)
#| # ===========================================================================
#| # The grid the gate is swept over. The paper quotes the pass count at the
#| # declared cuts; this shows what it would have been elsewhere.
#| SWEEP_FOUND = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
#| SWEEP_WIDTH = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
#|
#| TRIM_PCT = (1, 99)   # ignore the extreme percentiles of a variable when
#|                      # fitting its curve, so one outlying county cannot place
#|                      # a crossing
#| MIN_DRAWS_FOR_CI = 30
#|
#| # -- the two estimator choices that actually move the verdict ---------------
#| #
#| # A smoothed curve can cross zero more than once, and something has to decide
#| # which crossing is "the" turning point. That choice is not cosmetic: it
#| # changes how many candidates the gate passes, and this script reports the
#| # verdict under both rules so the dependence is visible rather than buried.
#| #
#| #   "steepest"        the crossing where the curve is steepest.
#| #                     This is what the published analysis used. It is the
#| #                     defensible choice, because a crossing on a steep section
#| #                     is a real change of sign and one on a flat section is
#| #                     noise; and it is the LESS forgiving choice, because on a
#| #                     genuinely flat curve the steepest crossing jumps from
#| #                     resample to resample, which is exactly the instability
#| #                     the gate is there to detect.
#| #
#| #   "nearest_median"  the crossing closest to the middle of the data.
#| #                     More stable by construction, and therefore a weaker
#| #                     test: it will pass curves that "steepest" rejects.
#| #
#| CROSSING_RULE = "steepest"      # "steepest" | "nearest_median"
#|
#| #   "median"  the median attribution within each bin (the published choice,
#| #             robust to a few extreme SHAP values)
#| #   "mean"    the mean within each bin
#| BIN_STATISTIC = "median"        # "median" | "mean"
#|
#| # Print the whole table a second time under the other rule, so the paper can
#| # state how far its headline depends on the estimator.
#| REPORT_BOTH_RULES = True
#| # ===========================================================================
#|
#| import argparse
#| import json
#| import warnings
#|
#| import numpy as np
#| import pandas as pd
#|
#| import _config as C
#|
#| warnings.filterwarnings("ignore")
#|
#| NICE = {
#|     "prism_tdmean_min": "Annual minimum dew point",
#|     "eng2_night_cooling_efficiency": "Night cooling efficiency (mean form)",
#|     "eng_night_cooling_eff": "Night cooling efficiency (tail form)",
#|     "lst_night_1km_mean": "Mean night land-surface temperature",
#|     "eng2_wetbulb_night": "Nocturnal wet-bulb temperature",
#|     "eng2_wetbulb_hot": "Warm-hour wet-bulb temperature",
#|     "eng2_hcho_no2_ratio": "Formaldehyde to nitrogen dioxide ratio",
#|     "pm25_mean": "Annual mean fine particulate matter",
#|     "prism_tmin_min": "Annual minimum daily minimum temperature",
#|     "cams_fot_formaldehyde_above75th_percentile":
#|         "Formaldehyde hours above the 75th percentile",
#|     "usda_cropland_usda_corn_pct": "Cropland in corn",
#| }
#| UNIT = {
#|     "prism_tdmean_min": "degC", "eng2_wetbulb_night": "degC",
#|     "eng2_wetbulb_hot": "degC", "prism_tmin_min": "degC",
#|     "lst_night_1km_mean": "degC", "pm25_mean": "ug/m3",
#|     "cams_fot_formaldehyde_above75th_percentile": "%",
#| }
#|
#|
#| def crossing(x, y, nbins, lo, hi, rule=None, binstat=None):
#|     """
#|     Locate the zero crossing of the smoothed attribution curve, or None.
#|
#|     Binned statistic, then a smoothing spline whose smoothing factor comes
#|     from the bin-to-bin scatter rather than from a chosen constant, then the
#|     crossing picked by `rule`. This reproduces the estimator used in the
#|     published analysis when rule="steepest" and binstat="median".
#|     """
#|     from scipy.interpolate import UnivariateSpline
#|     rule = rule or CROSSING_RULE
#|     binstat = binstat or BIN_STATISTIC
#|     m = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
#|     if m.sum() < nbins * 5:
#|         return None
#|     x, y = x[m], y[m]
#|     qs = np.quantile(x, np.linspace(0, 1, nbins + 1))
#|     qs[-1] += 1e-9
#|     bx, by = [], []
#|     for i in range(nbins):
#|         s = (x >= qs[i]) & (x < qs[i + 1])
#|         if s.sum() > 2:
#|             bx.append(np.median(x[s]))
#|             by.append(np.median(y[s]) if binstat == "median" else y[s].mean())
#|     bx, by = np.asarray(bx), np.asarray(by)
#|     keep = np.concatenate([[True], np.diff(bx) > 1e-9])
#|     bx, by = bx[keep], by[keep]
#|     if len(bx) < 8:
#|         return None
#|     r = np.diff(by)
#|     sc = np.median(np.abs(r - np.median(r))) * 1.4826
#|     try:
#|         spl = UnivariateSpline(bx, by, s=max(len(bx) * sc ** 2 * 1.2, 1e-12),
#|                                k=4)
#|     except Exception:
#|         return None
#|     gx = np.linspace(bx.min(), bx.max(), 600)
#|     gy = spl(gx)
#|     sign = np.sign(gy)
#|     idx = np.where(np.diff(sign) != 0)[0]
#|     if not len(idx):
#|         return None
#|
#|     def at(i):
#|         x0, x1, y0, y1 = gx[i], gx[i + 1], gy[i], gy[i + 1]
#|         return float(x0 - y0 * (x1 - x0) / (y1 - y0))
#|
#|     if rule == "steepest":
#|         d1 = spl.derivative(1)
#|         best, best_slope = None, -np.inf
#|         for i in idx:
#|             xi = at(i)
#|             sl = abs(float(d1(xi)))
#|             if sl > best_slope:
#|                 best, best_slope = xi, sl
#|         return best
#|     if rule == "nearest_median":
#|         med = np.median(x)
#|         return at(idx[np.argmin(np.abs(gx[idx] - med))])
#|     raise ValueError(f"unknown CROSSING_RULE {rule!r}")
#|
#|
#| def analyse(name, x, y, nboot, nbins, rng, rule=None, binstat=None):
#|     lo, hi = np.percentile(x[np.isfinite(x)], TRIM_PCT)
#|     span = float(hi - lo)
#|     point = crossing(x, y, nbins, lo, hi, rule, binstat)
#|
#|     draws = []
#|     n = len(x)
#|     for _ in range(nboot):
#|         idx = rng.integers(0, n, n)
#|         v = crossing(x[idx], y[idx], nbins, lo, hi, rule, binstat)
#|         if v is not None and np.isfinite(v):
#|             draws.append(v)
#|     v = np.asarray(draws, float)
#|     out = dict(feature=name, label=NICE.get(name, name),
#|                unit=UNIT.get(name, ""), n=int(n), n_boot_attempted=int(nboot),
#|                n_boot_found=int(len(v)), x_lo=float(lo), x_hi=float(hi),
#|                span=span, point_estimate=point)
#|
#|     if len(v) >= MIN_DRAWS_FOR_CI:
#|         ci_lo, ci_hi = np.percentile(v, [2.5, 97.5])
#|         out["ci_lo"], out["ci_hi"] = float(ci_lo), float(ci_hi)
#|         out["width_frac"] = float((ci_hi - ci_lo) / span) if span else np.nan
#|         # Monte Carlo error on the endpoints: bootstrap the bootstrap draws
#|         mc = [np.percentile(rng.choice(v, len(v)), [2.5, 97.5])
#|               for _ in range(200)]
#|         mc = np.asarray(mc)
#|         out["mc_se_lo"] = float(mc[:, 0].std())
#|         out["mc_se_hi"] = float(mc[:, 1].std())
#|         out["boot_sd"] = float(v.std())
#|     else:
#|         for k in ("ci_lo", "ci_hi", "width_frac", "mc_se_lo", "mc_se_hi",
#|                   "boot_sd"):
#|             out[k] = np.nan
#|     out["found_frac"] = len(v) / nboot if nboot else np.nan
#|     if point is not None and np.isfinite(out.get("ci_lo", np.nan)):
#|         out["pct_above"] = float(100 * (x > point).mean())
#|     else:
#|         out["pct_above"] = np.nan
#|     out["_draws"] = v
#|     return out
#|
#|
#| def passes(r, min_found, max_width, need_inside=True):
#|     if not np.isfinite(r.get("point_estimate") or np.nan):
#|         return False
#|     if r["found_frac"] < min_found:
#|         return False
#|     if not np.isfinite(r["width_frac"]) or r["width_frac"] >= max_width:
#|         return False
#|     if need_inside and not (r["ci_lo"] - 1e-9 <= r["point_estimate"]
#|                             <= r["ci_hi"] + 1e-9):
#|         return False
#|     return True
#|
#|
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--boot", type=int, default=C.N_BOOT)
#|     ap.add_argument("--bins", type=int, default=C.N_BINS)
#|     a = ap.parse_args()
#|
#|     C.banner("06  bootstrap turning points and the stability gate")
#|     src = C.DERIVED / "shap_dependence_candidates.csv"
#|     C.require(src)
#|     d = pd.read_csv(src)
#|     feats = [c for c in d.columns if "shap_" + c in d.columns]
#|     print(f"  {len(feats)} candidate variables, {len(d):,} rows")
#|     print(f"  {a.boot:,} resamples, {a.bins} bins")
#|     print(f"  declared gate: found >= {C.GATE_MIN_FOUND:.0%}, "
#|           f"width < {C.GATE_MAX_WIDTH:.0%} of range, estimate inside interval")
#|     print()
#|
#|     rng = C.seed_everything()
#|     rows = []
#|     for f in feats:
#|         x = d[f].to_numpy(float)
#|         y = d["shap_" + f].to_numpy(float)
#|         r = analyse(f, x, y, a.boot, a.bins, rng)
#|         r["stable"] = passes(r, C.GATE_MIN_FOUND, C.GATE_MAX_WIDTH,
#|                              C.GATE_INSIDE)
#|         rows.append(r)
#|         pt = r["point_estimate"]
#|         print(f"  {'PASS' if r['stable'] else 'fail'}  {r['label'][:44]:<46}"
#|               f"{'' if pt is None else f'{pt:8.3f}'}"
#|               f"  found {r['found_frac']:5.0%}"
#|               f"  width {r['width_frac']*100 if np.isfinite(r['width_frac']) else float('nan'):5.1f}%")
#|
#|     tab = pd.DataFrame([{k: v for k, v in r.items() if k != "_draws"}
#|                         for r in rows])
#|     n_pass = int(tab.stable.sum())
#|     print(f"\n  {len(tab)} candidates, {n_pass} stable, "
#|           f"{len(tab) - n_pass} not")
#|
#|     # -- is the bootstrap big enough? -----------------------------------
#|     print("\nMONTE CARLO ERROR ON THE INTERVAL ENDPOINTS")
#|     print("  If this is small next to the interval width, the resample count")
#|     print("  is sufficient. If it is comparable, raise --boot.")
#|     ok = tab[np.isfinite(tab.ci_lo)]
#|     for _, r in ok.iterrows():
#|         w = r.ci_hi - r.ci_lo
#|         rel = max(r.mc_se_lo, r.mc_se_hi) / w if w else np.nan
#|         flag = "ok   " if rel < 0.10 else "RAISE"
#|         print(f"  {flag} {r.label[:44]:<46}interval {w:8.3f}   "
#|               f"MC s.e. {max(r.mc_se_lo, r.mc_se_hi):7.4f}  ({rel:5.1%})")
#|     worst = (ok[["mc_se_lo", "mc_se_hi"]].max(axis=1)
#|              / (ok.ci_hi - ok.ci_lo)).max()
#|     if np.isfinite(worst):
#|         need = int(np.ceil(a.boot * (worst / 0.10) ** 2 / 100.0) * 100)
#|         print(f"\n  worst endpoint error is {worst:.1%} of its interval.")
#|         if worst >= 0.10:
#|             print(f"  To bring every endpoint under 10 %, use about "
#|                   f"--boot {need:,}.")
#|         else:
#|             print(f"  Every endpoint is under 10 % of its interval at "
#|                   f"{a.boot:,} resamples.")
#|
#|     # -- does the verdict depend on where we drew the line? -------------
#|     print("\nGATE SENSITIVITY  (number of candidates passing)")
#|     grid = []
#|     for mf in SWEEP_FOUND:
#|         row = []
#|         for mw in SWEEP_WIDTH:
#|             k = sum(passes(r, mf, mw, C.GATE_INSIDE) for r in rows)
#|             grid.append(dict(min_found=mf, max_width=mw, n_pass=k))
#|             row.append(k)
#|         print(f"  found >= {mf:.0%}   " +
#|               "".join(f"{v:4d}" for v in row))
#|     print("        width <     " +
#|           "".join(f"{w*100:3.0f}%" for w in SWEEP_WIDTH))
#|
#|     sw = pd.DataFrame(grid)
#|     at_declared = int(sw[(sw.min_found == C.GATE_MIN_FOUND)
#|                          & (sw.max_width == C.GATE_MAX_WIDTH)].n_pass.iloc[0])
#|     same = float((sw.n_pass == at_declared).mean())
#|     print(f"\n  At the declared cuts: {at_declared} pass.")
#|     print(f"  The same verdict holds over {same:.0%} of the swept grid "
#|           f"({len(sw)} combinations).")
#|     if same >= 0.6:
#|         print("  The convention is not driving the answer. That is the")
#|         print("  sentence the paper should make, with this table beside it.")
#|     else:
#|         print("  The verdict moves across the grid. Say so in the paper and")
#|         print("  report the range rather than a single count.")
#|
#|     # -- and how far does it depend on the estimator itself? ------------
#|     alt = None
#|     if REPORT_BOTH_RULES:
#|         other = ("nearest_median" if CROSSING_RULE == "steepest"
#|                  else "steepest")
#|         print(f"\nUNDER THE OTHER CROSSING RULE  ({other})")
#|         print("  The gate asks whether a located value survives resampling.")
#|         print("  Which of several crossings is located is a separate choice,")
#|         print("  and it moves the answer, so both are reported.")
#|         rng2 = C.seed_everything()
#|         arows = []
#|         for f in feats:
#|             r = analyse(f, d[f].to_numpy(float),
#|                         d["shap_" + f].to_numpy(float),
#|                         max(200, a.boot // 4), a.bins, rng2, rule=other)
#|             r["stable"] = passes(r, C.GATE_MIN_FOUND, C.GATE_MAX_WIDTH,
#|                                  C.GATE_INSIDE)
#|             arows.append(r)
#|         n_alt = sum(r["stable"] for r in arows)
#|         print(f"  {CROSSING_RULE:<15} {n_pass} of {len(tab)} stable")
#|         print(f"  {other:<15} {n_alt} of {len(arows)} stable")
#|         flipped = [r["label"] for r, b in zip(arows, rows)
#|                    if r["stable"] != b["stable"]]
#|         if flipped:
#|             print("  verdict changes for:")
#|             for f in flipped:
#|                 print(f"    {f}")
#|             print("\n  This is a real fork and the paper has to pick one rule")
#|             print("  and name it. 'steepest' is the published choice and the")
#|             print("  stricter one: on a flat curve the steepest crossing moves")
#|             print("  between resamples, which is the instability the gate is")
#|             print("  built to catch. 'nearest_median' is stabilised by")
#|             print("  construction and will pass curves that carry no signal.")
#|         else:
#|             print("  The verdict is the same under both rules.")
#|         alt = pd.DataFrame([{k: v for k, v in r.items() if k != "_draws"}
#|                             for r in arows])
#|         alt["rule"] = other
#|
#|     C.DERIVED.mkdir(parents=True, exist_ok=True)
#|     p1 = C.DERIVED / "turning_points.csv"
#|     tab["rule"] = CROSSING_RULE
#|     tab.to_csv(p1, index=False)
#|     if alt is not None:
#|         alt.to_csv(C.DERIVED / "turning_points_alt_rule.csv", index=False)
#|     p2 = C.DERIVED / "gate_sensitivity.csv"
#|     sw.to_csv(p2, index=False)
#|     p3 = C.DERIVED / "turning_points.json"
#|     json.dump(dict(
#|         settings=dict(n_boot=a.boot, n_bins=a.bins, trim=list(TRIM_PCT),
#|                       gate_min_found=C.GATE_MIN_FOUND,
#|                       gate_max_width=C.GATE_MAX_WIDTH,
#|                       gate_inside=C.GATE_INSIDE, seed=C.SEED,
#|                       crossing_rule=CROSSING_RULE,
#|                       bin_statistic=BIN_STATISTIC),
#|         n_candidates=len(tab), n_stable=n_pass,
#|         verdict_unchanged_fraction_of_grid=same,
#|         worst_mc_error_fraction=float(worst) if np.isfinite(worst) else None,
#|         results=tab.to_dict("records")), open(p3, "w"), indent=1, default=str)
#|     print(f"\n[saved] derived/{p1.name}, derived/{p2.name}, derived/{p3.name}")
#|     C.write_manifest(__file__, [src], [p1, p2, p3],
#|                      settings=dict(n_boot=a.boot, n_bins=a.bins),
#|                      notes=f"{n_pass}/{len(tab)} stable; verdict unchanged "
#|                            f"over {same:.0%} of the gate grid")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/06_bootstrap_turning_points_and_gate.py
#@@ FILE analysis/07_learner_and_hyperparameter_table.py
#| """
#| 07_learner_and_hyperparameter_table.py
#| The settings the model was fitted under, and what each learner scored.
#|
#| WHY THIS EXISTS
#| ---------------
#| A reader who wants to know whether a result is an artefact of tuning has no
#| way to tell from the current manuscript. This builds the table that answers
#| it, from two sources you already have: the parameter block in the main runner,
#| and the tournament lines in the run log.
#|
#| WHERE THE SETTINGS COME FROM
#| ----------------------------------------
#| The production model was fitted by pipeline_A_moisture_v2.py, which builds
#| every learner in pipeline_core_v2.build_models with fixed settings (1,200
#| trees for the full run). Those settings were not tuned on these data. This
#| step reads them from that function without running it and checks the XGBoost
#| values against the saved production model (studyA_final_model.joblib). The
#| "gold-mine" parameter block in run_ml_satallite_and_SDOH.py belongs to a
#| different runner and was not used for this model.
#|
#| READS   pipeline/pipeline_core_v2.py   (build_models: the production settings)
#|         studyA_final_model.joblib      (the saved production model, a check)
#|         superset_311981.log            (the tournament results)
#|         studyA_model_tournament.csv    (the same results, as a table)
#| WRITES  derived/learner_settings.csv
#|         tables_v3/T5_learner_settings.tex
#| TIME    seconds
#|
#|     python3 07_learner_and_hyperparameter_table.py
#|     python3 07_learner_and_hyperparameter_table.py --log ../../other_run.log
#| """
#| from __future__ import annotations
#|
#| # ===========================================================================
#| #  SETTINGS
#| # ===========================================================================
#| # Which learners to include, and how to name them in the table.
#| LEARNERS = {"XGB": "XGBoost", "CB": "CatBoost", "RF": "Random forest",
#|             "LGB": "LightGBM"}
#|
#| # Hyperparameters worth printing. Anything else in the block is noise for a
#| # reader and stays in the released code.
#| SHOW = ["n_estimators", "iterations", "max_depth", "depth", "learning_rate",
#|         "subsample", "colsample_bytree", "min_child_weight", "num_leaves",
#|         "reg_alpha", "reg_lambda", "l2_leaf_reg", "random_strength",
#|         "min_samples_leaf", "max_features"]
#|
#| CAPTION = (
#|     "Learner settings and out-of-fold performance. Every learner was fitted "
#|     "on the identical county-grouped folds of the same panel with the fixed "
#|     "settings of the production pipeline, which were not tuned on these "
#|     "data. LightGBM did not complete on the run platform and is reported as "
#|     "not evaluated rather than omitted.")
#| # ===========================================================================
#|
#| import argparse
#| import ast
#| import json
#| import re
#|
#| import pandas as pd
#|
#| import _config as C
#|
#|
#| def parse_params(core_path):
#|     """The settings the production run used, read without executing the file:
#|     the keyword arguments of each learner constructor in
#|     pipeline_core_v2.build_models, with n_est at its full-run value."""
#|     tree = ast.parse(open(core_path, encoding="utf-8").read())
#|     fn = next(n for n in tree.body
#|               if isinstance(n, ast.FunctionDef) and n.name == "build_models")
#|     n_est = None
#|     for node in ast.walk(fn):
#|         if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
#|                 and node.targets[0].id == "n_est"):
#|             n_est = ast.literal_eval(node.value.orelse)    # 400 if fast else 1200
#|     names = {"_xgb": "XGB", "_cat": "CB", "_lgb": "LGB"}
#|     out = {}
#|     for sub in ast.walk(fn):
#|         if isinstance(sub, ast.FunctionDef) and sub.name in names:
#|             for node in ast.walk(sub):
#|                 if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
#|                         and node.targets[0].id == "kw"):
#|                     kv = {}
#|                     for k in node.value.keywords:
#|                         if isinstance(k.value, ast.Name) and k.value.id == "n_est":
#|                             kv[k.arg] = n_est
#|                         else:
#|                             try:
#|                                 kv[k.arg] = ast.literal_eval(k.value)
#|                             except ValueError:
#|                                 pass                    # seed: set at run time
#|                     out[names[sub.name]] = kv
#|     return out
#|
#|
#| def check_against_model(params, model_path):
#|     """Compare the XGBoost settings with the saved production model."""
#|     import joblib
#|     if not model_path.exists():
#|         print(f"  saved model not found at {model_path}; not checked")
#|         return None
#|     m = joblib.load(model_path)["model"]
#|     saved = vars(m)
#|     keys = ["n_estimators", "max_depth", "learning_rate", "min_child_weight",
#|             "subsample", "colsample_bytree", "reg_lambda"]
#|     ok = all(saved.get(k) == params["XGB"].get(k) for k in keys)
#|     for k in keys:
#|         print(f"    {'ok   ' if saved.get(k) == params['XGB'].get(k) else 'CHECK'} "
#|               f"{k:<18} code {params['XGB'].get(k)}   saved model {saved.get(k)}")
#|     return ok
#|
#|
#| def parse_log(log_path, study="A"):
#|     """
#|     Tournament lines: R2, MAE, RMSE, seconds, and any failure.
#|
#|     The log holds BOTH studies one after the other, and their tournaments look
#|     identical apart from the numbers. Reading the whole file and keeping the
#|     last match returns Study B's fused model, whose R-squared is 0.86 rather
#|     than 0.67. This paper is Study A, so the file is cut at the study headings
#|     first. The cross-check against the stored tournament table at the bottom
#|     of this script exists because that mistake is easy to make and silent.
#|     """
#|     res = {}
#|     if not log_path.exists():
#|         return res
#|     text = open(log_path, encoding="utf-8", errors="replace").read()
#|     marks = [(m.start(), m.group(1)) for m in
#|              re.finditer(r"###\s*STUDY\s+([AB])\b", text)]
#|     if marks:
#|         spans = {}
#|         for i, (pos, tag) in enumerate(marks):
#|             stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
#|             spans[tag] = text[pos:stop]
#|         if study not in spans:
#|             print(f"  the log has no 'STUDY {study}' section; "
#|                   f"found {sorted(spans)}")
#|             return res
#|         text = spans[study]
#|         print(f"  reading the STUDY {study} section only "
#|               f"({len(text.splitlines())} lines of {len(open(log_path).readlines())})")
#|     for line in text.splitlines():
#|         m = re.search(r"\[tournament\]\s+(\w[\w ]*?)\s+"
#|                       r"R2=([\d.]+)\s+MAE=([\d.]+)\s+RMSE=([\d.]+)\s+\((\d+)s\)",
#|                       line)
#|         if m:
#|             res[m.group(1).strip()] = dict(
#|                 r2=float(m.group(2)), mae=float(m.group(3)),
#|                 rmse=float(m.group(4)), seconds=int(m.group(5)))
#|             continue
#|         f = re.search(r"\[tournament\]\s+(\w[\w ]*?)\s+FAILED:\s*(.+?)\s*$",
#|                       line)
#|         if f:
#|             res[f.group(1).strip()] = dict(failed=f.group(2).strip())
#|         w = re.search(r"\[tournament\]\s+winner:\s*(\w+)", line)
#|         if w:
#|             res.setdefault("_winner", w.group(1))
#|     return res
#|
#|
#| def fmt(v):
#|     if isinstance(v, float):
#|         return f"{v:.4g}"
#|     return str(v)
#|
#|
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--core", default=str(C.PAPER / "pipeline" / "pipeline_core_v2.py"))
#|     ap.add_argument("--log", default=str(C.RUN_LOG))
#|     ap.add_argument("--study", default="A", choices=["A", "B"],
#|                     help="which study's tournament to read from the log; "
#|                          "this paper is Study A")
#|     a = ap.parse_args()
#|
#|     from pathlib import Path
#|     core, log = Path(a.core), Path(a.log)
#|
#|     C.banner("07  learner settings and the tournament")
#|     if not core.exists():
#|         print(f"  pipeline_core_v2.py not found at {core}")
#|     if not log.exists():
#|         print(f"  run log not found at {log}")
#|
#|     params = parse_params(core) if core.exists() else {}
#|     print("  XGBoost settings against the saved production model:")
#|     matched = check_against_model(params, C.MODEL_A) if params else None
#|     scores = parse_log(log, a.study)
#|     winner = scores.pop("_winner", None)
#|     print(f"  parameter blocks found: {', '.join(sorted(params)) or 'none'}")
#|     print(f"  tournament rows found : {', '.join(sorted(scores)) or 'none'}")
#|     if winner:
#|         print(f"  winner recorded in the log: {winner}")
#|
#|     rows = []
#|     for key, nice in LEARNERS.items():
#|         kv = params.get(key, {})
#|         shown = {k: kv[k] for k in SHOW if k in kv}
#|         sc = scores.get(nice, {})
#|         rows.append(dict(
#|             learner=nice,
#|             settings="; ".join(f"{k}={fmt(v)}" for k, v in shown.items())
#|                      or "not recorded",
#|             r2=sc.get("r2"), mae=sc.get("mae"), rmse=sc.get("rmse"),
#|             seconds=sc.get("seconds"), failed=sc.get("failed", ""),
#|             winner=(nice == LEARNERS.get(winner, winner))))
#|     tab = pd.DataFrame(rows)
#|     print()
#|     print(tab[["learner", "r2", "mae", "rmse", "seconds", "failed"]]
#|           .to_string(index=False))
#|
#|     # cross-check against the stored tournament table
#|     if C.TOURNAMENT_A.exists():
#|         st = pd.read_csv(C.TOURNAMENT_A)
#|         print(f"\n  against {C.TOURNAMENT_A.name}:")
#|         for _, r in st.iterrows():
#|             mine = tab.loc[tab.learner == r.get("model"), "r2"]
#|             got = float(mine.iloc[0]) if len(mine) and pd.notna(mine.iloc[0]) \
#|                 else None
#|             same = got is not None and abs(got - float(r["r2"])) < 5e-5
#|             print(f"    {'ok   ' if same else 'CHECK'} {r.get('model'):<12}"
#|                   f"log {got}   table {r['r2']}")
#|
#|     # ---- LaTeX -------------------------------------------------------
#|     L = [r"\begin{table}[H]",
#|          r"\caption{" + CAPTION + r"}",
#|          r"\label{tab:learners}",
#|          r"\begin{adjustbox}{max width=\textwidth}",
#|          r"\begin{tabular}{llrrr}", r"\toprule",
#|          r"Learner & Settings & $R^{2}$ & MAE (yr) & RMSE (yr) \\",
#|          r"\midrule"]
#|     for _, r in tab.iterrows():
#|         s = (r.settings.replace("_", r"\_")
#|              .replace("colsample\\_bytree", "colsample")
#|              .replace("min\\_child\\_weight", "min child wt"))
#|         if r.failed:
#|             L.append(f"{r.learner} & {s} & \\multicolumn{{3}}{{l}}"
#|                      f"{{not evaluated: {r.failed}}} \\\\")
#|         else:
#|             star = r"\,$^{\ast}$" if r.winner else ""
#|             L.append(f"{r.learner}{star} & {s} & {r.r2:.4f} & {r.mae:.4f} & "
#|                      f"{r.rmse:.4f} \\\\")
#|     L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
#|           r"\vspace{2pt}",
#|           r"\footnotesize $^{\ast}$ the learner carried forward. "
#|           r"All scores are out of fold on county-grouped splits.",
#|           r"\end{table}"]
#|
#|     C.DERIVED.mkdir(parents=True, exist_ok=True)
#|     C.TABLES.mkdir(parents=True, exist_ok=True)
#|     p1 = C.DERIVED / "learner_settings.csv"
#|     tab.to_csv(p1, index=False)
#|     p2 = C.TABLES / "T5_learner_settings.tex"
#|     open(p2, "w", encoding="utf-8").write("\n".join(L) + "\n")
#|     p3 = C.DERIVED / "learner_settings.json"
#|     json.dump(dict(params=params, scores=scores, winner=winner,
#|                    source="pipeline_core_v2.build_models",
#|                    matches_saved_model=matched),
#|               open(p3, "w"), indent=1, default=str)
#|     print(f"\n[saved] derived/{p1.name}, tables_v3/{p2.name}, "
#|           f"derived/{p3.name}")
#|     C.write_manifest(__file__, [core, log], [p1, p2, p3],
#|                      settings=dict(study=a.study),
#|                      notes="settings-and-tournament table, not a search table")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/07_learner_and_hyperparameter_table.py
#@@ FILE analysis/10_verify_reproducibility_by_rerunning.py
#| """
#| 10_verify_reproducibility_by_rerunning.py
#| Run a script twice and prove its answers did not move.
#|
#| WHY THIS EXISTS
#| ---------------
#| A number in a manuscript has to survive being recomputed. The way to know that
#| is not to reason about seeds but to run the thing again and compare the bytes.
#| This does exactly that: it records the hash of every output a script wrote,
#| reruns the script, and compares. Any difference is printed with the file that
#| differs.
#|
#| It is also the script to reach for when a number in the paper has changed and
#| you want to know why. The manifests hold the settings each run used, so a
#| changed number is traced to a changed setting rather than to chance.
#|
#| WHAT IT CANNOT PROVE
#| --------------------
#| That the answer is right. Only that it is stable. A reproducible mistake is
#| still a mistake.
#|
#| WHAT IT COSTS
#| -------------
#| It reruns the script, so verifying script 04 costs a second full refit. Verify
#| the fast steps freely; each slow one costs as much as its first run.
#|
#|     python3 10_verify_reproducibility_by_rerunning.py 02
#|     python3 10_verify_reproducibility_by_rerunning.py 02 06
#|     python3 10_verify_reproducibility_by_rerunning.py --all-fast
#|     python3 10_verify_reproducibility_by_rerunning.py      # same as --all-fast
#| """
#| from __future__ import annotations
#|
#| # ===========================================================================
#| #  SETTINGS
#| # ===========================================================================
#| # Scripts that finish quickly enough to verify as a matter of routine.
#| FAST = ["02", "07"]
#| # Scripts that cost a full refit; verify these deliberately.
#| SLOW = ["03", "04", "05", "06"]
#| # Extra arguments passed to a script when verifying it, to keep the cost down
#| # while still exercising the same code path.
#| VERIFY_ARGS = {
#|     "06": ["--boot", "200"],
#| }
#| # ===========================================================================
#|
#| import argparse
#| import json
#| import subprocess
#| import sys
#| from pathlib import Path
#|
#| import _config as C
#|
#|
#| def find_script(prefix: str) -> Path:
#|     hits = sorted(C.ANALYSIS.glob(f"{prefix}_*.py"))
#|     if not hits:
#|         raise SystemExit(f"no script in this folder starts with {prefix!r}")
#|     return hits[0]
#|
#|
#| def manifest_for(script: Path) -> Path:
#|     return C.MANIFESTS / (script.stem + ".manifest.json")
#|
#|
#| def snapshot(script: Path):
#|     m = manifest_for(script)
#|     if not m.exists():
#|         return None
#|     return json.load(open(m))
#|
#|
#| def run(script: Path, extra):
#|     print(f"\n  running {script.name} "
#|           f"{' '.join(extra) if extra else ''}".rstrip())
#|     r = subprocess.run([sys.executable, str(script), *extra],
#|                        cwd=str(C.ANALYSIS), capture_output=True, text=True)
#|     if r.returncode != 0:
#|         print(r.stdout[-3000:])
#|         print(r.stderr[-3000:])
#|         raise SystemExit(f"  {script.name} exited {r.returncode}")
#|     return r.stdout
#|
#|
#| def compare(before, after, script):
#|     print(f"\n  {script.name}")
#|     if before is None:
#|         print("    no earlier manifest; this run recorded the baseline.")
#|         print("    run this command again to compare against it.")
#|         return None
#|     # The thread count is recorded but cannot move an answer: script 01
#|     # proves the fitted model identical at 1 and 4 threads. A run on a
#|     # machine with a different core count is therefore still a rerun.
#|     ignore = {"threads"}
#|     strip = lambda d: {k: v for k, v in d.items() if k not in ignore}
#|     same_settings = strip(before["settings"]) == strip(after["settings"])
#|     if before["settings"].get("threads") != after["settings"].get("threads"):
#|         print(f"    threads {before['settings'].get('threads')} -> "
#|               f"{after['settings'].get('threads')} (does not affect results; see 01)")
#|     if not same_settings:
#|         print("    SETTINGS DIFFER between the two runs:")
#|         for k in sorted((set(before["settings"]) | set(after["settings"])) - ignore):
#|             b, a = before["settings"].get(k), after["settings"].get(k)
#|             if b != a:
#|                 print(f"      {k}: {b!r}  ->  {a!r}")
#|         print("    A different answer is expected. Fix the settings, not the")
#|         print("    number.")
#|     ok = True
#|     keys = sorted(set(before["outputs"]) | set(after["outputs"]))
#|     if not keys:
#|         print("    this script writes no data files; nothing to compare")
#|         return same_settings
#|     for k in keys:
#|         b = before["outputs"].get(k, "absent before")
#|         a = after["outputs"].get(k, "absent after")
#|         if b == a:
#|             print(f"    identical  {k}")
#|         else:
#|             ok = False
#|             print(f"    DIFFERS    {k}   {b} -> {a}")
#|     return ok and same_settings
#|
#|
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("scripts", nargs="*", help="number prefixes, e.g. 02 06")
#|     ap.add_argument("--all-fast", action="store_true")
#|     ap.add_argument("--all", action="store_true")
#|     a = ap.parse_args()
#|
#|     picks = a.scripts or (FAST if a.all_fast else
#|                           (FAST + SLOW if a.all else []))
#|     if not picks:
#|         # Run with no arguments (for example from a loop over every numbered
#|         # script), verify the fast scripts rather than stopping the loop.
#|         print("  no script named; verifying the fast ones "
#|               f"({' '.join(FAST)}). Slow ones: {' '.join(SLOW)}, "
#|               "verify those by naming them.")
#|         picks = FAST
#|
#|     C.banner("10  reproducibility check")
#|     print("  Each script is rerun and its outputs compared, by hash, against")
#|     print("  the manifest of the previous run.\n")
#|
#|     verdicts = {}
#|     for pref in picks:
#|         s = find_script(pref)
#|         before = snapshot(s)
#|         run(s, VERIFY_ARGS.get(pref, []))
#|         after = snapshot(s)
#|         if after is None:
#|             print(f"\n  {s.name} wrote no manifest; cannot verify")
#|             verdicts[s.name] = None
#|             continue
#|         verdicts[s.name] = compare(before, after, s)
#|
#|     print("\n" + "=" * 74)
#|     bad = [k for k, v in verdicts.items() if v is False]
#|     new = [k for k, v in verdicts.items() if v is None]
#|     good = [k for k, v in verdicts.items() if v is True]
#|     for k in good:
#|         print(f"  REPRODUCES   {k}")
#|     for k in new:
#|         print(f"  baseline set {k}  (run again to compare)")
#|     for k in bad:
#|         print(f"  DOES NOT     {k}")
#|     print("=" * 74)
#|     if bad:
#|         print("  Do not publish a number from a script that does not")
#|         print("  reproduce until you know which setting moved.")
#|         sys.exit(1)
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/10_verify_reproducibility_by_rerunning.py
#@@ FILE analysis/11_verify_nce_identity_by_search.py
#| """
#| verify_nce_identity.py -- is the night-cooling pair one field described twice,
#| or two engineered variables built from two different pairs of sources?
#|
#| The paper's central claim rests on the first being true. If the two night
#| cooling variables were derived from different instruments -- one from MODIS,
#| say, and one from ERA5 or PRISM -- then the twenty-fold attribution gap between
#| them would be a comparison of two products and not a comparison of two summary
#| statistics, and the result would collapse.
#|
#| The panel holds enough temperature-like columns from enough instruments that
#| either variable COULD have been built several ways. So this does not check the
#| formula the pipeline claims to use. It searches every ordered pair of every
#| temperature-like column in the panel, across five ratio forms, and asks which
#| combinations reproduce each target. A claim survives only if the combinations
#| that reproduce it all come from one instrument family.
#|
#| Two passes, because the full search is 94 x 93 x 5 x 2 evaluations over 52,309
#| rows: a subsample scan to shortlist, then full-panel confirmation of the
#| shortlist at full precision.
#|
#| What "exact" means here is set by the data, not by a constant. Both targets are
#| stored float64, so the tolerance is stated as a relative agreement and reported
#| alongside the physical size of the residual in kelvin, which is what decides
#| whether a residual is a different source or a different arithmetic path.
#|
#|     python3 verify_nce_identity.py --frame <parquet> --out nce_identity.json
#| """
#| from __future__ import annotations
#|
#| import argparse
#| import itertools
#| import json
#| import re
#|
#| import numpy as np
#| import pandas as pd
#| import pyarrow.parquet as pq
#|
#| import os as _os, sys as _sys
#| _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
#| import _config as C
#|
#|
#| TARGETS = ["eng_night_cooling_eff", "eng2_night_cooling_efficiency"]
#|
#| # every column whose name marks it as a temperature, whatever instrument wrote
#| # it. Deliberately generous: a false candidate costs search time, a missing
#| # candidate would let a rival explanation go untested.
#| TEMP = re.compile(r"(lst|temp|tmax|tmin|tmean|skin|thermal)", re.I)
#|
#| FORMS = {
#|     "(a-b)/a": lambda A, B: (A - B) / A,
#|     "(a-b)/b": lambda A, B: (A - B) / B,
#|     "(a-b)/(a+273.15)": lambda A, B: (A - B) / (A + 273.15),
#|     "(a-b)/(a+b)": lambda A, B: (A - B) / (A + B),
#|     "a-b": lambda A, B: A - B,
#| }
#|
#| # a pair is shortlisted at NEAR and confirmed at EXACT_REL. EXACT_REL is a
#| # relative bound: an instrument disagreement shows up at 1e-2 or worse, an
#| # arithmetic-path difference at 1e-5 or better, and there is nothing in
#| # between, so the exact value of the cut does not decide anything.
#| NEAR = 1e-2
#| EXACT_REL = 1e-5
#| SUBSAMPLE = 2500
#|
#|
#| def source_of(col: str) -> str:
#|     return col.split("_", 1)[0]
#|
#|
#| def scan(frame: str) -> dict:
#|     names = pq.ParquetFile(frame).schema.names
#|     cand = [c for c in names if TEMP.search(c) and c not in TARGETS]
#|     d = pq.read_table(frame, columns=sorted(set(cand + TARGETS))).to_pandas()
#|
#|     by_src: dict[str, int] = {}
#|     for c in cand:
#|         by_src[source_of(c)] = by_src.get(source_of(c), 0) + 1
#|     print(f"{len(d):,} rows, {len(cand)} temperature-like columns "
#|           f"from {len(by_src)} sources")
#|     for s, n in sorted(by_src.items()):
#|         print(f"   {s:<10}{n:>4}")
#|
#|     sub = d.sample(min(SUBSAMPLE, len(d)), random_state=0)
#|     report = {}
#|     search = dict(n_rows=int(len(d)), n_columns=len(cand),
#|                   n_sources=len(by_src), by_source=by_src,
#|                   n_forms=len(FORMS), near=NEAR, exact_rel=EXACT_REL)
#|
#|     for tgt in TARGETS:
#|         near = []
#|         t_sub = sub[tgt].to_numpy(float)
#|         ok_sub = np.isfinite(t_sub)
#|         for a, b in itertools.permutations(cand, 2):
#|             A = sub[a].to_numpy(float)
#|             B = sub[b].to_numpy(float)
#|             for label, f in FORMS.items():
#|                 with np.errstate(all="ignore"):
#|                     v = f(A, B)
#|                 m = ok_sub & np.isfinite(v)
#|                 if m.sum() < 200:
#|                     continue
#|                 md = np.abs(v[m] - t_sub[m]).max()
#|                 if md < NEAR:
#|                     near.append((md, label, a, b))
#|         near.sort()
#|
#|         hits = []
#|         t = d[tgt].to_numpy(float)
#|         for _, label, a, b in near[:12]:
#|             A = d[a].to_numpy(float)
#|             B = d[b].to_numpy(float)
#|             with np.errstate(all="ignore"):
#|                 v = FORMS[label](A, B)
#|             m = np.isfinite(v) & np.isfinite(t)
#|             if m.sum() < 1000:
#|                 continue
#|             resid = v[m] - t[m]
#|             md = float(np.abs(resid).max())
#|             rel = float((np.abs(resid) / np.abs(t[m])).max())
#|             # the residual re-expressed as the offset in b that would remove
#|             # it, which for a temperature ratio is a temperature in kelvin
#|             if label == "(a-b)/a":
#|                 kelvin = float(np.median(np.abs(A[m] * (1 - t[m]) - B[m])))
#|             else:
#|                 kelvin = float("nan")
#|             hits.append(dict(form=label, a=a, b=b, n=int(m.sum()),
#|                              maxdiff=md, max_rel=rel, kelvin=kelvin,
#|                              r=float(np.corrcoef(v[m], t[m])[0, 1]),
#|                              source_a=source_of(a), source_b=source_of(b),
#|                              exact=rel < EXACT_REL))
#|         hits.sort(key=lambda h: h["max_rel"])
#|         report[tgt] = hits
#|
#|         print(f"\n### {tgt}")
#|         print(f"    {len(near)} pairs within {NEAR} on the subsample, "
#|               f"{sum(h['exact'] for h in hits)} confirmed on the full panel")
#|         for h in hits[:3]:
#|             tag = "  <-- IDENTITY" if h["exact"] else ""
#|             k = "" if np.isnan(h["kelvin"]) else f"  = {h['kelvin']:.1e} K"
#|             print(f"    max|d|={h['maxdiff']:.4e}  rel={h['max_rel']:.2e}{k}"
#|                   f"  {h['form']}  a={h['a']}  b={h['b']}  "
#|                   f"n={h['n']:,}{tag}")
#|     return report, search
#|
#|
#| def verdict(report: dict) -> dict:
#|     srcs, per_target = set(), {}
#|     ok = True
#|     for tgt, hits in report.items():
#|         ex = [h for h in hits if h["exact"]]
#|         s = {h["source_a"] for h in ex} | {h["source_b"] for h in ex}
#|         per_target[tgt] = sorted(s)
#|         srcs |= s
#|         if len(ex) != 1 or len(s) != 1:
#|             ok = False
#|     single = len(srcs) == 1
#|     print("\n" + "=" * 70)
#|     if ok and single:
#|         src = srcs.pop()
#|         print(f"VERDICT: each variable is reproduced by exactly one combination")
#|         print(f"         of columns, and both combinations come from the same")
#|         print(f"         instrument family, {src!r}. No column from any other")
#|         print(f"         source comes within {NEAR} of either target.")
#|         print("         The two variables are one field described twice.")
#|     else:
#|         print("VERDICT: NOT a clean single-source identity. Inspect the hits.")
#|         for t, s in per_target.items():
#|             print(f"         {t}: {s}")
#|     print("=" * 70)
#|     return dict(single_source=ok and single, sources=per_target)
#|
#|
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--frame", default=str(C.FRAME))
#|     ap.add_argument("--out", default=str(C.DERIVED / "nce_identity.json"))
#|     a = ap.parse_args()
#|     rep, search = scan(a.frame)
#|     v = verdict(rep)
#|     json.dump(dict(hits=rep, verdict=v, search=search),
#|               open(a.out, "w"), indent=2)
#|     print("[saved]", a.out)
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/11_verify_nce_identity_by_search.py
#@@ FILE analysis/12_fields_with_several_summaries.py
#| """
#| 12_fields_with_several_summaries.py
#| How far apart the summaries of one field are in attribution.
#|
#| Every admitted column name carries its field and its statistic
#| ("prism_tdmean_min" is the field prism_tdmean summarised by its minimum).
#| This step groups the admitted columns of the production ranking by field,
#| keeps the fields with more than one admitted summary, and for each field
#| divides the attribution of its most attributed summary by that of its least
#| attributed one. The final paper quotes the median of that spread over all
#| such fields (Section 3.2).
#|
#| This is the final-paper copy of 12_summary_statistic_census_and_matched_refit.py.
#| The functions split_name and census are unchanged; the census test and the
#| matched refit, which the final paper no longer reports, are not included.
#|
#| READS   results_studyA_*/tables/A_shap_ranking.csv   (the admitted columns)
#| WRITES  derived/summary_statistic_census.csv          (one row per field)
#|         derived/summary_statistic_census.json         (median and largest spread)
#|
#|     python3 12_fields_with_several_summaries.py
#| """
#| from __future__ import annotations
#|
#| STAT_CLASS = {
#|     "mean": "central", "p50": "central", "median": "central",
#|     "min": "extreme", "max": "extreme",
#|     "p05": "extreme", "p10": "extreme", "p25": "extreme",
#|     "p70": "extreme", "p75": "extreme", "p90": "extreme", "p95": "extreme",
#|     "stddev": "dispersion", "std": "dispersion", "range": "dispersion",
#|     "iqr": "dispersion", "cv": "dispersion",
#| }
#|
#| MIN_STATS_PER_FIELD = 2
#|
#| import json
#| import os
#| import re
#| import sys
#| import numpy as np
#| import pandas as pd
#|
#| sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#| import _config as C
#|
#| TOKEN = re.compile(r"^(?P<field>.+)_(?P<stat>[a-z0-9]+)$")
#|
#| def split_name(name: str):
#|     m = TOKEN.match(name)
#|     if not m: return None
#|     stat = m.group("stat")
#|     cls = STAT_CLASS.get(stat)
#|     if cls is None: return None
#|     return m.group("field"), cls, stat
#|
#| def census(rank: pd.DataFrame) -> pd.DataFrame:
#|     parsed = []
#|     for _, r in rank.iterrows():
#|         s = split_name(r.feature)
#|         if s is None: continue
#|         field, cls, stat = s
#|         parsed.append(dict(field=field, stat=stat, cls=cls, feature=r.feature, shap=float(r.mean_abs_shap),
#|                            rank=int(r["rank"]), modality=r.modality, label=r.get("label", r.feature)))
#|     p = pd.DataFrame(parsed)
#|     if p.empty: raise SystemExit("No feature name in the ranking carries a statistic token.")
#|
#|     rows = []
#|     for field, g in p.groupby("field"):
#|         if len(g) < MIN_STATS_PER_FIELD: continue
#|         cen = g[g.cls == "central"]
#|         non = g[g.cls != "central"]
#|         best = g.loc[g.shap.idxmax()]
#|         worst = g.loc[g.shap.idxmin()]
#|         rows.append(dict(
#|             field=field, n_stats=len(g), modality=g.modality.iloc[0],
#|             has_central=len(cen) > 0, has_noncentral=len(non) > 0,
#|             best_stat=best.stat, best_cls=best.cls, best_shap=best.shap, best_rank=best["rank"], best_label=best.label,
#|             worst_stat=worst.stat, worst_shap=worst.shap, worst_rank=worst["rank"],
#|             spread=float(best.shap / worst.shap) if worst.shap > 0 else np.nan,
#|             central_stat=cen.loc[cen.shap.idxmax()].stat if len(cen) else "",
#|             central_shap=float(cen.shap.max()) if len(cen) else np.nan, central_rank=int(cen["rank"].min()) if len(cen) else -1,
#|             noncentral_stat=non.loc[non.shap.idxmax()].stat if len(non) else "",
#|             noncentral_shap=float(non.shap.max()) if len(non) else np.nan, noncentral_rank=int(non["rank"].min()) if len(non) else -1,
#|         ))
#|     out = pd.DataFrame(rows).sort_values("best_shap", ascending=False)
#|     both = out.has_central & out.has_noncentral
#|     out["comparable"] = both
#|     out["noncentral_wins"] = np.where(both, out.noncentral_shap > out.central_shap, False)
#|     out["ratio"] = np.where(both, out.noncentral_shap / out.central_shap, np.nan)
#|     return out
#|
#|
#| def main():
#|     C.seed_everything()
#|     rank = C.load_table("shap_ranking")
#|     if "mean_abs_shap" not in rank.columns:
#|         raise SystemExit("The ranking has no mean_abs_shap column")
#|     cen = census(rank)
#|     C.banner("fields with more than one admitted summary")
#|     print(f"  fields                                   {len(cen)}")
#|     print(f"  median spread, most over least attributed {cen.spread.median():.2f}x")
#|     C.DERIVED.mkdir(parents=True, exist_ok=True)
#|     p_csv = C.DERIVED / "summary_statistic_census.csv"
#|     cen.to_csv(p_csv, index=False)
#|     p_json = C.DERIVED / "summary_statistic_census.json"
#|     json.dump(dict(n_fields=int(len(cen)),
#|                    median_spread=float(cen.spread.median()),
#|                    max_spread=float(cen.spread.max())),
#|               open(p_json, "w"), indent=1)
#|     C.write_manifest(__file__, [], [p_csv, p_json], notes="fields and spreads")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/12_fields_with_several_summaries.py
#@@ FILE analysis/13_spatial_holdout_and_attribution_structure.py
#| """
#| 13_spatial_holdout_and_attribution_structure.py
#| The four checks a methodological referee asks for, each written to a file.
#|
#| WHY THIS SCRIPT EXISTS
#| ----------------------
#| County-grouped cross-validation stops a county predicting itself. It does not
#| stop a county being predicted by its neighbours, and a referee who works on
#| spatial data will say so. These four checks answer that objection and the three
#| that usually follow it.
#|
#| 1.  STATE-LEVEL HOLDOUT.  Folds are grouped by state rather than by county, so
#|     every test county is in a state the model never saw. This is a harder test
#|     than county grouping and the drop from it is the honest measure of how much
#|     of the skill is spatial interpolation.
#|
#| 2.  RESIDUAL SPATIAL STRUCTURE.  Global Moran's I on the out-of-fold residuals
#|     over a four-nearest-neighbour graph of county centroids, with a permutation
#|     reference distribution so the value carries a p-value rather than standing
#|     alone.
#|
#| 3.  MULTICOLLINEARITY.  Maximum pairwise correlation and variance inflation
#|     among the leading variables. Interventional attribution is defined under
#|     correlation, but a referee will still want the number.
#|
#| 4.  INTERACTION STRUCTURE.  TreeSHAP interaction values among the leading
#|     variables, which say whether those variables act additively or whether the
#|     effect of one depends on the level of another. A turning point reported for
#|     an additive variable means something simpler than one reported for a
#|     variable in a strong interaction.
#|
#| Every number this script prints is also written to a file. Nothing is typed
#| into the source: an earlier version of this script carried a previous run's
#| result in a print statement and skipped the computation, which is how a number
#| reaches a manuscript without anything having produced it.
#|
#| READS   the modelling panel
#|         conus_counties_5070.gpkg   (county centroids, checks 2 only)
#| WRITES  derived/spatial_holdout.json
#|         derived/residual_spatial_structure.json
#|         derived/multicollinearity.json
#|         derived/attribution_interactions.json
#| TIME    state holdout 5 to 15 min, Moran 5 to 15 min, the other two under 5
#|
#|     python3 13_spatial_holdout_and_attribution_structure.py
#|     python3 13_spatial_holdout_and_attribution_structure.py --only holdout
#|     python3 13_spatial_holdout_and_attribution_structure.py --only vif,interact
#| """
#| from __future__ import annotations
#|
#| # ===========================================================================
#| #  SETTINGS
#| # ===========================================================================
#| N_TOP = 15                 # leading variables used by the collinearity check
#| N_TOP_INTERACT = 6         # leading variables used by the interaction check
#| INTERACT_ROWS = 800        # rows for the interaction values, which are O(p^2)
#| # The variables whose turning points the paper reports. They are included in
#| # the interaction check whether or not they are among the leading few, because
#| # the question the check answers is about those turning points.
#| INTERACT_ALWAYS = ["prism_tdmean_min", "eng2_wetbulb_night",
#|                    "eng_night_cooling_eff"]
#| MORAN_NEIGHBOURS = 4       # k in the k-nearest-neighbour spatial weights
#| MORAN_PERMUTATIONS = 999   # reference distribution for the pseudo p-value
#| VIF_WARN = 10.0            # the conventional line for "severe"
#| # ===========================================================================
#|
#| import argparse
#| import gc
#| import json
#| import os
#| import sys
#| import time
#|
#| import numpy as np
#| import pandas as pd
#|
#| sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#| import _config as C
#|
#| HERE = os.path.dirname(os.path.abspath(__file__))
#|
#|
#| # ---------------------------------------------------------------------------
#| #  Fold-level checkpoints
#| # ---------------------------------------------------------------------------
#| #  The two cross-validated checks each fit the production model eight times,
#| #  which takes ten to thirty minutes. Each fold's out-of-fold predictions are
#| #  written as they are produced, so a run interrupted by a wall clock or a
#| #  suspended machine resumes at the first fold that is missing.
#|
#| def fold_store(tag):
#|     d = C.DERIVED / "checkpoints_stress" / tag
#|     d.mkdir(parents=True, exist_ok=True)
#|     return d
#|
#|
#| def cv_oof(X, y, split_groups, tag):
#|     """Out-of-fold predictions, through the shared checkpointed helper.
#|
#|     The folds already written under the old per-script store are honoured, so
#|     a run started before this moved into _config does not refit them.
#|     """
#|     old = fold_store(tag)
#|     new = C.DERIVED / "checkpoints_cv" / tag
#|     new.mkdir(parents=True, exist_ok=True)
#|     for f in sorted(old.glob("fold*.npy")):
#|         dest = new / f.name
#|         if not dest.exists():
#|             dest.write_bytes(f.read_bytes())
#|     return C.cv_oof(X, y, split_groups, tag)
#|
#|
#| def fit_full(X, y):
#|     import xgboost as xgb
#|     m = xgb.XGBRegressor(**C.XGB_PARAMS, n_jobs=C.threads())
#|     m.fit(X, y)
#|     return m
#|
#|
#| # ---------------------------------------------------------------------------
#| #  1. State-level holdout
#| # ---------------------------------------------------------------------------
#| def spatial_holdout(panel):
#|     from sklearn.metrics import r2_score, mean_absolute_error
#|
#|     print("\n" + "=" * 74)
#|     print("  1. STATE-LEVEL HOLDOUT")
#|     print("=" * 74)
#|     X, y, groups = panel["X"], panel["y"], panel["groups"]
#|     fips = pd.to_numeric(pd.Series(groups), errors="coerce")
#|     if fips.isna().any():
#|         raise SystemExit("Some county codes are not numeric; cannot derive "
#|                          "a state code from them.")
#|     states = (fips // 1000).astype(int).to_numpy()
#|     n_states = len(np.unique(states))
#|     print(f"  {len(np.unique(fips)):,} counties in {n_states} states, "
#|           f"{C.N_FOLDS} folds grouped by state")
#|
#|     t0 = time.time()
#|     oof = cv_oof(X, y, states, "state_holdout")
#|
#|     out = dict(r2=float(r2_score(y, oof)),
#|                mae=float(mean_absolute_error(y, oof)),
#|                n_states=int(n_states), n_folds=int(C.N_FOLDS),
#|                n_obs=int(len(y)), seconds=round(time.time() - t0, 1))
#|     print(f"\n  state-grouped out-of-fold R2  {out['r2']:.4f}")
#|     print(f"  state-grouped out-of-fold MAE {out['mae']:.4f} years")
#|     print("  Compare against the county-grouped figure in the performance")
#|     print("  table. The difference between them is the part of the skill that")
#|     print("  county grouping alone does not control for.")
#|     p = C.DERIVED / "spatial_holdout.json"
#|     json.dump(out, open(p, "w"), indent=1)
#|     print(f"[saved] {p}")
#|     return [p]
#|
#|
#| # ---------------------------------------------------------------------------
#| #  2. Residual spatial structure
#| # ---------------------------------------------------------------------------
#| def centroids():
#|     import geopandas as gpd
#|     gpkg = os.path.join(HERE, "conus_counties_5070.gpkg")
#|     if not os.path.exists(gpkg):
#|         raise SystemExit(f"County boundaries missing: {gpkg}")
#|     g = gpd.read_file(gpkg, layer="county")
#|     if "fips" not in g.columns:
#|         g["fips"] = pd.to_numeric(g["STATE"] + g["COUNTY"], errors="coerce")
#|     c = g.geometry.centroid
#|     return pd.DataFrame({"fips": pd.to_numeric(g["fips"], errors="coerce"),
#|                          "x": c.x.to_numpy(), "y": c.y.to_numpy()}
#|                         ).dropna().astype({"fips": int})
#|
#|
#| def residual_structure(panel):
#|     from scipy.spatial import cKDTree
#|
#|     print("\n" + "=" * 74)
#|     print("  2. RESIDUAL SPATIAL STRUCTURE")
#|     print("=" * 74)
#|     X, y, groups = panel["X"], panel["y"], panel["groups"]
#|
#|     print("  out-of-fold predictions, county-grouped")
#|     oof = cv_oof(X, y, groups, "county_oof")
#|
#|     # one residual per county, averaged over its years, because the spatial
#|     # weights are between counties and not between county-years
#|     res = (pd.DataFrame({"fips": pd.to_numeric(pd.Series(groups)),
#|                          "r": y - oof})
#|            .groupby("fips", as_index=False)["r"].mean())
#|     xy = centroids().merge(res, on="fips")
#|     print(f"  {len(xy):,} counties placed")
#|
#|     coords = xy[["x", "y"]].to_numpy()
#|     tree = cKDTree(coords)
#|     _, nb = tree.query(coords, k=MORAN_NEIGHBOURS + 1)
#|     nb = nb[:, 1:]
#|
#|     z = xy["r"].to_numpy() - xy["r"].mean()
#|     denom = float(np.sum(z ** 2))
#|
#|     def moran(vals):
#|         lag = vals[nb].mean(axis=1)
#|         return float(np.sum(vals * lag) / denom)
#|
#|     obs = moran(z)
#|     rng = C.seed_everything()
#|     null = np.array([moran(rng.permutation(z))
#|                      for _ in range(MORAN_PERMUTATIONS)])
#|     p = float((np.sum(np.abs(null) >= abs(obs)) + 1)
#|               / (MORAN_PERMUTATIONS + 1))
#|
#|     out = dict(morans_i=obs, p_value=p, n_counties=int(len(xy)),
#|                neighbours=MORAN_NEIGHBOURS,
#|                permutations=MORAN_PERMUTATIONS,
#|                null_mean=float(null.mean()), null_sd=float(null.std()))
#|     # The same statistic for the production model's own out-of-fold residual
#|     # (the one step 28 audits and Figure 8 maps) and for the outcome itself,
#|     # which shows how much spatial clustering the model has absorbed.
#|     prod = pd.read_csv(C.OOF_A)
#|     prod["r"] = prod[C.TARGET] - prod["pred"]
#|     for key, col in (("production", "r"), ("outcome", C.TARGET)):
#|         ser = prod.groupby(C.GROUP)[col].mean().rename("v").reset_index()
#|         ser = ser.rename(columns={C.GROUP: "fips"})
#|         v = xy[["fips"]].merge(ser, on="fips", how="left")["v"].to_numpy()
#|         zz = v - v.mean()
#|         dd = float(np.sum(zz ** 2))
#|         stat = lambda q: float(np.sum(q * q[nb].mean(axis=1)) / dd)
#|         o = stat(zz)
#|         nl = np.array([stat(rng.permutation(zz))
#|                        for _ in range(MORAN_PERMUTATIONS)])
#|         out[key] = dict(morans_i=o, p_value=float(
#|             (np.sum(np.abs(nl) >= abs(o)) + 1) / (MORAN_PERMUTATIONS + 1)))
#|         print(f"  Moran's I, {key:<10}              {o:.4f}   "
#|               f"p {out[key]['p_value']:.4f}")
#|     print(f"\n  Moran's I on the county mean residual   {obs:.4f}")
#|     print(f"  permutation pseudo p-value              {p:.4f}")
#|     print(f"  null distribution                       mean {null.mean():+.4f}, "
#|           f"sd {null.std():.4f}")
#|     pth = C.DERIVED / "residual_spatial_structure.json"
#|     json.dump(out, open(pth, "w"), indent=1)
#|     print(f"[saved] {pth}")
#|     return [pth]
#|
#|
#| # ---------------------------------------------------------------------------
#| #  3. Multicollinearity among the leading variables
#| # ---------------------------------------------------------------------------
#| def multicollinearity(panel):
#|     from sklearn.preprocessing import StandardScaler
#|
#|     print("\n" + "=" * 74)
#|     print("  3. MULTICOLLINEARITY AMONG THE LEADING VARIABLES")
#|     print("=" * 74)
#|     X, y = panel["X"], panel["y"]
#|     rank = C.load_table("shap_ranking").sort_values("rank")
#|     top = [c for c in rank.feature.head(N_TOP) if c in X.columns]
#|     print(f"  the {len(top)} leading variables by attribution")
#|
#|     Z = StandardScaler().fit_transform(X[top].fillna(X[top].median()))
#|     corr = np.corrcoef(Z, rowvar=False)
#|     np.fill_diagonal(corr, 0.0)
#|     i, j = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
#|
#|     # The variance inflation factor is 1/(1-R2) of each column on the others.
#|     # Solving that by ordinary least squares overflows when two columns are
#|     # nearly identical, which is exactly the case this check exists to measure.
#|     # The correlation matrix inverse gives the same quantity and is stable: the
#|     # kth diagonal element of the inverse correlation matrix IS the kth VIF.
#|     # A pseudo-inverse keeps it finite when the matrix is singular.
#|     full = np.corrcoef(Z, rowvar=False)
#|     inv = np.linalg.pinv(full, rcond=1e-12)
#|     vifs = {col: float(abs(inv[k, k])) for k, col in enumerate(top)}
#|
#|     vals = np.array([v for v in vifs.values() if np.isfinite(v)])
#|     out = dict(n_variables=len(top),
#|                max_abs_correlation=float(abs(corr[i, j])),
#|                max_correlation_pair=[top[i], top[j]],
#|                median_vif=float(np.median(vals)),
#|                max_vif=float(vals.max()),
#|                max_vif_variable=max(vifs, key=lambda c: vifs[c]),
#|                n_over_warn=int((vals > VIF_WARN).sum()),
#|                warn_level=VIF_WARN, vif=vifs)
#|     print(f"  largest absolute pairwise correlation   {out['max_abs_correlation']:.3f}")
#|     print(f"      between {top[i]} and {top[j]}")
#|     print(f"  median variance inflation factor        {out['median_vif']:.2f}")
#|     print(f"  largest                                 {out['max_vif']:.2f} "
#|           f"({out['max_vif_variable']})")
#|     print(f"  variables above the conventional {VIF_WARN:.0f}        "
#|           f"{out['n_over_warn']} of {len(top)}")
#|     p = C.DERIVED / "multicollinearity.json"
#|     json.dump(out, open(p, "w"), indent=1)
#|     print(f"[saved] {p}")
#|     return [p]
#|
#|
#| # ---------------------------------------------------------------------------
#| #  4. Interaction structure
#| # ---------------------------------------------------------------------------
#| def interactions(panel):
#|     # XGBoost's own pred_interactions is used rather than the shap package's
#|     # TreeExplainer. They compute the same quantity, and the package's XGBoost
#|     # loader fails against boosters whose base_score is serialised as a vector,
#|     # which is how recent XGBoost writes it. Calling the booster directly has
#|     # no version coupling between two libraries to go wrong.
#|     import xgboost as xgb
#|
#|     print("\n" + "=" * 74)
#|     print("  4. INTERACTION STRUCTURE AMONG THE LEADING VARIABLES")
#|     print("=" * 74)
#|     X, y = panel["X"], panel["y"]
#|     rank = C.load_table("shap_ranking").sort_values("rank")
#|     top = [c for c in rank.feature.head(N_TOP_INTERACT) if c in X.columns]
#|     for c in INTERACT_ALWAYS:
#|         if c in X.columns and c not in top:
#|             top.append(c)
#|     print(f"  {len(top)} variables, {INTERACT_ROWS} county-years")
#|     print("  Interaction values are quadratic in the number of columns, so the")
#|     print("  model is refitted on these columns alone. Interaction strength is")
#|     print("  therefore reported relative to this reduced model, which is what")
#|     print("  the question is about: do these variables act on each other.")
#|
#|     rng = C.seed_everything()
#|     idx = np.sort(rng.choice(len(y), size=min(INTERACT_ROWS, len(y)),
#|                              replace=False))
#|     Xi = X[top].iloc[idx]
#|     m = fit_full(X[top], y)
#|     t0 = time.time()
#|     dm = xgb.DMatrix(Xi, feature_names=list(Xi.columns))
#|     iv = m.get_booster().predict(dm, pred_interactions=True)
#|     # the last row and column are the bias term, which is not a variable
#|     iv = np.asarray(iv)[:, :len(top), :len(top)]
#|
#|     main = np.abs(np.einsum("nii->ni", iv)).mean(axis=0)
#|     pairs = []
#|     for a in range(len(top)):
#|         for b in range(a + 1, len(top)):
#|             pairs.append(dict(a=top[a], b=top[b],
#|                               strength=float(np.abs(iv[:, a, b]).mean() * 2)))
#|     pairs.sort(key=lambda d: -d["strength"])
#|     total_main = float(main.sum())
#|     total_inter = float(sum(p["strength"] for p in pairs))
#|     share = total_inter / (total_main + total_inter) if total_main else np.nan
#|
#|     out = dict(variables=top, n_rows=int(len(idx)),
#|                reduced_model=True, n_columns_in_model=len(top),
#|                main_effect_total=total_main,
#|                interaction_total=total_inter,
#|                interaction_share=float(share),
#|                strongest_pair=pairs[0], pairs=pairs[:10],
#|                main_effects={c: float(v) for c, v in zip(top, main)},
#|                seconds=round(time.time() - t0, 1))
#|     print(f"\n  strongest pair   {pairs[0]['a']}")
#|     print(f"                   {pairs[0]['b']}")
#|     print(f"                   {pairs[0]['strength']:.4f} years")
#|     print(f"  interaction share of total attribution   {100 * share:.1f} per cent")
#|     print("  This is the share within this reduced model. Interactions are")
#|     print("  larger in a model given only these columns than in the full one,")
#|     print("  because structure the full model attributes to other variables")
#|     print("  has nowhere else to go. Read it as an upper bound on how far")
#|     print("  these variables act on one another, not as an estimate of the")
#|     print("  share in the production model.")
#|     p = C.DERIVED / "attribution_interactions.json"
#|     json.dump(out, open(p, "w"), indent=1)
#|     print(f"[saved] {p}")
#|     return [p]
#|
#|
#| CHECKS = {"holdout": spatial_holdout, "moran": residual_structure,
#|           "vif": multicollinearity, "interact": interactions}
#|
#|
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--only", default="all",
#|                     help="comma separated: " + ", ".join(CHECKS))
#|     a = ap.parse_args()
#|     want = list(CHECKS) if a.only == "all" else [s.strip()
#|                                                  for s in a.only.split(",")]
#|     bad = [w for w in want if w not in CHECKS]
#|     if bad:
#|         raise SystemExit(f"unknown check(s): {', '.join(bad)}\n"
#|                          f"    known: {', '.join(CHECKS)}")
#|
#|     C.banner("13  spatial holdout and attribution structure")
#|     C.seed_everything()
#|     panel = C.load_panel()
#|     out = []
#|     for w in want:
#|         out += CHECKS[w](panel)
#|     C.write_manifest(__file__, [], out, notes=f"checks={','.join(want)}")
#|     print("\n" + "=" * 74)
#|     print(f"  {len(out)} result file(s) written to {C.DERIVED.name}/")
#|     print("=" * 74)
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/13_spatial_holdout_and_attribution_structure.py
#@@ FILE analysis/14_regional_invariance_of_turning_points.py
#| """
#| 14_regional_invariance_of_turning_points.py
#| Does a located turning point move when the country is cut in two?
#|
#| A turning point located on a national panel can be an artefact of how the
#| variable is distributed across the country rather than a property of the
#| fitted response. This step splits the attributed county-years at the median
#| longitude or at the 100th meridian, relocates each crossing on each side with
#| the estimator of script 06 (same bins, same rule, same attribution surface),
#| and reports the west-minus-east shift, in units and as a share of the
#| variable's observed range (Sections 2.9 and 3.5, Figure 7C).
#|
#| This is the final-paper copy of 14_information_theory_and_regional_invariance.py.
#| Its functions are unchanged; the mutual-information census, which the final
#| paper no longer reports, is not included.
#|
#| READS   derived/shap_dependence_candidates.csv     (script 05)
#|         conus_counties_5070.gpkg                   (county centroids)
#| WRITES  derived/threshold_regional_invariance_all.json        (median longitude)
#|         derived/threshold_regional_invariance_100W_all.json   (100th meridian)
#|
#|     python3 14_regional_invariance_of_turning_points.py --all-candidates
#|     python3 14_regional_invariance_of_turning_points.py --all-candidates --split meridian
#| """
#| from __future__ import annotations
#|
#| # ===========================================================================
#| #  SETTINGS
#| # ===========================================================================
#| # The three variables that describe one nocturnal physical state. Names are the
#| # column names in the panel, not labels.
#| TRIPLET = [
#|     "prism_tdmean_min",          # annual minimum dew point
#|     "eng2_wetbulb_night",        # nocturnal wet-bulb temperature
#|     "eng_night_cooling_eff",     # night-time cooling efficiency, tail form
#| ]
#|
#| # Where the panel is cut in two. "median" splits at the median longitude of the
#| # county-years, which puts about half the observations on each side; a fixed
#| # meridian splits at a named line and leaves the halves unequal.
#| SPLIT = "median"             # "median" | "meridian"
#| MERIDIAN = -100.0            # degrees east, used only when SPLIT="meridian"
#|
#| # A region must hold at least this many county-years for a crossing located in
#| # it to be reported. Below it the estimator is being asked for more than the
#| # data supports.
#| MIN_REGION_ROWS = 2000
#| # ===========================================================================
#|
#| import argparse
#| import json
#| import os
#| import sys
#|
#| import numpy as np
#| import pandas as pd
#|
#| sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#| import _config as C
#|
#| HERE = os.path.dirname(os.path.abspath(__file__))
#|
#|
#| def canonical_crossing():
#|     """The crossing estimator of script 06, imported rather than reimplemented.
#|
#|     A second implementation of this estimator would give values that differ in
#|     the second decimal from the ones in the threshold table, and no reader
#|     could tell whether the difference was regional or procedural.
#|     """
#|     import importlib.util
#|     path = os.path.join(HERE, "06_bootstrap_turning_points_and_gate.py")
#|     if not os.path.exists(path):
#|         raise SystemExit(
#|             f"06_bootstrap_turning_points_and_gate.py is not beside this file.\n"
#|             f"    This script locates crossings with that script's estimator so\n"
#|             f"    the regional values are comparable with the threshold table.\n"
#|             f"    Copy the whole analysis folder rather than single scripts.")
#|     spec = importlib.util.spec_from_file_location("_s06", path)
#|     mod = importlib.util.module_from_spec(spec)
#|     spec.loader.exec_module(mod)
#|     return mod
#|
#|
#| # ---------------------------------------------------------------------------
#| #  2. Regional invariance of the located turning points
#| # ---------------------------------------------------------------------------
#| def county_longitude():
#|     """Longitude of each county centroid, in degrees, keyed by FIPS."""
#|     import geopandas as gpd
#|     gpkg = os.path.join(HERE, "conus_counties_5070.gpkg")
#|     if not os.path.exists(gpkg):
#|         raise SystemExit(
#|             f"The county boundaries are missing:\n    {gpkg}\n"
#|             f"    This check needs county centroids to split the panel.")
#|     g = gpd.read_file(gpkg, layer="county")
#|     if "fips" not in g.columns:
#|         g["fips"] = pd.to_numeric(g["STATE"] + g["COUNTY"], errors="coerce")
#|     cen = g.geometry.centroid.to_crs("EPSG:4326")
#|     out = pd.DataFrame({"fips": pd.to_numeric(g["fips"], errors="coerce"),
#|                         "lon": cen.x.to_numpy()})
#|     return out.dropna().astype({"fips": int}).set_index("fips")["lon"]
#|
#|
#| def regional_invariance(args):
#|     print("\n" + "=" * 74)
#|     print("  2. REGIONAL INVARIANCE of the located turning points")
#|     print("=" * 74)
#|
#|     s06 = canonical_crossing()
#|     print(f"  estimator: script 06, rule '{s06.CROSSING_RULE}', bin statistic "
#|           f"'{s06.BIN_STATISTIC}', {C.N_BINS} bins")
#|
#|     dep = C.load_table("shap_dependence_candidates")
#|     if "fips" not in dep.columns:
#|         raise SystemExit(
#|             "shap_dependence_candidates.csv has no fips column, so the rows "
#|             "cannot be placed geographically. Rerun script 05.")
#|     lon = county_longitude()
#|     dep = dep.assign(lon=dep["fips"].map(lon))
#|     dep = dep[dep["lon"].notna()]
#|
#|     split = getattr(args, "split", None) or SPLIT
#|     cut = float(dep["lon"].median()) if split == "median" else MERIDIAN
#|     west = dep["lon"] < cut
#|     east = ~west
#|     print(f"  split at {cut:.2f} degrees east: "
#|           f"{int(west.sum()):,} county-years west, {int(east.sum()):,} east")
#|
#|     results = {}
#|     # The three nocturnal variables are the test the paper is built on. With
#|     # --all-candidates every variable the threshold table reports is located in
#|     # both halves as well, so a reader can see which turning points are
#|     # climate-invariant and which are not, instead of only the ones that are.
#|     feats = TRIPLET
#|     if getattr(args, "all_candidates", False):
#|         feats = [c for c in dep.columns
#|                  if f"shap_{c}" in dep.columns and c not in ("fips",)]
#|     for feat in feats:
#|         if feat not in dep.columns or f"shap_{feat}" not in dep.columns:
#|             print(f"\n  {feat}: not in the dependence sample, skipped")
#|             results[feat] = dict(reason="absent from the dependence sample")
#|             continue
#|
#|         def locate(mask):
#|             d = dep[mask]
#|             if len(d) < MIN_REGION_ROWS:
#|                 return None
#|             x = d[feat].to_numpy(float)
#|             y = d["shap_" + feat].to_numpy(float)
#|             lo, hi = np.percentile(x[np.isfinite(x)], s06.TRIM_PCT)
#|             return s06.crossing(x, y, C.N_BINS, lo, hi)
#|
#|         all_, w, e = locate(dep.index == dep.index), locate(west), locate(east)
#|         row = dict(national=all_, west=w, east=e,
#|                    n_west=int(west.sum()), n_east=int(east.sum()))
#|         if w is not None and e is not None:
#|             span = float(np.nanpercentile(dep[feat], 99)
#|                          - np.nanpercentile(dep[feat], 1))
#|             row["delta"] = abs(w - e)
#|             row["delta_pct_of_range"] = 100 * abs(w - e) / span if span else None
#|             row["span"] = span
#|         results[feat] = row
#|
#|         print(f"\n  {feat}")
#|         for k in ("national", "west", "east"):
#|             v = row.get(k)
#|             print(f"      {k:<9} {'none located' if v is None else f'{v:.3f}'}")
#|         if "delta" in row:
#|             print(f"      west minus east: {row['delta']:.3f} "
#|                   f"({row['delta_pct_of_range']:.1f} per cent of the "
#|                   f"observed range)")
#|
#|     located = [r for r in results.values() if r.get("delta") is not None]
#|     if located:
#|         worst = max(r["delta_pct_of_range"] for r in located)
#|         print(f"\n  largest regional disagreement across the {len(located)} "
#|               f"variables located in both halves: {worst:.1f} per cent of the "
#|               f"observed range")
#|     # The median split and the 100th-meridian split are both kept, in separate
#|     # files, because they answer different questions: the median gives two
#|     # halves of equal size, the meridian is the classical arid-humid divide.
#|     name = ("threshold_regional_invariance.json" if split == "median"
#|             else "threshold_regional_invariance_100W.json")
#|     if getattr(args, "all_candidates", False):
#|         name = name.replace(".json", "_all.json")
#|     p = C.DERIVED / name
#|     json.dump(dict(split=split, cut_degrees_east=cut,
#|                    estimator=dict(rule=s06.CROSSING_RULE,
#|                                   bin_statistic=s06.BIN_STATISTIC,
#|                                   n_bins=C.N_BINS,
#|                                   trim_pct=list(s06.TRIM_PCT)),
#|                    results=results), open(p, "w"), indent=1, default=str)
#|     print(f"[saved] {p}")
#|     return [p]
#|
#|
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--split", choices=("median", "meridian"), default=SPLIT,
#|                     help="median longitude, or the 100th meridian west")
#|     ap.add_argument("--all-candidates", action="store_true",
#|                     help="locate every threshold candidate, not only the "
#|                          "three nocturnal variables")
#|     a = ap.parse_args()
#|
#|     C.banner("14  regional invariance of the turning points")
#|     C.seed_everything()
#|     out = regional_invariance(a)
#|     C.write_manifest(__file__, [], out,
#|                      notes=f"split={a.split}, all_candidates={a.all_candidates}")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/14_regional_invariance_of_turning_points.py
#@@ FILE analysis/16_export_frozen_panels_and_gate_audit.py
#| """
#| 16_export_frozen_panels_and_gate_audit.py
#| Two frozen modelling panels, and a measurement of what the admission gate is
#| worth.
#|
#| WHY THIS SCRIPT EXISTS
#| ----------------------
#| Every script in this folder takes its column list from the fitted model
#| artefact, so every one of them inherits the coverage gate and the variance and
#| correlation filters. That is true and it is not obvious, and an analysis that
#| started from the master CSV instead would silently include the 776 columns the
#| gate removed. Scores from such a run are higher and mean nothing, because the
#| removed columns include ones available in only part of the window and ones that
#| are near-duplicates of a column already present.
#|
#| This script makes that impossible to get wrong twice. It writes the admitted
#| panel out as a flat CSV with the key columns beside it, so an analysis can start
#| from a file that already has the gate applied, and it writes a manifest naming
#| every column with the checksum of the file it came from.
#|
#| It then measures the thing the frozen panel protects against: the same learner,
#| the same folds and the same seed, fitted on the gated set and on the ungated
#| candidate set, so the difference is a number rather than an intuition.
#|
#| READS   the reference frame, the fitted model, the admission record
#| WRITES  derived/panel_all_admitted.csv          fips, year, outcome + admitted
#|         derived/panel_environment_only.csv      the same minus any non-physical
#|         derived/panel_manifest.json             columns, checksums, provenance
#|         derived/gate_audit.json                 what the gate costs in R2
#| TIME    export under two minutes; the audit adds 20 to 50
#|
#|     python3 16_export_frozen_panels_and_gate_audit.py
#|     python3 16_export_frozen_panels_and_gate_audit.py --skip-audit
#|     python3 16_export_frozen_panels_and_gate_audit.py --audit-only
#| """
#| from __future__ import annotations
#|
#| # ===========================================================================
#| #  SETTINGS
#| # ===========================================================================
#| # Three panels are written, each a strict subset of the one before it.
#| #
#| #   all_admitted      every column the fitted model uses
#| #   environment_only  minus anything socioeconomic, clinical or economic
#| #   satellite_only    minus the interpolated and reanalysis products too, so
#| #                     what is left is measured from orbit
#| #
#| # The second is a guard rather than a filter: no socioeconomic column clears
#| # the coverage gate in this run, so it currently equals the first. If a future
#| # gate admits one, the panel still will not contain it.
#| NON_PHYSICAL_MODALITIES = {
#|     "Socioeconomic (SDOH)", "Health outcomes", "Macro economics",
#| }
#| NON_PHYSICAL_PATTERNS = [
#|     r"^sdoh_", r"^acs_", r"^census_", r"^svi_", r"mortality", r"_death",
#|     r"lifeexpect", r"^econ_", r"poverty", r"income", r"uninsured",
#| ]
#|
#| # Modalities that are measured from orbit. PRISM, gridMET and TerraClimate are
#| # interpolations of surface station records and ERA5 and CAMS are reanalyses
#| # that assimilate observations into a model, so none of the four is a satellite
#| # measurement however much satellite data informs it. The engineered layers are
#| # excluded from this panel because each is built from more than one source and
#| # a single provenance cannot be claimed for them.
#| SATELLITE_MODALITIES = {
#|     "MODIS land-surface temperature", "MODIS vegetation", "Landsat optical",
#|     "Night-time lights", "Canopy structure", "Surface water",
#|     "Cropland and agriculture", "Fine particulate matter",
#|     "Human modification", "Sentinel-1 radar", "Sentinel-2 optical",
#|     "Sentinel-5P air quality",
#| }
#|
#| # The audit's arms. "gated" is the production set. "ungated" is every numeric
#| # candidate column present in the frame, which is what an analysis that skipped
#| # the gate would see.
#| AUDIT_ARMS = ["gated", "ungated", "gated_no_protected"]
#| AUDIT_FOLDS = None            # None uses C.N_FOLDS
#| # ===========================================================================
#|
#| import argparse
#| import gc
#| import json
#| import os
#| import re
#| import sys
#|
#| import numpy as np
#| import pandas as pd
#|
#| sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#| import _config as C
#|
#| NON_PHYSICAL_RE = re.compile("|".join(NON_PHYSICAL_PATTERNS), re.I)
#|
#|
#| def mod_of(gate, col):
#|     hit = gate.loc[gate.feature == col, "modality"]
#|     return str(hit.iloc[0]) if len(hit) else "unknown"
#|
#|
#| def gate_record():
#|     """The admission record, with the modality of every candidate column."""
#|     g = C.load_table("feature_gate")
#|     need = {"feature", "admitted"}
#|     if not need <= set(g.columns):
#|         raise SystemExit(
#|             f"The admission record has columns {list(g.columns)}; this script "
#|             f"needs at least {sorted(need)}.")
#|     return g
#|
#|
#| def split_columns(model_features, gate):
#|     """The three nested column lists, in the model's own order."""
#|     mod = dict(zip(gate.feature, gate.get("modality", pd.Series(dtype=str))))
#|     env, sat = [], []
#|     for c in model_features:
#|         if mod.get(c) in NON_PHYSICAL_MODALITIES or NON_PHYSICAL_RE.search(c):
#|             continue
#|         env.append(c)
#|         if mod.get(c) in SATELLITE_MODALITIES:
#|             sat.append(c)
#|     return list(model_features), env, sat
#|
#|
#| def export(panel, model_features, gate, args):
#|     C.banner("16  frozen panels")
#|     df = panel["df"]
#|     X = panel["X"]
#|     keys = [C.GROUP, C.YEAR, C.TARGET]
#|     missing = [k for k in keys if k not in df.columns]
#|     if missing:
#|         raise SystemExit(f"The frame has no {missing}; cannot key the panel.")
#|
#|     admitted, env, sat = split_columns(model_features, gate)
#|     print(f"  {len(admitted)} admitted columns")
#|     print(f"  {len(env)} after removing socioeconomic, clinical and economic")
#|     print(f"  {len(sat)} measured from orbit")
#|     if len(env) == len(admitted):
#|         print("  The first two are identical: no socioeconomic column clears")
#|         print("  the coverage gate in this run, which is the environment-only")
#|         print("  claim checked at the level of the columns themselves.")
#|     by_mod = {}
#|     for c in sat:
#|         by_mod[mod_of(gate, c)] = by_mod.get(mod_of(gate, c), 0) + 1
#|     for k, v in sorted(by_mod.items(), key=lambda kv: -kv[1]):
#|         print(f"      {v:>4}  {k}")
#|
#|     out = []
#|     for name, cols in (("panel_all_admitted", admitted),
#|                        ("panel_environment_only", env),
#|                        ("panel_satellite_only", sat)):
#|         frame = pd.concat([df[keys].reset_index(drop=True),
#|                            X[cols].reset_index(drop=True)], axis=1)
#|         frame = frame.rename(columns={C.GROUP: "fips", C.YEAR: "year",
#|                                       C.TARGET: "life_expectancy"})
#|         p = C.DERIVED / f"{name}.csv"
#|         frame.to_csv(p, index=False)
#|         print(f"  [saved] {p.name}  {frame.shape[0]:,} rows x "
#|               f"{frame.shape[1]} columns")
#|         out.append(p)
#|
#|     man = dict(
#|         source_frame=str(C.FRAME), source_frame_sha256=C.sha256(C.FRAME),
#|         source_model=str(C.MODEL_A), source_model_sha256=C.sha256(C.MODEL_A),
#|         n_candidates=int(len(gate)),
#|         n_gate_admitted=int(gate.admitted.sum()),
#|         n_modelled=len(admitted),
#|         n_environment_only=len(env),
#|         n_satellite_only=len(sat),
#|         window=[int(panel["window"][0]), int(panel["window"][1])],
#|         n_rows=int(len(df)), n_counties=int(df[C.GROUP].nunique()),
#|         key_columns=["fips", "year", "life_expectancy"],
#|         columns_all_admitted=admitted,
#|         columns_environment_only=env,
#|         columns_satellite_only=sat,
#|         satellite_modalities=sorted(SATELLITE_MODALITIES),
#|         note=("These columns are the fitted model's own feature list. They have "
#|               "already passed the coverage gate and the variance and "
#|               "correlation filters of the main run. An analysis starting from "
#|               "the master CSV instead would see every candidate column and is "
#|               "not comparable with anything in this paper."),
#|     )
#|     p = C.DERIVED / "panel_manifest.json"
#|     json.dump(man, open(p, "w"), indent=1)
#|     print(f"  [saved] {p.name}")
#|     out.append(p)
#|     return out
#|
#|
#| def audit(panel, model_features, gate, args):
#|     """What the gate is worth, measured on the same folds with the same seed."""
#|     import pyarrow.parquet as pq
#|
#|     C.banner("16  what the admission gate is worth")
#|     y, groups = panel["y"], panel["groups"]
#|     n_folds = AUDIT_FOLDS or C.N_FOLDS
#|
#|     # the ungated arm: every numeric candidate column in the frame
#|     sch = pq.read_schema(C.FRAME)
#|     drop = {C.GROUP, C.YEAR, C.TARGET}
#|     cand = [n for n in sch.names if n not in drop]
#|     print(f"  the frame carries {len(cand)} candidate columns")
#|     print(f"  the gate and the filters leave {len(model_features)}")
#|
#|     protected = set()
#|     if "protected" in gate.columns:
#|         protected = set(gate.loc[gate.protected.astype(bool), "feature"])
#|     print(f"  {len(protected & set(model_features))} of the modelled columns "
#|           f"are protected")
#|
#|     def fit_arm(cols, label, frame=None):
#|         # Per-fold checkpoints live in _config.cv_score, so an arm interrupted
#|         # part way resumes at the fold it reached rather than at fold one.
#|         ck = C.DERIVED / "checkpoints_gate_audit" / f"{label}.json"
#|         ck.parent.mkdir(parents=True, exist_ok=True)
#|         if ck.exists():
#|             done = json.load(open(ck))
#|             print(f"    {label:<22} already done: R2 {done['r2']:.4f}")
#|             return done
#|         Xa = frame if frame is not None else panel["X"][cols]
#|         r = C.cv_score(Xa, y, groups, f"gate_{label}", n_folds=n_folds)
#|         r["label"] = label
#|         json.dump(r, open(ck, "w"), indent=1)
#|         print(f"    {label:<22} R2 {r['r2']:.4f}  [checkpointed]", flush=True)
#|         return r
#|
#|     arms = {}
#|     if "gated" in AUDIT_ARMS:
#|         arms["gated"] = fit_arm(model_features, "gated")
#|     if "ungated" in AUDIT_ARMS:
#|         lo, hi = panel["window"]
#|         raw = pq.read_table(C.FRAME,
#|                             columns=sorted(set(cand + list(drop)))).to_pandas()
#|         raw = raw[(raw[C.YEAR] >= lo) & (raw[C.YEAR] <= hi)
#|                   & raw[C.TARGET].notna()].reset_index(drop=True)
#|         Xu = raw[cand].apply(pd.to_numeric, errors="coerce")
#|         Xu = Xu.loc[:, Xu.notna().any()]
#|         Xu = Xu.fillna(Xu.median()).astype("float32")
#|         print(f"  ungated arm: {Xu.shape[1]} numeric columns after coercion")
#|         arms["ungated"] = fit_arm(None, "ungated", frame=Xu)
#|         del raw, Xu
#|         gc.collect()
#|     if "gated_no_protected" in AUDIT_ARMS and protected:
#|         keep = [c for c in model_features if c not in protected]
#|         arms["gated_no_protected"] = fit_arm(keep, "gated_no_protected")
#|
#|     print("\n  out of fold, identical county-grouped folds:")
#|     for k, r in arms.items():
#|         print(f"      {k:<22} R2 {r['r2']:.4f}   MAE {r['mae']:.4f}   "
#|               f"{r['n_features']} columns")
#|     if "gated" in arms and "ungated" in arms:
#|         d = arms["ungated"]["r2"] - arms["gated"]["r2"]
#|         print(f"\n  skipping the gate changes out-of-fold R2 by {d:+.4f}, on "
#|               f"{arms['ungated']['n_features'] - arms['gated']['n_features']:+d} "
#|               f"columns.")
#|         print("  A higher score there is not a better model. The removed")
#|         print("  columns include ones present for only part of the window and")
#|         print("  ones that duplicate a column already in the set, so the extra")
#|         print("  skill is partly the model learning which years and which")
#|         print("  duplicates it is looking at.")
#|     if "gated" in arms and "gated_no_protected" in arms:
#|         d = arms["gated"]["r2"] - arms["gated_no_protected"]["r2"]
#|         print(f"\n  the protected columns are worth {d:+.4f} R2 between them.")
#|
#|     p = C.DERIVED / "gate_audit.json"
#|     json.dump(dict(arms=arms, n_candidates_in_frame=len(cand),
#|                    n_modelled=len(model_features),
#|                    n_protected_modelled=len(protected & set(model_features)),
#|                    n_folds=n_folds), open(p, "w"), indent=1)
#|     print(f"[saved] {p}")
#|     return [p]
#|
#|
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--skip-audit", action="store_true")
#|     ap.add_argument("--audit-only", action="store_true")
#|     a = ap.parse_args()
#|
#|     C.seed_everything()
#|     gate = gate_record()
#|     panel = C.load_panel()
#|     feats = panel["features"]
#|     out = []
#|     if not a.audit_only:
#|         out += export(panel, feats, gate, a)
#|     if not a.skip_audit:
#|         out += audit(panel, feats, gate, a)
#|     C.write_manifest(__file__, [], out)
#|     print("\n" + "=" * 74)
#|     print("  The two panels carry the gate with them. An analysis that starts")
#|     print("  from panel_all_admitted.csv cannot skip it by accident.")
#|     print("=" * 74)
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/16_export_frozen_panels_and_gate_audit.py
#@@ FILE analysis/17_drop_formaldehyde_durations.py
#| #!/usr/bin/env python3
#| """
#| 17_drop_formaldehyde_durations.py -- what happens to skill and to attribution
#| when the formaldehyde durations are taken away.
#|
#| The leading variable of the model is the time a county spends above the
#| national 75th percentile of formaldehyde. Its near-twin is the time above the
#| 90th percentile (the two correlate at 0.99), and the annual means of the same
#| species rank far lower. This step refits the model four times on the same
#| folds: with both durations, without the first, without the second, and without
#| both. It reports the out-of-fold skill of each refit, which involves no
#| attribution, and where the attribution goes: to the other duration when one is
#| removed, and to the mean forms of formaldehyde when both are.
#|
#| Each refit is attributed on 3,000 county-years against 120 reference rows, the
#| same in every arm, so attributions are comparable between arms; their ranks
#| are not the ranks of the production ranking.
#|
#| READS   the modelling panel
#| WRITES  ../derived*/formaldehyde_drop.json
#|         ../derived*/checkpoints_pair_anatomy/   one file per refit
#| TIME    seconds from the stored refits; about 15 minutes per refit otherwise
#|
#|     python3 17_drop_formaldehyde_durations.py
#| """
#| from __future__ import annotations
#|
#| # ===========================================================================
#| #  SETTINGS
#| # ===========================================================================
#| PAIR = ("cams_fot_formaldehyde_above75th_percentile",
#|         "cams_fot_formaldehyde_above_90th_percentile")
#| # the mean forms of the same species, followed through the four refits
#| MEAN_FORMS = ("cams_formaldehyde_x1e9", "cams_total_column_formaldehyde_x1e9")
#| SHAP_ROWS = 3000             # county-years explained in each refit
#| SHAP_BACKGROUND = 120        # reference rows for the interventional value
#| # ===========================================================================
#|
#| import gc
#| import json
#| import os
#| import sys
#| import time
#|
#| import numpy as np
#| import pandas as pd
#|
#| sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#| import _config as C
#|
#| _BUDGET = {"left": 0, "limited": False}
#| _SMOKE = {"folds": 0}
#| OUT = C.DERIVED / "formaldehyde_drop.json"
#|
#|
#| def store():
#|     tag = f"_smoke{_SMOKE['folds']}" if _SMOKE["folds"] else ""
#|     d = C.DERIVED / f"checkpoints_pair_anatomy{tag}"
#|     d.mkdir(parents=True, exist_ok=True)
#|     return d
#|
#|
#| def arm(panel, drop, label, extra=None):
#|     """Refit without `drop`, optionally with an extra column, and attribute.
#|
#|     Returns out-of-fold skill and the mean absolute attribution of every
#|     column, so the attribution a removed variable used to hold can be followed
#|     to wherever it went.
#|     """
#|     import shap
#|     import xgboost as xgb
#|     from sklearn.metrics import r2_score, mean_absolute_error
#|
#|     ck = store() / f"{label}.json"
#|     if ck.exists():
#|         done = json.load(open(ck))
#|         print(f"    {label:<26} already done: R2 {done['r2']:.4f}")
#|         return done
#|
#|     if _BUDGET["limited"] and _BUDGET["left"] <= 0:
#|         raise SystemExit(
#|             f"\n  --max-arms reached before fitting '{label}'.\n"
#|             f"    Everything so far is checkpointed in {store()}\n"
#|             f"    Run the same command again to continue.")
#|     _BUDGET["left"] -= 1
#|
#|     X, y, groups = panel["X"], panel["y"], panel["groups"]
#|     cols = [c for c in X.columns if c not in set(drop)]
#|     Xa = X[cols]
#|     if extra is not None:
#|         Xa = pd.concat([Xa, extra], axis=1)
#|
#|     t0 = time.time()
#|     nf = _SMOKE["folds"] or C.N_FOLDS
#|     tag = f"pair_{label}" + (f"_smoke{nf}" if _SMOKE["folds"] else "")
#|     oof = C.cv_oof(Xa, y, groups, tag, n_folds=nf)
#|
#|     full = xgb.XGBRegressor(**C.XGB_PARAMS, n_jobs=C.threads())
#|     full.fit(Xa, y)
#|     rng = C.seed_everything()
#|     pick = rng.choice(len(y), size=min(SHAP_ROWS + SHAP_BACKGROUND, len(y)),
#|                       replace=False)
#|     # the background must not be part of what is explained, or a row acts as
#|     # its own reference
#|     bg, idx = np.sort(pick[:SHAP_BACKGROUND]), np.sort(pick[SHAP_BACKGROUND:])
#|     ex = shap.TreeExplainer(full, Xa.iloc[bg],
#|                             feature_perturbation="interventional")
#|     print(f"    {label:<26} attributing {len(idx):,} rows against "
#|           f"{len(bg)} reference rows", flush=True)
#|     sv = ex.shap_values(Xa.iloc[idx], check_additivity=False)
#|     mean_abs = np.abs(np.asarray(sv)).mean(axis=0)
#|     order = np.argsort(-mean_abs)
#|     attribution = {Xa.columns[k]: float(mean_abs[k]) for k in order}
#|     ranks = {c: i + 1 for i, c in enumerate(attribution)}
#|
#|     out = dict(label=label, dropped=list(drop),
#|                n_folds=int(_SMOKE["folds"] or C.N_FOLDS),
#|                smoke_test=bool(_SMOKE["folds"]),
#|                extra=list(extra.columns) if extra is not None else [],
#|                n_features=int(Xa.shape[1]),
#|                shap_rows=int(len(idx)), shap_background=int(len(bg)),
#|                r2=float(r2_score(y, oof)),
#|                mae=float(mean_absolute_error(y, oof)),
#|                attribution=attribution, ranks=ranks,
#|                seconds=round(time.time() - t0, 1))
#|     json.dump(out, open(ck, "w"), indent=1)
#|     print(f"    {label:<26} R2 {out['r2']:.4f}  [checkpointed]", flush=True)
#|     del full, sv
#|     gc.collect()
#|     return out
#|
#|
#| def show(a, b, keys):
#|     """A compact comparison of two arms on named columns."""
#|     for k in keys:
#|         va = a["attribution"].get(k)
#|         vb = b["attribution"].get(k)
#|         ra = a["ranks"].get(k)
#|         rb = b["ranks"].get(k)
#|         fa = "removed" if va is None else f"{va:.4f} (#{ra})"
#|         fb = "removed" if vb is None else f"{vb:.4f} (#{rb})"
#|         print(f"      {k[:52]:<52} {fa:>18}  ->  {fb:>18}")
#|
#|
#| def removal_and_inheritance(panel, args):
#|     print("\n" + "=" * 74)
#|     print("  1 and 2. REMOVAL AND INHERITANCE")
#|     print("=" * 74)
#|     lo_col, hi_col = PAIR
#|     X = panel["X"]
#|     for c in PAIR:
#|         if c not in X.columns:
#|             raise SystemExit(f"{c} is not in the modelled set.")
#|     if X[lo_col].mean() <= X[hi_col].mean():
#|         raise SystemExit(
#|             f"{lo_col} should spend more time above its threshold than "
#|             f"{hi_col}; the pair is named the wrong way round in SETTINGS.")
#|
#|     r = np.corrcoef(X[lo_col], X[hi_col])[0, 1]
#|     print(f"  the pair correlates at {r:.4f}")
#|
#|     both = arm(panel, [], "both_present")
#|     no_lo = arm(panel, [lo_col], "without_p75")
#|     no_hi = arm(panel, [hi_col], "without_p90")
#|     neither = arm(panel, list(PAIR), "without_both")
#|
#|     print("\n  out of fold:")
#|     for k, v in (("both present", both), ("without the 75th", no_lo),
#|                  ("without the 90th", no_hi), ("without both", neither)):
#|         print(f"      {k:<20} R2 {v['r2']:.4f}   MAE {v['mae']:.4f}   "
#|               f"{v['n_features']} columns")
#|
#|     print("\n  where the attribution went when the 75th was removed:")
#|     show(both, no_lo, [hi_col])
#|     print("\n  where the attribution went when the 90th was removed:")
#|     show(both, no_hi, [lo_col])
#|
#|     a_lo = both["attribution"][lo_col]
#|     a_hi = both["attribution"][hi_col]
#|     inherit_lo = (no_lo["attribution"][hi_col] - a_hi) / a_lo
#|     inherit_hi = (no_hi["attribution"][lo_col] - a_lo) / a_hi
#|     print(f"\n  the 90th inherits {100 * inherit_lo:.1f} per cent of what the "
#|           f"75th was carrying")
#|     print(f"  the 75th inherits {100 * inherit_hi:.1f} per cent of what the "
#|           f"90th was carrying")
#|     print(f"  skill cost of removing the 75th   {both['r2'] - no_lo['r2']:+.4f}")
#|     print(f"  skill cost of removing the 90th   {both['r2'] - no_hi['r2']:+.4f}")
#|     print(f"  skill cost of removing both       "
#|           f"{both['r2'] - neither['r2']:+.4f}")
#|     print("\n  Reading: if the pair were one measurement made twice, the")
#|     print("  survivor would take up most of the removed attribution and the")
#|     print("  skill cost would be near zero. A survivor that takes up little,")
#|     print("  with a skill cost larger than either column removed alone, says")
#|     print("  the two carry different information about the outcome.")
#|
#|     return dict(correlation=float(r), both=both, without_p75=no_lo,
#|                 without_p90=no_hi, without_both=neither,
#|                 inherit_p90_of_p75=float(inherit_lo),
#|                 inherit_p75_of_p90=float(inherit_hi))
#|
#|
#| def main():
#|     C.banner("17  refits without the formaldehyde durations")
#|     C.seed_everything()
#|     panel = C.load_panel()
#|     r = removal_and_inheritance(panel, None)
#|     arms = ("both", "without_p75", "without_p90", "without_both")
#|     out = dict(
#|         correlation=r["correlation"],
#|         shap_rows=int(r["both"]["shap_rows"]),
#|         shap_background=int(r["both"]["shap_background"]),
#|         r2={a: r[a]["r2"] for a in arms},
#|         mae={a: r[a]["mae"] for a in arms},
#|         inherit_p90_of_p75=r["inherit_p90_of_p75"],
#|         inherit_p75_of_p90=r["inherit_p75_of_p90"],
#|         rank={c: {a: r[a]["ranks"].get(c) for a in arms}
#|               for c in PAIR + MEAN_FORMS},
#|         attribution={c: {a: r[a]["attribution"].get(c) for a in arms}
#|                      for c in PAIR + MEAN_FORMS})
#|     print("\n  the mean forms of formaldehyde, with both durations and with neither:")
#|     for c in MEAN_FORMS:
#|         print(f"      {c:<40} rank {out['rank'][c]['both']:>4} -> "
#|               f"{out['rank'][c]['without_both']:>4}")
#|     json.dump(out, open(OUT, "w"), indent=1)
#|     print(f"  [saved] {OUT}")
#|     C.write_manifest(__file__, [C.FRAME, C.MODEL_A], [OUT],
#|                      settings=dict(refit_shap_rows=SHAP_ROWS,
#|                                    refit_shap_background=SHAP_BACKGROUND))
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/17_drop_formaldehyde_durations.py
#@@ FILE analysis/19_temporal_validation.py
#| #!/usr/bin/env python3
#| """
#| 19_temporal_validation.py -- can the model predict a year it has not seen?
#|
#| READS   the modelling panel, through _config.load_panel, so the admission gate
#|         and both pruning filters are inherited and nothing here sees a column
#|         the published model did not see
#| WRITES  ../derived*/temporal_validation.json
#|         ../derived*/temporal_validation_folds.csv
#|
#| THE OBJECTION
#| -------------
#|     "Every fold of yours contains the year it predicts. The folds are grouped
#|      by county, which stops a county appearing on both sides, but every year
#|      appears on both sides. This is interpolation in time dressed as
#|      prediction."
#|
#| That is correct about the production protocol, and the production protocol is
#| the right one for the question the paper asks, which is about variation across
#| counties. It is the wrong protocol for a different question, and a referee is
#| entitled to ask the different question. Three answers:
#|
#|   1. LEAVE ONE YEAR OUT. Train on every other year, predict the held-out one.
#|      The same counties appear on both sides, deliberately: this isolates
#|      extrapolation in time from extrapolation across counties, which the state
#|      holdout of script 13 already measures.
#|
#|   2. FORWARD CHAINING. Train on everything up to year t, predict t + 1, and
#|      walk forward. Nothing after the prediction year is ever in the training
#|      set, which is the only protocol that answers "could this have been used
#|      at the time".
#|
#|   3. ERA SPLIT. Split the window in half and ask whether the paper's central
#|      count -- how often the non-central summary of a field carries more
#|      information than the central one -- is the same in both halves. This one
#|      is model-free and takes seconds, by design: if the finding were an
#|      artefact of one period it would show here before any model is fitted.
#|
#| WHAT HONEST LOOKS LIKE HERE
#| ---------------------------
#| Forward chaining on a seventeen-year window with a slowly moving outcome is a
#| hard test and the number will be lower than the headline. That is the point. A
#| score reported under a protocol that nobody would use is worth less than a
#| lower score reported under one that answers a question.
#|
#|     python3 19_temporal_validation.py
#|     python3 19_temporal_validation.py --only loyo
#|     python3 19_temporal_validation.py --max-fits 4      # check the code path
#| """
#| from __future__ import annotations
#|
#| import argparse
#| import gc
#| import json
#| import os
#| import sys
#| import time
#| from pathlib import Path
#|
#| import numpy as np
#| import pandas as pd
#|
#| HERE = os.path.dirname(os.path.abspath(__file__))
#| if HERE not in sys.path:
#|     sys.path.insert(0, HERE)
#|
#| import _config as C                                            # noqa: E402
#|
#| OUT_JSON = C.DERIVED / "temporal_validation.json"
#| OUT_CSV = C.DERIVED / "temporal_validation_folds.csv"
#| STORE = C.DERIVED / "checkpoints_temporal"
#|
#| # ===========================================================================
#| #  SETTINGS
#| # ===========================================================================
#| MIN_TRAIN_YEARS = 3        # forward chaining starts once this many years are
#|                            # available to train on. Two years cannot carry a
#|                            # 1,200-tree model and the fit would say nothing.
#| ERA_MI_SAMPLE = 20000      # rows per era for the mutual-information census.
#|                            # The estimator is a k-nearest-neighbour one and its
#|                            # cost is superlinear in n.
#| ERA_MI_NEIGHBOURS = 3      # the default of mutual_info_regression, named here
#|                            # because it moves the numbers
#|
#| _BUDGET = {"left": 0, "limited": False}
#|
#|
#| def _spend():
#|     if not _BUDGET["limited"]:
#|         return True
#|     if _BUDGET["left"] <= 0:
#|         return False
#|     _BUDGET["left"] -= 1
#|     return True
#|
#|
#| # ---------------------------------------------------------------------------
#| #  One fit, checkpointed
#| # ---------------------------------------------------------------------------
#| def fit_predict(X, y, train, test, tag):
#|     """Predictions for `test` from a model fitted on `train`, cached by tag."""
#|     import xgboost as xgb
#|     STORE.mkdir(parents=True, exist_ok=True)
#|     f = STORE / f"{tag}.npy"
#|     if f.exists():
#|         got = np.load(f)
#|         if len(got) == len(test):
#|             print(f"      {tag:<28} from checkpoint", flush=True)
#|             return got
#|         print(f"      {tag:<28} checkpoint has {len(got)} rows, needs "
#|               f"{len(test)}; refitting", flush=True)
#|     if not _spend():
#|         print(f"      {tag:<28} not fitted: fit budget spent", flush=True)
#|         return None
#|     t0 = time.time()
#|     m = xgb.XGBRegressor(**C.XGB_PARAMS, n_jobs=C.threads())
#|     m.fit(X.iloc[train], y[train])
#|     pred = m.predict(X.iloc[test])
#|     np.save(f, pred)
#|     del m
#|     gc.collect()
#|     print(f"      {tag:<28} fitted in {(time.time() - t0) / 60:.1f} min",
#|           flush=True)
#|     return pred
#|
#|
#| def score(y_true, pred):
#|     from sklearn.metrics import r2_score, mean_absolute_error
#|     return dict(r2=float(r2_score(y_true, pred)),
#|                 mae=float(mean_absolute_error(y_true, pred)),
#|                 n=int(len(y_true)))
#|
#|
#| # ---------------------------------------------------------------------------
#| #  1. Leave one year out
#| # ---------------------------------------------------------------------------
#| def leave_one_year_out(p):
#|     print("\n" + "=" * 74)
#|     print("  1. LEAVE ONE YEAR OUT")
#|     print("=" * 74)
#|     print("  Train on every other year, predict the held-out one. The same")
#|     print("  counties are on both sides on purpose: this isolates")
#|     print("  extrapolation in time. Extrapolation across place is script 13.")
#|     print()
#|     X, y = p["X"], p["y"]
#|     years = p["df"][C.YEAR].to_numpy()
#|     rows, preds = [], np.full(len(y), np.nan)
#|     for yr in sorted(np.unique(years)):
#|         te = np.where(years == yr)[0]
#|         tr = np.where(years != yr)[0]
#|         pr = fit_predict(X, y, tr, te, f"loyo_{int(yr)}")
#|         if pr is None:
#|             continue
#|         preds[te] = pr
#|         s = score(y[te], pr)
#|         s.update(arm="leave_one_year_out", held_out=int(yr),
#|                  n_train=int(len(tr)))
#|         rows.append(s)
#|         print(f"    {int(yr)}   n {s['n']:>6}   R2 {s['r2']:+.4f}   "
#|               f"MAE {s['mae']:.4f}")
#|     if not rows:
#|         return dict(folds=[], note="no fold completed"), rows
#|     done = ~np.isnan(preds)
#|     pooled = score(y[done], preds[done]) if done.any() else {}
#|     r2s = [r["r2"] for r in rows]
#|     out = dict(n_years=len(rows), pooled=pooled,
#|                r2_mean=float(np.mean(r2s)), r2_min=float(np.min(r2s)),
#|                r2_max=float(np.max(r2s)),
#|                worst_year=int(rows[int(np.argmin(r2s))]["held_out"]),
#|                folds=rows)
#|     print(f"\n    pooled over the years that ran: R2 {pooled.get('r2', float('nan')):+.4f}"
#|           f"   MAE {pooled.get('mae', float('nan')):.4f}")
#|     print(f"    per-year R2 spans {out['r2_min']:+.4f} to {out['r2_max']:+.4f}"
#|           f"   (worst year {out['worst_year']})")
#|     return out, rows
#|
#|
#| # ---------------------------------------------------------------------------
#| #  2. Forward chaining
#| # ---------------------------------------------------------------------------
#| def forward_chaining(p):
#|     print("\n" + "=" * 74)
#|     print("  2. FORWARD CHAINING")
#|     print("=" * 74)
#|     print("  Train on everything up to year t, predict t + 1. Nothing after")
#|     print("  the prediction year is ever in the training set. This is the only")
#|     print("  protocol here that answers whether the model could have been")
#|     print("  used at the time, and it is the hardest of the three.")
#|     print()
#|     X, y = p["X"], p["y"]
#|     years = p["df"][C.YEAR].to_numpy()
#|     uniq = sorted(np.unique(years))
#|     rows, preds = [], np.full(len(y), np.nan)
#|     for i in range(MIN_TRAIN_YEARS, len(uniq)):
#|         target = uniq[i]
#|         tr = np.where(years < target)[0]
#|         te = np.where(years == target)[0]
#|         pr = fit_predict(X, y, tr, te, f"fwd_{int(target)}")
#|         if pr is None:
#|             continue
#|         preds[te] = pr
#|         s = score(y[te], pr)
#|         s.update(arm="forward_chaining", held_out=int(target),
#|                  n_train=int(len(tr)), train_years=len(uniq[:i]))
#|         rows.append(s)
#|         print(f"    train <= {int(uniq[i - 1])}  predict {int(target)}   "
#|               f"n {s['n']:>6}   R2 {s['r2']:+.4f}   MAE {s['mae']:.4f}")
#|     if not rows:
#|         return dict(folds=[], note="no fold completed"), rows
#|     done = ~np.isnan(preds)
#|     pooled = score(y[done], preds[done])
#|     r2s = [r["r2"] for r in rows]
#|     # does it improve as the training window grows? A positive slope says the
#|     # model is still data-limited in time rather than drifting.
#|     slope = float(np.polyfit([r["train_years"] for r in rows], r2s, 1)[0]) \
#|         if len(rows) > 2 else float("nan")
#|     out = dict(n_steps=len(rows), pooled=pooled,
#|                r2_first=float(r2s[0]), r2_last=float(r2s[-1]),
#|                r2_mean=float(np.mean(r2s)),
#|                slope_r2_per_train_year=slope, folds=rows)
#|     print(f"\n    pooled over the steps that ran: R2 {pooled['r2']:+.4f}   "
#|           f"MAE {pooled['mae']:.4f}")
#|     print(f"    first step R2 {out['r2_first']:+.4f}, last {out['r2_last']:+.4f}"
#|           f", trend {slope:+.4f} per extra training year")
#|     return out, rows
#|
#|
#| # ---------------------------------------------------------------------------
#| #  3. Era split, model-free
#| # ---------------------------------------------------------------------------
#| def era_split(p):
#|     """Is the central count the same in the first half of the window and the
#|     second? Mutual information, so no model is fitted and no attribution
#|     method is involved."""
#|     from sklearn.feature_selection import mutual_info_regression
#|     print("\n" + "=" * 74)
#|     print("  3. ERA SPLIT, WITH NO MODEL IN IT")
#|     print("=" * 74)
#|     print("  The paper's central count, recomputed inside each half of the")
#|     print("  window using mutual information alone. If the finding belonged to")
#|     print("  one period rather than to the data, it would show here, before")
#|     print("  any model is fitted and without any attribution method.")
#|     print()
#|     df, X = p["df"], p["X"]
#|     years = df[C.YEAR].to_numpy()
#|     mid = float(np.median(years))
#|     eras = {"early": years <= mid, "late": years > mid}
#|     print(f"    window split at {mid:g}: "
#|           + ", ".join(f"{k} n={int(v.sum())}" for k, v in eras.items()))
#|
#|     pairs = C.census_pairs(columns=X.columns)
#|     print(f"    {len(pairs)} fields have a central and a non-central column "
#|           f"in the model")
#|
#|     rng = C.seed_everything(extra=19)
#|     rows, per_era = [], {}
#|     for era, mask in eras.items():
#|         idx = np.where(mask)[0]
#|         if len(idx) > ERA_MI_SAMPLE:
#|             idx = rng.choice(idx, ERA_MI_SAMPLE, replace=False)
#|         yy = p["y"][idx]
#|         wins, got = 0, 0
#|         for fieldname, c_col, n_col in pairs:
#|             sub = X.iloc[idx][[c_col, n_col]].to_numpy()
#|             mi = mutual_info_regression(
#|                 sub, yy, n_neighbors=ERA_MI_NEIGHBOURS,
#|                 random_state=C.SEED)
#|             got += 1
#|             win = bool(mi[1] > mi[0])
#|             wins += int(win)
#|             rows.append(dict(arm="era_split", era=era, field=fieldname,
#|                              central_mi=float(mi[0]), noncentral_mi=float(mi[1]),
#|                              noncentral_wins=win))
#|         per_era[era] = dict(n_rows=int(len(idx)), n_fields=got,
#|                             noncentral_wins=wins,
#|                             share=float(wins / got) if got else float("nan"))
#|         print(f"    {era:<6} n {len(idx):>6}   non-central carries more "
#|               f"information in {wins} of {got}")
#|
#|     a, b = per_era.get("early", {}), per_era.get("late", {})
#|     out = dict(split_at=mid, eras=per_era)
#|     if a and b:
#|         # Two independent counts of the same quantity. A two-sided Fisher
#|         # exact test on the 2x2 of (era) by (which description wins) says
#|         # whether the two halves disagree by more than sampling.
#|         from scipy.stats import fisher_exact
#|         table = [[a["noncentral_wins"], a["n_fields"] - a["noncentral_wins"]],
#|                  [b["noncentral_wins"], b["n_fields"] - b["noncentral_wins"]]]
#|         odds, pv = fisher_exact(table)
#|         out.update(fisher_odds=float(odds), fisher_p=float(pv),
#|                    share_difference=float(a["share"] - b["share"]))
#|         print(f"\n    difference between halves {out['share_difference']:+.3f}"
#|               f"   Fisher exact p = {pv:.4f}")
#|         print("    A large p here is the result worth having: it says the two")
#|         print("    halves of the window do not disagree about which")
#|         print("    description carries the information.")
#|     return out, rows
#|
#|
#| # ---------------------------------------------------------------------------
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--only", default="all",
#|                     help="comma separated: loyo, forward, era")
#|     ap.add_argument("--max-fits", type=int, default=0,
#|                     help="stop after this many models are FITTED in this run. "
#|                          "Checkpointed fits from an earlier run do not count, "
#|                          "so the same command run again continues.")
#|     a = ap.parse_args()
#|     want = (["loyo", "forward", "era"] if a.only == "all"
#|             else [s.strip() for s in a.only.split(",")])
#|     if a.max_fits:
#|         _BUDGET["left"], _BUDGET["limited"] = a.max_fits, True
#|
#|     C.banner("19  can the model predict a year it has not seen")
#|     C.seed_everything()
#|     C.DERIVED.mkdir(parents=True, exist_ok=True)
#|     p = C.load_panel()
#|     lo, hi = p["window"]
#|     print(f"  panel {p['X'].shape[0]:,} rows x {p['X'].shape[1]} columns, "
#|           f"years {lo} to {hi}")
#|
#|     blob, rows = {}, []
#|     if "loyo" in want:
#|         out, r = leave_one_year_out(p)
#|         blob["leave_one_year_out"], rows = out, rows + r
#|     if "forward" in want:
#|         out, r = forward_chaining(p)
#|         blob["forward_chaining"], rows = out, rows + r
#|     if "era" in want:
#|         out, r = era_split(p)
#|         blob["era_split"], rows = out, rows + r
#|
#|     # keep whatever an earlier partial run wrote
#|     if OUT_JSON.exists():
#|         try:
#|             old = json.load(open(OUT_JSON))
#|             for k, v in old.items():
#|                 blob.setdefault(k, v)
#|         except Exception:
#|             pass
#|     blob["window"] = [int(lo), int(hi)]
#|     blob["n_rows"] = int(len(p["y"]))
#|     blob["n_features"] = int(p["X"].shape[1])
#|     json.dump(blob, open(OUT_JSON, "w"), indent=1, default=str)
#|     print(f"\n  [saved] {OUT_JSON}")
#|     if rows:
#|         new = pd.DataFrame(rows)
#|         if OUT_CSV.exists():
#|             try:
#|                 prev = pd.read_csv(OUT_CSV)
#|                 keys = [k for k in ("arm", "held_out", "era", "field")
#|                         if k in new.columns and k in prev.columns]
#|                 new = (pd.concat([prev, new], ignore_index=True)
#|                        .drop_duplicates(subset=keys, keep="last"))
#|             except Exception:
#|                 pass
#|         new.to_csv(OUT_CSV, index=False)
#|         print(f"  [saved] {OUT_CSV}")
#|
#|     print("\n" + "-" * 74)
#|     print("  HOW TO READ THESE THREE NUMBERS TOGETHER")
#|     print("-" * 74)
#|     print("  The headline score is county-grouped and answers a question about")
#|     print("  variation across place. Leave-one-year-out answers a question")
#|     print("  about time with place held constant. Forward chaining answers")
#|     print("  both at once and under the hardest constraint, and is expected to")
#|     print("  be the lowest of the three. None of them is the others' check:")
#|     print("  they are three different questions and the paper should say which")
#|     print("  one each number answers.")
#|
#|     C.write_manifest("19_temporal_validation.py",
#|                      inputs=[C.FRAME, C.MODEL_A],
#|                      outputs=[OUT_JSON, OUT_CSV],
#|                      settings=dict(MIN_TRAIN_YEARS=MIN_TRAIN_YEARS,
#|                                    ERA_MI_SAMPLE=ERA_MI_SAMPLE,
#|                                    ERA_MI_NEIGHBOURS=ERA_MI_NEIGHBOURS,
#|                                    arms=want),
#|                      notes="Temporal validation: leave one year out, forward "
#|                            "chaining, and a model-free era split of the "
#|                            "central count.")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/19_temporal_validation.py
#@@ FILE analysis/20_null_skill_and_null_crossing.py
#| #!/usr/bin/env python3
#| """
#| 20_null_skill_and_null_crossing.py -- what this machinery returns when there
#| is nothing to find.
#|
#|     --only skill
#|     Permute the outcome three ways (across all rows, within each county,
#|     between whole counties) and rerun the production protocol unchanged. Each
#|     permutation destroys something different, so the three numbers together
#|     say where the explained variance lives (Section 3.1, Figure 2D).
#|
#|     --only crossing
#|     Run the whole turning-point estimator, bootstrap and stability rule
#|     included, on attribution permuted against its own predictor, and on curves
#|     raised until no crossing exists. Report how often a crossing is found and
#|     how often it passes the rule (Section 3.5, Figure 7B).
#|
#| Nothing in this script invents a value. A permutation reorders real numbers,
#| and no permuted column is written to any exported panel.
#|
#| This is the final-paper copy of 20_null_calibration_and_multiplicity.py. Its
#| functions are unchanged; the census null and the multiplicity family, which
#| the final paper no longer reports, are not included.
#|
#| READS   the modelling panel; the attribution surface of script 05
#| WRITES  ../derived*/null_skill.json
#|         ../derived*/null_crossing.json
#|
#|     python3 20_null_skill_and_null_crossing.py
#|     python3 20_null_skill_and_null_crossing.py --only crossing
#| """
#| from __future__ import annotations
#|
#| import argparse
#| import importlib.util
#| import json
#| import os
#| import sys
#|
#| import numpy as np
#|
#| HERE = os.path.dirname(os.path.abspath(__file__))
#| if HERE not in sys.path:
#|     sys.path.insert(0, HERE)
#|
#| import _config as C                                            # noqa: E402
#|
#| SKILL_JSON = C.DERIVED / "null_skill.json"
#| CROSS_JSON = C.DERIVED / "null_crossing.json"
#|
#| # ===========================================================================
#| #  SETTINGS
#| # ===========================================================================
#| NULL_CROSS_DRAWS = 40       # permutations of the attribution per variable.
#| NULL_CROSS_BOOT = 120       # bootstrap draws inside each permutation; a null
#|                             # needs only enough to decide whether the rule
#|                             # would pass. Checkpointed per variable.
#|
#|
#| def canonical_crossing():
#|     """Script 06's estimator, imported rather than reimplemented.
#|
#|     A second implementation would give crossings differing in the second
#|     decimal from the published table and no reader could tell whether the
#|     difference was the null or the code.
#|     """
#|     path = os.path.join(HERE, "06_bootstrap_turning_points_and_gate.py")
#|     if not os.path.exists(path):
#|         raise SystemExit(
#|             "06_bootstrap_turning_points_and_gate.py is not beside this file.\n"
#|             "    This script runs that script's estimator on permuted\n"
#|             "    attribution, so the null is comparable with the published\n"
#|             "    table. Copy the whole analysis folder, not single scripts.")
#|     spec = importlib.util.spec_from_file_location("_s06", path)
#|     mod = importlib.util.module_from_spec(spec)
#|     spec.loader.exec_module(mod)
#|     return mod
#|
#|
#| # ---------------------------------------------------------------------------
#| #  1. What the protocol returns when the outcome carries no signal
#| # ---------------------------------------------------------------------------
#| def null_skill(p, args):
#|     print("\n" + "=" * 74)
#|     print("  1. NULL CALIBRATION OF THE SCORE")
#|     print("=" * 74)
#|     print("  The production protocol, unchanged, on a permuted outcome. Three")
#|     print("  permutations, each destroying something different:")
#|     print()
#|     print("    full          every row's outcome reassigned at random. Nothing")
#|     print("                  survives. This is the true null.")
#|     print("    within county the years of each county reordered among")
#|     print("                  themselves. The county mean is preserved exactly,")
#|     print("                  the year-to-year pattern is destroyed.")
#|     print("    between county whole counties' outcome series swapped between")
#|     print("                  counties. The within-county pattern is preserved,")
#|     print("                  the link between a county's predictors and its")
#|     print("                  own level is destroyed.")
#|     print()
#|     print("  The three answers locate the explained variance. A high score")
#|     print("  under the within-county permutation and a null score under the")
#|     print("  between-county one would say the model is predicting the county's")
#|     print("  level and not its year-to-year movement, which is worth knowing")
#|     print("  and is not a flaw as long as the paper says so.")
#|     print()
#|     X, y, groups = p["X"], p["y"], p["groups"]
#|     rng = C.seed_everything(extra=20)
#|     arms = {}
#|
#|     # full
#|     perm = rng.permutation(len(y))
#|     arms["full"] = y[perm]
#|
#|     # within county: reorder each county's own values
#|     wy = y.copy()
#|     starts = {}
#|     for i, g in enumerate(groups):
#|         starts.setdefault(g, []).append(i)
#|     for g, idx in starts.items():
#|         idx = np.asarray(idx)
#|         wy[idx] = y[rng.permutation(idx)]
#|     arms["within_county"] = wy
#|
#|     # between county: give each county another county's series, aligned by
#|     # position so a county with fewer years is not padded
#|     by = y.copy()
#|     keys = list(starts.keys())
#|     shuffled = list(rng.permutation(len(keys)))
#|     for a_i, b_i in enumerate(shuffled):
#|         ia, ib = np.asarray(starts[keys[a_i]]), np.asarray(starts[keys[b_i]])
#|         n = min(len(ia), len(ib))
#|         by[ia[:n]] = y[ib[:n]]
#|     arms["between_county"] = by
#|
#|     out = {}
#|     for name, yy in arms.items():
#|         if args.only_arm and name != args.only_arm:
#|             continue
#|         tag = f"null_skill_{name}_seed{C.SEED + 20}"
#|         s = C.cv_score(X, np.asarray(yy, float), groups, tag,
#|                        n_folds=args.null_folds or None)
#|         out[name] = s
#|         print(f"    {name:<16} R2 {s['r2']:+.4f}   MAE {s['mae']:.4f}   "
#|               f"({s['n_features']} columns)")
#|     out["note"] = ("Permuted outcomes. These are null distributions and are "
#|                    "not results about life expectancy.")
#|     out["n_folds"] = int(args.null_folds or C.N_FOLDS)
#|     out["seed_offset"] = 20
#|     # the published number, for the comparison the referee wants in one place
#|     try:
#|         pub = C.load_result(C.DERIVED / "baselines.json", ("satellite_model",))
#|         out["published_for_comparison"] = dict(
#|             satellite_model=pub.get("satellite_model"),
#|             naive_mean=pub.get("naive_mean"))
#|     except SystemExit:
#|         print("    (baselines.json absent; run script 03 for the published "
#|               "score to sit beside these)")
#|     print("\n    The published county-grouped score sits beside these in the")
#|     print("    file, so the comparison does not depend on anyone remembering")
#|     print("    it.")
#|     return out
#|
#|
#| # ---------------------------------------------------------------------------
#| #  2. Does the estimator find thresholds in noise?
#| # ---------------------------------------------------------------------------
#| def null_crossing(args):
#|     print("\n" + "=" * 74)
#|     print("  2. DOES THE CROSSING ESTIMATOR FIND THRESHOLDS IN NOISE?")
#|     print("=" * 74)
#|     print("  Two nulls, both using script 06's own estimator, bootstrap and")
#|     print("  stability gate:")
#|     print()
#|     print("    permuted     the attribution values reordered against the")
#|     print("                 predictor values. Any relation is destroyed; the")
#|     print("                 marginal distribution of both is untouched.")
#|     print("    shifted      the real attribution curve raised until it is")
#|     print("                 everywhere positive. There is no zero crossing to")
#|     print("                 find, by construction. An estimator that reports")
#|     print("                 one here is broken.")
#|     print()
#|     s06 = canonical_crossing()
#|     dep, feats = C.dependence_table()
#|     print(f"    {len(feats)} candidates in the attribution surface, "
#|           f"{len(dep):,} rows each")
#|
#|     published = None
#|     try:
#|         published = C.load_table("turning_points").set_index("feature")
#|     except SystemExit:
#|         print("    (turning_points.csv absent, so the published verdict is "
#|               "not shown beside the null)")
#|
#|     rng = C.seed_everything(extra=21)
#|     rows, per_var = [], {}
#|     n_draws_want = args.null_draws or NULL_CROSS_DRAWS
#|     n_boot_want = args.null_boot or NULL_CROSS_BOOT
#|     store = C.DERIVED / "checkpoints_null_crossing"
#|     store.mkdir(parents=True, exist_ok=True)
#|     stamp = f"d{n_draws_want}_b{n_boot_want}"
#|     for feat in feats:
#|         ck = store / f"{feat}__{stamp}.json"
#|         if ck.exists():
#|             try:
#|                 rec = json.load(open(ck))
#|                 per_var[str(feat)] = rec
#|                 rows.append(rec)
#|                 print(f"    {str(feat)[:44]:<44} from checkpoint: gate passed "
#|                       f"in {rec['permuted_gate_passed']}/{rec['n_draws']}")
#|                 continue
#|             except Exception:
#|                 pass
#|         x = dep[feat].to_numpy(float)
#|         v = dep["shap_" + feat].to_numpy(float)
#|         ok = np.isfinite(x) & np.isfinite(v)
#|         x, v = x[ok], v[ok]
#|         if len(x) < 2000:
#|             continue
#|         found_p, passed_p, points = 0, 0, []
#|         for d in range(args.null_draws or NULL_CROSS_DRAWS):
#|             r = s06.analyse(str(feat), x, v[rng.permutation(len(v))],
#|                             args.null_boot or NULL_CROSS_BOOT, C.N_BINS, rng)
#|             if r["point_estimate"] is not None and \
#|                     np.isfinite(r["point_estimate"]):
#|                 found_p += 1
#|                 points.append(float(r["point_estimate"]))
#|             if s06.passes(r, C.GATE_MIN_FOUND, C.GATE_MAX_WIDTH,
#|                           C.GATE_INSIDE):
#|                 passed_p += 1
#|         # shifted: no crossing exists
#|         shifted = v - v.min() + abs(v).max() * 0.05
#|         rs = s06.analyse(str(feat), x, shifted,
#|                          args.null_boot or NULL_CROSS_BOOT, C.N_BINS, rng)
#|         shift_found = bool(rs["point_estimate"] is not None
#|                            and np.isfinite(rs["point_estimate"]))
#|         shift_passed = bool(s06.passes(rs, C.GATE_MIN_FOUND,
#|                                        C.GATE_MAX_WIDTH, C.GATE_INSIDE))
#|         n_draws = args.null_draws or NULL_CROSS_DRAWS
#|         rec = dict(feature=str(feat), n_draws=n_draws,
#|                    permuted_crossing_found=found_p,
#|                    permuted_gate_passed=passed_p,
#|                    permuted_pass_rate=passed_p / n_draws,
#|                    shifted_crossing_found=shift_found,
#|                    shifted_gate_passed=shift_passed)
#|         if points:
#|             rec["permuted_point_spread"] = float(np.std(points))
#|         if published is not None and feat in published.index:
#|             rec["published_stable"] = bool(published.loc[feat, "stable"])
#|             rec["published_point"] = float(
#|                 published.loc[feat, "point_estimate"])
#|         per_var[str(feat)] = rec
#|         rows.append(rec)
#|         json.dump(rec, open(ck, "w"), indent=1, default=str)
#|         print(f"    {str(feat)[:44]:<44} crossing located in "
#|               f"{found_p:>3}/{n_draws}, gate passed in {passed_p:>3}/{n_draws}"
#|               + ("   shifted curve: FOUND ONE" if shift_passed else ""))
#|
#|     n_draws = args.null_draws or NULL_CROSS_DRAWS
#|     total_pass = sum(r["permuted_gate_passed"] for r in rows)
#|     total_draw = sum(r["n_draws"] for r in rows)
#|     overall = dict(
#|         n_variables=len(rows), n_draws_each=n_draws,
#|         total_permutations=total_draw,
#|         total_gate_passes=total_pass,
#|         pass_rate=float(total_pass / total_draw) if total_draw else float("nan"),
#|         shifted_any_pass=int(sum(1 for r in rows if r["shifted_gate_passed"])),
#|         n_boot_per_draw=int(args.null_boot or NULL_CROSS_BOOT))
#|     print(f"\n    across {len(rows)} variables and {total_draw} permutations, "
#|           f"the gate passed {total_pass} times "
#|           f"({100 * overall['pass_rate']:.2f} per cent)")
#|     print(f"    on the shifted curves, where no crossing exists, the gate "
#|           f"passed for {overall['shifted_any_pass']} of {len(rows)} variables")
#|     print()
#|     print("    This is the number to quote when the estimator is challenged.")
#|     print("    The bootstrap below the gate is run at "
#|           f"{overall['n_boot_per_draw']} draws rather than")
#|     print(f"    the paper's {C.N_BOOT}, which is stated in the result file: a")
#|     print("    null needs only enough resolution to decide pass or fail.")
#|     return dict(per_variable=per_var, overall=overall,
#|                 note="Permuted and shifted attribution. Null distributions, "
#|                      "not results."), rows
#|
#|
#| # ---------------------------------------------------------------------------
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--only", default="all",
#|                     help="comma separated: skill, crossing")
#|     ap.add_argument("--only-arm", default="",
#|                     help="with --only skill, one permutation: full, "
#|                          "within_county, between_county")
#|     ap.add_argument("--null-folds", type=int, default=0,
#|                     help="folds for the null refits. Default is the paper's "
#|                          f"{C.N_FOLDS}; a smaller number is a smoke test and "
#|                          "is written into the result file.")
#|     ap.add_argument("--null-draws", type=int, default=0,
#|                     help="permutations per variable (crossing)")
#|     ap.add_argument("--null-boot", type=int, default=0,
#|                     help="bootstrap draws inside each permutation")
#|     a = ap.parse_args()
#|     want = (["skill", "crossing"] if a.only == "all"
#|             else [s.strip() for s in a.only.split(",")])
#|
#|     C.banner("20  what this machinery returns when there is nothing to find")
#|     C.seed_everything()
#|     C.DERIVED.mkdir(parents=True, exist_ok=True)
#|     written = []
#|
#|     if "skill" in want:
#|         p = C.load_panel()
#|         print(f"  panel {p['X'].shape[0]:,} rows x {p['X'].shape[1]} columns")
#|         out = null_skill(p, a)
#|         json.dump(out, open(SKILL_JSON, "w"), indent=1, default=str)
#|         print(f"  [saved] {SKILL_JSON}")
#|         written.append(SKILL_JSON)
#|
#|     if "crossing" in want:
#|         out, rows = null_crossing(a)
#|         json.dump(out, open(CROSS_JSON, "w"), indent=1, default=str)
#|         print(f"  [saved] {CROSS_JSON}")
#|         written.append(CROSS_JSON)
#|
#|     C.write_manifest("20_null_skill_and_null_crossing.py",
#|                      inputs=[C.FRAME, C.MODEL_A],
#|                      outputs=written,
#|                      settings=dict(arms=want,
#|                                    NULL_CROSS_DRAWS=a.null_draws or NULL_CROSS_DRAWS,
#|                                    NULL_CROSS_BOOT=a.null_boot or NULL_CROSS_BOOT,
#|                                    null_folds=a.null_folds or C.N_FOLDS),
#|                      notes="Null calibration. Every number here is a null "
#|                            "distribution, not a result about life expectancy.")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/20_null_skill_and_null_crossing.py
#@@ FILE analysis/22_modality_arms.py
#| #!/usr/bin/env python3
#| """
#| 22_modality_arms.py -- how much of the skill is satellite-borne, and does the
#| central finding hold inside the satellite-only panel?
#|
#| READS   ../derived*/panel_manifest.json  (script 16, which froze the columns)
#|         the modelling panel, through _config.load_panel
#|         the census of script 12
#| WRITES  ../derived*/modality_arms.json
#|         ../derived*/modality_arms.csv
#|
#| THE OBJECTION
#| -------------
#|     "This is a socioeconomic model with satellite decoration. You have not
#|      separated the two, and the reader cannot tell which does the work."
#|
#| The first thing this script does is answer that objection in a way that was
#| not anticipated when it was written. Script 16's export found that the model's
#| 581 columns contain NO socioeconomic, census or health-outcome column at all:
#| the arm built as a control turned out to be the model. Whatever else is true
#| of this paper, the predictors are environmental measurements and nothing else,
#| and the count is in the manifest rather than in anyone's recollection.
#|
#| That makes the interesting split a different one, and it is the one this
#| script measures:
#|
#|     all admitted       every column the production model saw
#|     satellite only     columns from an instrument in orbit
#|     non satellite      reanalysis, soil survey, agricultural census,
#|                        livestock inventory, and the engineered combinations
#|                        built from them
#|
#| The three scores say how much of the skill needs an instrument in orbit and
#| how much a reanalysis product already carries. Both answers are publishable
#| and they are different papers, so the number has to be measured rather than
#| assumed.
#|
#| THE SECOND QUESTION
#| -------------------
#| Within the satellite-only panel, refitted and re-attributed from scratch, does
#| the non-central summary of a field still lead? If the central finding is a
#| property of the data rather than of one particular column set, it should
#| survive being asked inside a third of the columns.
#|
#|     python3 22_modality_arms.py
#|     python3 22_modality_arms.py --only arms
#|     python3 22_modality_arms.py --folds 3        # smoke test, says so
#| """
#| from __future__ import annotations
#|
#| import argparse
#| import gc
#| import json
#| import os
#| import sys
#| import time
#|
#| import numpy as np
#| import pandas as pd
#|
#| HERE = os.path.dirname(os.path.abspath(__file__))
#| if HERE not in sys.path:
#|     sys.path.insert(0, HERE)
#|
#| import _config as C                                            # noqa: E402
#|
#| MANIFEST = C.DERIVED / "panel_manifest.json"
#| OUT_JSON = C.DERIVED / "modality_arms.json"
#| OUT_CSV = C.DERIVED / "modality_arms.csv"
#|
#| SHAP_ROWS = 6000          # rows attributed inside each arm
#| _SMOKE = {"folds": 0}
#|
#|
#| def folds():
#|     return _SMOKE["folds"] or C.N_FOLDS
#|
#|
#| def tag(name):
#|     return f"modality_{name}" + (f"_f{_SMOKE['folds']}" if _SMOKE["folds"]
#|                                  else "")
#|
#|
#| def sign_test_p(wins, n):
#|     from math import comb
#|     if not n:
#|         return float("nan")
#|     return float(sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n)
#|
#|
#| def path_dependent(model, X, rng, n_rows=SHAP_ROWS):
#|     """Mean absolute path-dependent TreeSHAP, computed by xgboost itself."""
#|     import xgboost as xgb
#|     idx = rng.choice(len(X), min(n_rows, len(X)), replace=False)
#|     sub = X.iloc[idx]
#|     dm = xgb.DMatrix(sub, feature_names=list(sub.columns))
#|     contrib = np.asarray(model.get_booster().predict(dm, pred_contribs=True))
#|     return dict(zip(sub.columns,
#|                     np.abs(contrib[:, :-1]).mean(axis=0).astype(float)))
#|
#|
#| def fit_full(X, y, name):
#|     """One model on all rows, for attribution inside an arm. Not a score."""
#|     import xgboost as xgb
#|     store = C.DERIVED / "checkpoints_modality"
#|     store.mkdir(parents=True, exist_ok=True)
#|     path = store / f"{name}_full.json"
#|     m = xgb.XGBRegressor(**C.XGB_PARAMS, n_jobs=C.threads())
#|     if path.exists():
#|         C.load_xgb(m, path)
#|         print(f"      {name}: full-data model from checkpoint", flush=True)
#|         return m
#|     t0 = time.time()
#|     m.fit(X, y)
#|     m.save_model(str(path))
#|     print(f"      {name}: full-data model fitted in "
#|           f"{(time.time() - t0) / 60:.1f} min", flush=True)
#|     return m
#|
#|
#| # ---------------------------------------------------------------------------
#| def read_manifest():
#|     if not MANIFEST.exists():
#|         raise SystemExit(
#|             f"{MANIFEST.name} is not there.\n"
#|             f"    It is written by 16_export_frozen_panels_and_gate_audit.py,\n"
#|             f"    which is where the column lists for these arms come from.\n"
#|             f"    Run:  python3 16_export_frozen_panels_and_gate_audit.py "
#|             f"--only export")
#|     blob = json.load(open(MANIFEST))
#|     need = ("columns_all_admitted", "columns_satellite_only",
#|             "n_modelled", "n_satellite_only")
#|     gone = [k for k in need if k not in blob]
#|     if gone:
#|         raise SystemExit(f"{MANIFEST.name} is missing {', '.join(gone)}. "
#|                          f"Rerun script 16's export arm.")
#|     return blob
#|
#|
#| def arms_from_manifest(man, present):
#|     allc = [c for c in man["columns_all_admitted"] if c in present]
#|     sat = [c for c in man["columns_satellite_only"] if c in present]
#|     non = [c for c in allc if c not in set(sat)]
#|     return {"all_admitted": allc, "satellite_only": sat,
#|             "non_satellite": non}
#|
#|
#| # ---------------------------------------------------------------------------
#| def run_arms(p, cols):
#|     print("\n" + "=" * 74)
#|     print("  1. THREE COLUMN SETS, THE SAME PROTOCOL")
#|     print("=" * 74)
#|     print("  County-grouped folds, the production learner, the production")
#|     print("  settings. The only thing that changes between arms is which")
#|     print("  columns are available, so the difference between two scores is")
#|     print("  the value of those columns and nothing else.")
#|     if _SMOKE["folds"]:
#|         print(f"\n  SMOKE TEST at {_SMOKE['folds']} folds instead of "
#|               f"{C.N_FOLDS}. Not comparable with the paper.")
#|     print()
#|     X, y, groups = p["X"], p["y"], p["groups"]
#|     rows = {}
#|     for name in ("all_admitted", "satellite_only", "non_satellite"):
#|         use = cols[name]
#|         if not use:
#|             print(f"    {name}: no columns, skipped")
#|             continue
#|         s = C.cv_score(X[use], y, groups, tag(name), n_folds=folds())
#|         s["arm"] = name
#|         rows[name] = s
#|         print(f"    {name:<18} {len(use):>4} columns   R2 {s['r2']:+.4f}   "
#|               f"MAE {s['mae']:.4f}")
#|     if "all_admitted" in rows and "satellite_only" in rows:
#|         base = rows["all_admitted"]
#|         print()
#|         for name in ("satellite_only", "non_satellite"):
#|             if name in rows:
#|                 d = rows[name]["r2"] - base["r2"]
#|                 share = (rows[name]["r2"] / base["r2"]
#|                          if base["r2"] else float("nan"))
#|                 print(f"    {name:<18} keeps {100 * share:5.1f} per cent of "
#|                       f"the full model's R2   ({d:+.4f})")
#|         print()
#|         print("    Two column sets that overlap in nothing and each keep most")
#|         print("    of the skill is the signature of a well covered signal")
#|         print("    measured twice, not of one indispensable instrument. The")
#|         print("    paper should say which of those it is claiming.")
#|     return rows
#|
#|
#| def census_within(p, cols, rng):
#|     print("\n" + "=" * 74)
#|     print("  2. THE CENSUS QUESTION, ASKED INSIDE THE SATELLITE-ONLY PANEL")
#|     print("=" * 74)
#|     print("  A model refitted on the satellite columns alone, attributed from")
#|     print("  scratch, and asked the paper's own question: of the two")
#|     print("  descriptions of a field, which takes the attribution?")
#|     print()
#|     X, y = p["X"], p["y"]
#|     sat = cols["satellite_only"]
#|     pairs_all = C.census_pairs(columns=X.columns)
#|     pairs = [(f, c, n) for f, c, n in pairs_all
#|              if c in set(sat) and n in set(sat)]
#|     print(f"    {len(pairs)} of the census's {len(pairs_all)} pairs have both "
#|           f"members inside the satellite panel")
#|     if not pairs:
#|         print("    Nothing to compare. The satellite panel does not contain")
#|         print("    both descriptions of any field, which is itself a fact")
#|         print("    about the instrument coverage and is recorded as one.")
#|         return dict(n_pairs=0,
#|                     note="no census pair lies wholly inside the satellite "
#|                          "panel"), []
#|     m = fit_full(X[sat], y, "satellite_only")
#|     vals = path_dependent(m, X[sat], rng)
#|     del m
#|     gc.collect()
#|     rows, wins = [], 0
#|     for field, c_col, n_col in pairs:
#|         cv, nv = vals.get(c_col), vals.get(n_col)
#|         if cv is None or nv is None:
#|             continue
#|         win = bool(nv > cv)
#|         wins += int(win)
#|         rows.append(dict(arm="census_within_satellite", field=field,
#|                          central_col=c_col, noncentral_col=n_col,
#|                          central_value=float(cv), noncentral_value=float(nv),
#|                          noncentral_wins=win))
#|     n = len(rows)
#|     pv = sign_test_p(wins, n)
#|     print(f"    non-central leads in {wins} of {n}   sign test p = {pv:.4g}")
#|     # the same pairs, as the published full-model attribution ranked them
#|     pub = C.load_table("A_shap_ranking")
#|     pubv = dict(zip(pub["feature"].astype(str),
#|                     pub["mean_abs_shap"].astype(float)))
#|     agree = sum(1 for r in rows
#|                 if (pubv.get(r["noncentral_col"], np.nan) >
#|                     pubv.get(r["central_col"], np.nan)) == r["noncentral_wins"])
#|     print(f"    the full model agreed on the direction for {agree} of {n} of "
#|           f"the same pairs")
#|     print()
#|     print("    This arm changes two things at once, the column set and the")
#|     print("    attribution estimator, so a disagreement here does not say")
#|     print("    which of the two caused it. Script 21 separates the estimator;")
#|     print("    this arm separates the column set.")
#|     return dict(n_pairs=n, noncentral_wins=wins, sign_test_p=pv,
#|                 agreement_with_full_model=int(agree),
#|                 attribution="path dependent TreeSHAP on a satellite-only "
#|                             "refit",
#|                 note="Both the column set and the attribution estimator "
#|                      "differ from the published ranking."), rows
#|
#|
#| # ---------------------------------------------------------------------------
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--only", default="all",
#|                     help="comma separated: arms, census")
#|     ap.add_argument("--folds", type=int, default=0,
#|                     help="SMOKE TEST ONLY. Fewer folds than the study uses. "
#|                          "Results are written under a separate tag and the "
#|                          "file says so.")
#|     a = ap.parse_args()
#|     want = (["arms", "census"] if a.only == "all"
#|             else [s.strip() for s in a.only.split(",")])
#|     if a.folds:
#|         _SMOKE["folds"] = a.folds
#|
#|     C.banner("22  how much of the skill needs an instrument in orbit")
#|     C.seed_everything()
#|     rng = C.seed_everything(extra=24)
#|     C.DERIVED.mkdir(parents=True, exist_ok=True)
#|
#|     man = read_manifest()
#|     p = C.load_panel()
#|     cols = arms_from_manifest(man, set(p["X"].columns))
#|     print(f"  panel {p['X'].shape[0]:,} rows x {p['X'].shape[1]} columns")
#|     print(f"  candidates {man['n_candidates']}, gate admitted "
#|           f"{man['n_gate_admitted']}, modelled {man['n_modelled']}")
#|     for k, v in cols.items():
#|         print(f"    {k:<18} {len(v):>4} columns")
#|     nonphys = man["n_modelled"] - man.get("n_environment_only",
#|                                           man["n_modelled"])
#|     print(f"\n  socioeconomic, census and health-outcome columns in the "
#|           f"model: {nonphys}")
#|     if nonphys == 0:
#|         print("  The model has none. The predictors are environmental")
#|         print("  measurements only, which answers the objection this script")
#|         print("  was written for before any arm is fitted, and is the first")
#|         print("  thing the paper should say about the panel.")
#|
#|     blob, rows = {}, []
#|     if "arms" in want:
#|         blob["arms"] = run_arms(p, cols)
#|     if "census" in want:
#|         out, r = census_within(p, cols, rng)
#|         blob["census_within_satellite"], rows = out, rows + r
#|
#|     if OUT_JSON.exists():
#|         try:
#|             old = json.load(open(OUT_JSON))
#|             for k, v in old.items():
#|                 blob.setdefault(k, v)
#|         except Exception:
#|             pass
#|     blob["column_counts"] = {k: len(v) for k, v in cols.items()}
#|     blob["n_non_physical_columns_in_model"] = int(nonphys)
#|     blob["n_folds"] = int(folds())
#|     blob["smoke_test"] = bool(_SMOKE["folds"])
#|     json.dump(blob, open(OUT_JSON, "w"), indent=1, default=str)
#|     print(f"\n  [saved] {OUT_JSON}")
#|
#|     table = [dict(arm=k, **{kk: vv for kk, vv in v.items() if kk != "arm"})
#|              for k, v in (blob.get("arms") or {}).items()]
#|     out = pd.DataFrame(table + rows)
#|     if not out.empty:
#|         out.to_csv(OUT_CSV, index=False)
#|         print(f"  [saved] {OUT_CSV}")
#|
#|     C.write_manifest("22_modality_arms.py",
#|                      inputs=[MANIFEST, C.FRAME, C.MODEL_A],
#|                      outputs=[OUT_JSON, OUT_CSV],
#|                      settings=dict(arms=want, n_folds=folds(),
#|                                    SHAP_ROWS=SHAP_ROWS,
#|                                    smoke_test=bool(_SMOKE["folds"])),
#|                      notes="Satellite-borne against non-satellite column sets, "
#|                            "and the census question asked inside the "
#|                            "satellite-only panel.")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/22_modality_arms.py
#@@ FILE analysis/23_influence_and_perturbation.py
#| #!/usr/bin/env python3
#| """
#| 23_influence_and_perturbation.py -- is any of this carried by a few counties,
#| and how far do the thresholds move under measurement error?
#|
#| READS   the fitted model and the modelling panel; the out-of-fold predictions
#|         of the main run; the census of script 12; the attribution surface of
#|         script 05; the turning points of script 06
#| WRITES  ../derived*/census_jackknife.json, census_jackknife.csv
#|         ../derived*/influence_drop.json
#|         ../derived*/perturbation_thresholds.json,
#|                     perturbation_thresholds.csv
#|
#| THREE OBJECTIONS
#| ----------------
#|     --only jackknife
#|         "Drop California and your thirty-eight of forty-seven is twenty of
#|          forty-seven."
#|     The census count is a function of a row average, so it can be recomputed
#|     exactly with any state's rows removed, at the cost of one attribution pass
#|     rather than fifty refits. What this holds fixed is stated below and
#|     matters.
#|
#|     --only influence
#|         "A hundred unusual counties are driving this."
#|     Remove the counties the model handles worst, and the counties with the
#|     most extreme outcomes, and refit. The score will RISE when hard cases are
#|     dropped, which is not evidence of anything; the ranking is what is being
#|     tested.
#|
#|     --only perturb
#|         "These are reanalysis fields with their own error and you report
#|          thresholds to three decimals."
#|     Perturb each predictor at several stated magnitudes and relocate its
#|     crossing with script 06's own estimator. Report the movement against the
#|     width of the published interval.
#|
#| WHAT THE PERTURBATION ARM DOES NOT DO
#| -------------------------------------
#| It perturbs the predictor and relocates the crossing on the EXISTING
#| attribution values. It does not refit the model on perturbed predictors, which
#| would be the full error-in-variables experiment and costs a refit per
#| magnitude per variable. So it measures the sensitivity of the located value to
#| predictor error, not the sensitivity of the model. That is the narrower claim
#| and it is the one the result file makes.
#|
#| No perturbed value is written to any exported panel, used by any figure, or
#| read by the manuscript as a measurement. These are sensitivity arms and every
#| file says so.
#|
#|     python3 23_influence_and_perturbation.py --only jackknife
#|     python3 23_influence_and_perturbation.py --only perturb
#|     python3 23_influence_and_perturbation.py --only influence --folds 3
#| """
#| from __future__ import annotations
#|
#| import argparse
#| import gc
#| import importlib.util
#| import json
#| import os
#| import sys
#| import time
#|
#| import numpy as np
#| import pandas as pd
#|
#| HERE = os.path.dirname(os.path.abspath(__file__))
#| if HERE not in sys.path:
#|     sys.path.insert(0, HERE)
#|
#| import _config as C                                            # noqa: E402
#|
#| JACK_JSON = C.DERIVED / "census_jackknife.json"
#| JACK_CSV = C.DERIVED / "census_jackknife.csv"
#| INFL_JSON = C.DERIVED / "influence_drop.json"
#| PERT_JSON = C.DERIVED / "perturbation_thresholds.json"
#| PERT_CSV = C.DERIVED / "perturbation_thresholds.csv"
#|
#| # ===========================================================================
#| #  SETTINGS
#| # ===========================================================================
#| JACK_ROWS = 30000          # rows attributed once, then averaged with each
#|                            # state's rows left out in turn
#| DROP_SHARES = (0.01, 0.05)  # share of counties removed in the influence arms
#| PERTURB_SDS = (0.01, 0.02, 0.05, 0.10)   # noise as a share of the column's
#|                                          # own standard deviation
#| PERTURB_DRAWS = 25         # independent perturbations per magnitude
#| PERTURB_BOOT = 200         # bootstrap draws inside each. Lower than the
#|                            # paper's 5,000 because this arm asks how far the
#|                            # point estimate moves, not what the interval is,
#|                            # and the number is recorded in the result file
#| TOPK = 15                  # ranking overlap reported at this depth
#|
#| _SMOKE = {"folds": 0}
#|
#|
#| def folds():
#|     return _SMOKE["folds"] or C.N_FOLDS
#|
#|
#| def sign_test_p(wins, n):
#|     from math import comb
#|     if not n:
#|         return float("nan")
#|     return float(sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n)
#|
#|
#| def canonical_crossing():
#|     path = os.path.join(HERE, "06_bootstrap_turning_points_and_gate.py")
#|     if not os.path.exists(path):
#|         raise SystemExit(
#|             "06_bootstrap_turning_points_and_gate.py is not beside this file.\n"
#|             "    The perturbation arm relocates each crossing with that\n"
#|             "    script's estimator so the movement is comparable with the\n"
#|             "    published table. Copy the whole analysis folder.")
#|     spec = importlib.util.spec_from_file_location("_s06", path)
#|     mod = importlib.util.module_from_spec(spec)
#|     spec.loader.exec_module(mod)
#|     return mod
#|
#|
#| def path_contribs(model, X, idx, keep):
#|     """Per-row path-dependent contributions for the columns in `keep`."""
#|     import xgboost as xgb
#|     sub = X.iloc[idx]
#|     dm = xgb.DMatrix(sub, feature_names=list(sub.columns))
#|     contrib = np.asarray(model.get_booster().predict(dm, pred_contribs=True))
#|     cols = list(sub.columns)
#|     pos = {c: i for i, c in enumerate(cols)}
#|     return {c: np.abs(contrib[:, pos[c]]) for c in keep if c in pos}
#|
#|
#| # ---------------------------------------------------------------------------
#| #  1. Leave one state out of the census count
#| # ---------------------------------------------------------------------------
#| def jackknife(p, model, rng):
#|     print("\n" + "=" * 74)
#|     print("  1. LEAVE ONE STATE OUT OF THE CENSUS COUNT")
#|     print("=" * 74)
#|     print("  The census count compares the mean absolute attribution of two")
#|     print("  columns, and a mean is a row average, so the count with one")
#|     print("  state's rows removed can be computed exactly from one")
#|     print("  attribution pass. Fifty refits are not needed and would answer a")
#|     print("  different question.")
#|     print()
#|     print("  HELD FIXED: the fitted model. This arm asks whether the count")
#|     print("  depends on which counties are ATTRIBUTED. Whether it depends on")
#|     print("  which counties were TRAINED on is the state holdout of script 13.")
#|     print()
#|     X, y, groups = p["X"], p["y"], p["groups"]
#|     pairs = C.census_pairs(columns=X.columns)
#|     keep = sorted({c for _, a_, b_ in pairs for c in (a_, b_)})
#|     print(f"    {len(pairs)} pairs, {len(keep)} columns")
#|
#|     n = len(X)
#|     idx = (rng.choice(n, JACK_ROWS, replace=False) if n > JACK_ROWS
#|            else np.arange(n))
#|     t0 = time.time()
#|     contrib = path_contribs(model, X, idx, keep)
#|     print(f"    attributed {len(idx):,} rows in "
#|           f"{(time.time() - t0) / 60:.1f} min")
#|
#|     # a county FIPS is five digits, the first two of which are the state
#|     fips = np.asarray([f"{int(g):05d}" for g in groups[idx]])
#|     states = np.array([f[:2] for f in fips])
#|     uniq = sorted(set(states.tolist()))
#|     print(f"    {len(uniq)} states present among the attributed rows")
#|
#|     def count(mask):
#|         wins, got = 0, 0
#|         for field, c_col, n_col in pairs:
#|             cv, nv = contrib.get(c_col), contrib.get(n_col)
#|             if cv is None or nv is None:
#|                 continue
#|             got += 1
#|             wins += int(nv[mask].mean() > cv[mask].mean())
#|         return got, wins
#|
#|     base_got, base_wins = count(np.ones(len(idx), bool))
#|     print(f"\n    all states: non-central leads in {base_wins} of {base_got}")
#|
#|     rows = []
#|     for st in uniq:
#|         m = states != st
#|         got, wins = count(m)
#|         rows.append(dict(arm="leave_one_state_out", state_fips=st,
#|                          n_rows_dropped=int((~m).sum()),
#|                          n_comparable=got, noncentral_wins=wins,
#|                          sign_test_p=sign_test_p(wins, got)))
#|     df = pd.DataFrame(rows)
#|     lo, hi = int(df.noncentral_wins.min()), int(df.noncentral_wins.max())
#|     worst = df.loc[df.noncentral_wins.idxmin()]
#|     print(f"    leaving out any one state, the count stays between {lo} and "
#|           f"{hi} of {base_got}")
#|     print(f"    the state whose removal lowers it most is FIPS "
#|           f"{worst['state_fips']} ({int(worst['noncentral_wins'])})")
#|     worst_p = float(df.sign_test_p.max())
#|     print(f"    the largest sign-test p over all fifty removals is "
#|           f"{worst_p:.3g}")
#|     print()
#|     print("    That last number is the one to quote. It says the count")
#|     print("    remains significant no matter which single state is removed.")
#|     df.to_csv(JACK_CSV, index=False)
#|     print(f"  [saved] {JACK_CSV}")
#|     return dict(n_rows_attributed=int(len(idx)), n_states=len(uniq),
#|                 n_comparable=base_got, baseline_wins=base_wins,
#|                 attribution="path dependent TreeSHAP, model held fixed",
#|                 range=[lo, hi], worst_state=str(worst["state_fips"]),
#|                 worst_count=int(worst["noncentral_wins"]),
#|                 worst_sign_test_p=worst_p,
#|                 per_state=df.to_dict("records"),
#|                 note="The fitted model is held fixed; only the attributed "
#|                      "rows change.")
#|
#|
#| # ---------------------------------------------------------------------------
#| #  2. Drop the counties the model relies on most
#| # ---------------------------------------------------------------------------
#| def influence(p, rng):
#|     print("\n" + "=" * 74)
#|     print("  2. DROP THE COUNTIES THE MODEL HANDLES WORST, AND THE EXTREMES")
#|     print("=" * 74)
#|     print("  The score will RISE when the hardest counties are removed. That")
#|     print("  is arithmetic, not evidence, and the result file says so. What is")
#|     print("  being tested is the RANKING: if the leading variables are the")
#|     print("  same after the removal, no small set of counties is carrying")
#|     print("  them.")
#|     if _SMOKE["folds"]:
#|         print(f"\n  SMOKE TEST at {_SMOKE['folds']} folds. Not the study.")
#|     print()
#|     X, y, groups = p["X"], p["y"], p["groups"]
#|
#|     # mean absolute out-of-fold residual per county, from the main run's own
#|     # predictions rather than from a refit here
#|     oof = None
#|     try:
#|         o = pd.read_csv(C.OOF_A)
#|         cols = {c.lower(): c for c in o.columns}
#|         pc = cols.get("prediction") or cols.get("pred") or cols.get("yhat")
#|         tc = cols.get("actual") or cols.get("observed") or cols.get("y") \
#|             or cols.get(C.TARGET)
#|         gc_ = cols.get(C.GROUP) or cols.get("geoid") or cols.get("county")
#|         if pc and tc and gc_:
#|             o = o[[gc_, tc, pc]].dropna()
#|             o["abs_err"] = (o[tc] - o[pc]).abs()
#|             oof = o.groupby(gc_)["abs_err"].mean()
#|             print(f"    out-of-fold residuals for {len(oof):,} counties, from "
#|                   f"{C.OOF_A.name}")
#|     except Exception as e:
#|         print(f"    could not read {C.OOF_A.name} ({type(e).__name__}); the "
#|               f"residual arms are skipped")
#|
#|     county_mean_y = pd.Series(y).groupby(pd.Series(groups)).mean()
#|     arms = {}
#|
#|     def run(name, drop_counties, why):
#|         keep = ~np.isin(groups, np.asarray(list(drop_counties)))
#|         if keep.sum() < 0.5 * len(groups):
#|             print(f"    {name}: would drop more than half the rows, skipped")
#|             return
#|         Xk, yk, gk = X[keep], y[keep], groups[keep]
#|         s = C.cv_score(Xk, yk, gk,
#|                        f"influence_{name}" + (f"_f{_SMOKE['folds']}"
#|                                               if _SMOKE["folds"] else ""),
#|                        n_folds=folds())
#|         s.update(arm=name, why=why, n_counties_dropped=len(drop_counties),
#|                  n_rows_dropped=int((~keep).sum()))
#|         arms[name] = s
#|         print(f"    {name:<22} dropped {len(drop_counties):>4} counties   "
#|               f"R2 {s['r2']:+.4f}   MAE {s['mae']:.4f}")
#|
#|     for share in DROP_SHARES:
#|         if oof is not None:
#|             k = max(int(round(len(oof) * share)), 1)
#|             worst = set(oof.sort_values(ascending=False).head(k).index.tolist())
#|             run(f"worst_residual_{int(100 * share)}pct", worst,
#|                 "counties with the largest mean absolute out-of-fold residual")
#|         k = max(int(round(len(county_mean_y) * share / 2)), 1)
#|         srt = county_mean_y.sort_values()
#|         extreme = set(srt.head(k).index.tolist()) | \
#|             set(srt.tail(k).index.tolist())
#|         run(f"extreme_outcome_{int(100 * share)}pct", extreme,
#|             "counties in the lowest and highest tails of mean life expectancy")
#|
#|     # the ranking, after the largest removal, against the published one
#|     overlap = {}
#|     if arms:
#|         pub = C.load_table("A_shap_ranking")
#|         pub_top = list(pub.sort_values("rank")["feature"].astype(str)[:TOPK])
#|         import xgboost as xgb
#|         drop_name = max(arms, key=lambda k: arms[k]["n_counties_dropped"])
#|         print(f"\n    refitting once on the {drop_name} panel to compare the "
#|               f"top {TOPK}")
#|         # recover the mask for that arm by rerunning its selection
#|         if oof is not None and drop_name.startswith("worst_residual"):
#|             share = int(drop_name.split("_")[-1].replace("pct", "")) / 100
#|             k = max(int(round(len(oof) * share)), 1)
#|             dropped = set(oof.sort_values(ascending=False).head(k).index)
#|         else:
#|             share = int(drop_name.split("_")[-1].replace("pct", "")) / 100
#|             k = max(int(round(len(county_mean_y) * share / 2)), 1)
#|             srt = county_mean_y.sort_values()
#|             dropped = set(srt.head(k).index) | set(srt.tail(k).index)
#|         keep = ~np.isin(groups, np.asarray(list(dropped)))
#|         m = xgb.XGBRegressor(**C.XGB_PARAMS, n_jobs=C.threads())
#|         t0 = time.time()
#|         m.fit(X[keep], y[keep])
#|         sub = X[keep]
#|         ridx = (C.seed_everything(extra=25)
#|                 .choice(len(sub), min(6000, len(sub)), replace=False))
#|         dm = xgb.DMatrix(sub.iloc[ridx], feature_names=list(sub.columns))
#|         contrib = np.asarray(m.get_booster().predict(dm, pred_contribs=True))
#|         mean_abs = pd.Series(np.abs(contrib[:, :-1]).mean(axis=0),
#|                              index=sub.columns).sort_values(ascending=False)
#|         new_top = list(mean_abs.index[:TOPK])
#|         shared = len(set(pub_top) & set(new_top))
#|         overlap = dict(arm=drop_name, topk=TOPK, shared=shared,
#|                        published_top=pub_top, refit_top=new_top,
#|                        attribution="path dependent TreeSHAP on the refit, so "
#|                                    "the comparison with the published "
#|                                    "interventional ranking crosses two "
#|                                    "estimators and is a direction check, not "
#|                                    "a value check")
#|         print(f"    top {TOPK} overlap with the published ranking: "
#|               f"{shared} of {TOPK}   "
#|               f"({(time.time() - t0) / 60:.1f} min)")
#|         del m
#|         gc.collect()
#|
#|     return dict(arms=arms, topk_overlap=overlap,
#|                 drop_shares=list(DROP_SHARES), n_folds=int(folds()),
#|                 smoke_test=bool(_SMOKE["folds"]),
#|                 note="Dropping the hardest counties raises the score by "
#|                      "construction. The ranking overlap is the test.")
#|
#|
#| # ---------------------------------------------------------------------------
#| #  3. Perturb the predictor and relocate the crossing
#| # ---------------------------------------------------------------------------
#| def perturb(args):
#|     print("\n" + "=" * 74)
#|     print("  3. PREDICTOR PERTURBATION AND THE LOCATED CROSSINGS")
#|     print("=" * 74)
#|     print("  Each predictor is perturbed by Gaussian noise at several stated")
#|     print("  magnitudes, as a share of that column's own standard deviation,")
#|     print("  and its crossing is relocated with script 06's estimator. The")
#|     print("  movement is reported against the width of the published interval,")
#|     print("  because a shift of a tenth of the interval is a different fact")
#|     print("  from a shift of three times its width.")
#|     print()
#|     print("  The attribution values are NOT recomputed. This arm measures the")
#|     print("  sensitivity of the located value to predictor error, not the")
#|     print("  sensitivity of the model to it.")
#|     print()
#|     s06 = canonical_crossing()
#|     dep, feats = C.dependence_table()
#|     print(f"    {len(feats)} candidates in the attribution surface, "
#|           f"{len(dep):,} rows each")
#|     try:
#|         pub = C.load_table("turning_points").set_index("feature")
#|     except SystemExit:
#|         pub = None
#|         print("    (turning_points.csv absent, so the movement cannot be "
#|               "scaled by the published interval; run script 06)")
#|
#|     rng = C.seed_everything(extra=26)
#|     rows, per_var = [], {}
#|     mags = args.perturb_sds or PERTURB_SDS
#|     draws = args.perturb_draws or PERTURB_DRAWS
#|     boot = args.perturb_boot or PERTURB_BOOT
#|     # One checkpoint per variable, keyed by the settings that produced it.
#|     # Eleven variables at four magnitudes and twenty-five draws is eleven
#|     # hundred bootstrapped relocations, which is longer than one sitting, and
#|     # an arm that cannot be resumed is an arm that never finishes.
#|     store = C.DERIVED / "checkpoints_perturb"
#|     store.mkdir(parents=True, exist_ok=True)
#|     stamp = f"d{draws}_b{boot}_m" + "-".join(f"{m:g}" for m in mags)
#|     for feat in feats:
#|         ck = store / f"{feat}__{stamp}.json"
#|         if ck.exists():
#|             try:
#|                 rec = json.load(open(ck))
#|                 per_var[str(feat)] = rec
#|                 for mk, mv in (rec.get("magnitudes") or {}).items():
#|                     if isinstance(mv, dict) and mv.get("n_found"):
#|                         rows.append(dict(
#|                             arm="perturbation", feature=str(feat),
#|                             magnitude=float(mk),
#|                             **{kk: mv[kk] for kk in
#|                                ("n_found", "median_point", "shift", "spread",
#|                                 "shift_over_interval_width") if kk in mv}))
#|                 print(f"    {str(feat)[:44]:<44} from checkpoint")
#|                 continue
#|             except Exception:
#|                 pass
#|         x = dep[feat].to_numpy(float)
#|         v = dep["shap_" + feat].to_numpy(float)
#|         ok = np.isfinite(x) & np.isfinite(v)
#|         x, v = x[ok], v[ok]
#|         if len(x) < 2000:
#|             continue
#|         sd = float(np.std(x))
#|         base = s06.analyse(str(feat), x, v, boot, C.N_BINS, rng)
#|         bp = base["point_estimate"]
#|         width = np.nan
#|         if pub is not None and feat in pub.index:
#|             width = float(pub.loc[feat, "ci_hi"] - pub.loc[feat, "ci_lo"])
#|             bp_pub = float(pub.loc[feat, "point_estimate"])
#|         else:
#|             bp_pub = bp
#|         rec = dict(feature=str(feat), sd=sd, published_point=bp_pub,
#|                    published_interval_width=width, magnitudes={})
#|         for mag in mags:
#|             pts = []
#|             for _ in range(draws):
#|                 xx = x + rng.normal(0.0, mag * sd, len(x))
#|                 r = s06.analyse(str(feat), xx, v, boot, C.N_BINS, rng)
#|                 if r["point_estimate"] is not None and \
#|                         np.isfinite(r["point_estimate"]):
#|                     pts.append(float(r["point_estimate"]))
#|             if not pts:
#|                 rec["magnitudes"][f"{mag:g}"] = dict(n_found=0)
#|                 continue
#|             pts = np.asarray(pts)
#|             shift = float(np.median(pts) - bp_pub)
#|             spread = float(pts.std())
#|             rel = (abs(shift) / width if np.isfinite(width) and width
#|                    else float("nan"))
#|             rec["magnitudes"][f"{mag:g}"] = dict(
#|                 n_found=int(len(pts)), n_draws=int(draws),
#|                 median_point=float(np.median(pts)), shift=shift,
#|                 spread=spread, shift_over_interval_width=rel)
#|             rows.append(dict(arm="perturbation", feature=str(feat),
#|                              magnitude=float(mag), n_found=int(len(pts)),
#|                              median_point=float(np.median(pts)), shift=shift,
#|                              spread=spread, shift_over_interval_width=rel))
#|         per_var[str(feat)] = rec
#|         json.dump(rec, open(ck, "w"), indent=1, default=str)
#|         worst = max((m.get("shift_over_interval_width", np.nan)
#|                      for m in rec["magnitudes"].values()
#|                      if isinstance(m, dict)), default=np.nan)
#|         verdict = (f"worst shift {worst:.2f} x the published interval width"
#|                    if np.isfinite(worst) else "interval width unknown")
#|         print(f"    {str(feat)[:44]:<44} {verdict}")
#|
#|     df = pd.DataFrame(rows)
#|     out = dict(per_variable=per_var, magnitudes=[float(m) for m in mags],
#|                n_draws=int(draws), n_boot=int(boot),
#|                note="Gaussian perturbation of real predictor values. A "
#|                     "sensitivity arm. No perturbed value is exported, "
#|                     "plotted, or read by the manuscript as a measurement. "
#|                     "The attribution values are not recomputed.")
#|     if not df.empty:
#|         df.to_csv(PERT_CSV, index=False)
#|         print(f"  [saved] {PERT_CSV}")
#|         big = df.dropna(subset=["shift_over_interval_width"])
#|         if len(big):
#|             at = big[big.magnitude == max(mags)]
#|             if len(at):
#|                 print(f"\n    at the largest magnitude ({max(mags):g} of the "
#|                       f"column's own spread), the median shift is")
#|                 print(f"    {at.shift_over_interval_width.median():.2f} times "
#|                       f"the published interval width, and the worst is "
#|                       f"{at.shift_over_interval_width.max():.2f}")
#|             out["summary_at_largest_magnitude"] = dict(
#|                 magnitude=float(max(mags)),
#|                 median_shift_over_width=float(
#|                     at.shift_over_interval_width.median()) if len(at) else None,
#|                 max_shift_over_width=float(
#|                     at.shift_over_interval_width.max()) if len(at) else None)
#|     return out
#|
#|
#| # ---------------------------------------------------------------------------
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--only", default="all",
#|                     help="comma separated: jackknife, influence, perturb")
#|     ap.add_argument("--folds", type=int, default=0,
#|                     help="SMOKE TEST ONLY, for the influence arm")
#|     ap.add_argument("--perturb-sds", type=float, nargs="*", default=None,
#|                     help="noise magnitudes as a share of the column's own sd")
#|     ap.add_argument("--perturb-draws", type=int, default=0)
#|     ap.add_argument("--perturb-boot", type=int, default=0)
#|     a = ap.parse_args()
#|     want = (["jackknife", "influence", "perturb"] if a.only == "all"
#|             else [s.strip() for s in a.only.split(",")])
#|     if a.folds:
#|         _SMOKE["folds"] = a.folds
#|
#|     C.banner("23  influence, and the thresholds under measurement error")
#|     C.seed_everything()
#|     C.DERIVED.mkdir(parents=True, exist_ok=True)
#|     rng = C.seed_everything(extra=27)
#|
#|     need_panel = bool({"jackknife", "influence"} & set(want))
#|     p = C.load_panel() if need_panel else None
#|     if p is not None:
#|         print(f"  panel {p['X'].shape[0]:,} rows x {p['X'].shape[1]} columns")
#|
#|     if "jackknife" in want:
#|         import joblib
#|         model = joblib.load(C.MODEL_A)["model"]
#|         out = jackknife(p, model, rng)
#|         json.dump(out, open(JACK_JSON, "w"), indent=1, default=str)
#|         print(f"  [saved] {JACK_JSON}")
#|         del model
#|         gc.collect()
#|
#|     if "influence" in want:
#|         out = influence(p, rng)
#|         json.dump(out, open(INFL_JSON, "w"), indent=1, default=str)
#|         print(f"  [saved] {INFL_JSON}")
#|
#|     if "perturb" in want:
#|         out = perturb(a)
#|         json.dump(out, open(PERT_JSON, "w"), indent=1, default=str)
#|         print(f"  [saved] {PERT_JSON}")
#|
#|     C.write_manifest("23_influence_and_perturbation.py",
#|                      inputs=[C.MODEL_A, C.FRAME, C.OOF_A],
#|                      outputs=[q for q in (JACK_JSON, JACK_CSV, INFL_JSON,
#|                                           PERT_JSON, PERT_CSV) if q.exists()],
#|                      settings=dict(arms=want, DROP_SHARES=list(DROP_SHARES),
#|                                    PERTURB_SDS=list(a.perturb_sds
#|                                                     or PERTURB_SDS),
#|                                    PERTURB_DRAWS=a.perturb_draws
#|                                    or PERTURB_DRAWS,
#|                                    JACK_ROWS=JACK_ROWS, TOPK=TOPK,
#|                                    n_folds=folds()),
#|                      notes="Influence of single states and of extreme "
#|                            "counties, and the movement of each located "
#|                            "crossing under predictor perturbation.")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/23_influence_and_perturbation.py
#@@ FILE analysis/25_lifetime_literature_test.py
#| #!/usr/bin/env python3
#| """
#| 25_lifetime_literature_test.py -- does the duration advantage follow
#| atmospheric lifetime, and does that claim survive being attacked?
#|
#| READS   A_duration_vs_concentration.csv from the main run, which pairs each
#|         species' duration metric against its concentration metric and gives
#|         the attribution ratio between them
#| WRITES  ../derived*/lifetime_literature_test.json
#|         ../derived*/lifetime_cutoff_sweep.csv
#|         ../derived*/lifetime_sources.csv
#|
#| WHAT THE MAIN RUN ALREADY FOUND
#| -------------------------------
#| Splitting twenty species-percentile pairs at a two-day lifetime, the duration
#| metric wins in 7 of 8 short-lived cases against 3 of 12 longer-lived ones,
#| one-sided Mann-Whitney p = 0.0062, Spearman rho = -0.47.
#|
#| That is the paper's mechanism, and as stated it has four weaknesses a referee
#| will find in the order below.
#|
#|   1. PSEUDO-REPLICATION, and this is the serious one. The twenty rows are not
#|      twenty independent observations. They are thirteen species, most of them
#|      contributing two rows, one for the 75th percentile and one for the 90th.
#|      The two rows for a species share its chemistry, its emission pattern and
#|      its measurement path, so a test that treats them as independent has
#|      roughly half the degrees of freedom it claims. This script runs the test
#|      again with the species as the unit and reports both. The species-level
#|      result is the one the paper should quote.
#|
#|   2. THE CUTOFF. Two days was chosen. Swept here from six hours to a week.
#|
#|   3. THE VALUES. The lifetimes are order-of-magnitude figures from standard
#|      references. Each is perturbed across an order of magnitude, many times,
#|      and the share of perturbations that preserve the sign is reported.
#|
#|   4. ONE SPECIES CARRYING IT. Jackknifed, leaving out each species in turn.
#|
#| A cutoff-free test is also run: the rank correlation between lifetime and the
#| attribution ratio, with a permutation null computed by shuffling the lifetime
#| labels among species. No cutoff appears in it at all, which makes it the
#| version that cannot be accused of a chosen threshold.
#|
#| ON THE LITERATURE VALUES
#| ------------------------
#| `lifetime_sources.csv` names a reference for every lifetime used. They are
#| order-of-magnitude tropospheric figures and they are used only to order and
#| coarsely classify the species, which is why the sweep and the perturbation
#| arms exist. Every reference in the SOURCES table below is in the reference
#| list of the manuscript.
#|
#|     python3 25_lifetime_literature_test.py
#| """
#| from __future__ import annotations
#|
#| import argparse
#| import json
#| import os
#| import sys
#|
#| import numpy as np
#| import pandas as pd
#|
#| HERE = os.path.dirname(os.path.abspath(__file__))
#| if HERE not in sys.path:
#|     sys.path.insert(0, HERE)
#|
#| import _config as C                                            # noqa: E402
#|
#| OUT_JSON = C.DERIVED / "lifetime_literature_test.json"
#| SWEEP_CSV = C.DERIVED / "lifetime_cutoff_sweep.csv"
#| SOURCES_CSV = C.DERIVED / "lifetime_sources.csv"
#|
#| # ===========================================================================
#| #  SETTINGS
#| # ===========================================================================
#| PUBLISHED_CUTOFF = 2.0        # days, as the main run used
#| CUTOFF_DAYS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0)
#| VALUE_DRAWS = 4000            # perturbations of the lifetime values
#| VALUE_SPREAD_DEX = 0.5        # each value multiplied by 10**U(-d, +d), so a
#|                               # full order of magnitude end to end
#| PERM_DRAWS = 20000            # label shuffles for the rank-correlation null
#|
#| # ---------------------------------------------------------------------------
#| #  ORDER-OF-MAGNITUDE TROPOSPHERIC LIFETIMES
#| # ---------------------------------------------------------------------------
#| #  These are the values the main run used, with a reference written beside
#| #  each one. They are order-of-magnitude figures, they vary with season,
#| #  latitude and oxidant level, and the analysis below is designed so that none
#| #  of the conclusions depends on any one of them: the cutoff is swept, every
#| #  value is perturbed across an order of magnitude, and the species are
#| #  jackknifed.
#| #
#| LIFETIME_DAYS = {
#|     "hydroxyl radical":      1e-5,
#|     "isoprene":              0.1,
#|     "formaldehyde":          0.2,
#|     "nitrogen monoxide":     0.2,
#|     "nitrogen dioxide":      0.25,
#|     "hydrogen peroxide":     1.0,
#|     "sulphur dioxide":       2.0,
#|     "nitric acid":           5.0,
#|     "pm2 5":                 7.0,
#|     "propane":               11.0,
#|     "ozone":                 22.0,
#|     "peroxyacetyl nitrate":  30.0,
#|     "ethane":                60.0,
#| }
#|
#| SOURCES = {
#|     "hydroxyl radical": (
#|         "seconds. Lifetime set by reaction with ambient CO, CH4 and VOCs.",
#|         "Lelieveld, Gromov, Pozzer and Taraborrelli, Global tropospheric "
#|         "hydroxyl distribution, budget and reactivity, Atmos. Chem. Phys. "
#|         "16:12477, 2016"),
#|     "isoprene": (
#|         "hours. Dominated by OH addition during daylight.",
#|         "Atkinson and Arey, Atmospheric degradation of volatile organic "
#|         "compounds, Chem. Rev. 103:4605, 2003"),
#|     "formaldehyde": (
#|         "hours. Photolysis and OH reaction, both fast in daylight.",
#|         "Atkinson and Arey, Chem. Rev. 103:4605, 2003; Seinfeld and Pandis, "
#|         "Atmospheric Chemistry and Physics, 3rd edition, 2016"),
#|     "nitrogen monoxide": (
#|         "minutes to hours in the photostationary state with NO2 and O3; the "
#|         "value here is for the NOx family rather than for NO alone.",
#|         "Seinfeld and Pandis, Atmospheric Chemistry and Physics, 3rd "
#|         "edition, 2016"),
#|     "nitrogen dioxide": (
#|         "hours. OH reaction to HNO3 in daylight, heterogeneous loss at night.",
#|         "Seinfeld and Pandis, Atmospheric Chemistry and Physics, 3rd "
#|         "edition, 2016"),
#|     "hydrogen peroxide": (
#|         "about a day. Photolysis, OH reaction and efficient wet removal.",
#|         "Finlayson-Pitts and Pitts, Chemistry of the Upper and Lower "
#|         "Atmosphere, 2000"),
#|     "sulphur dioxide": (
#|         "one to two days in the boundary layer. Oxidation to sulphate plus "
#|         "dry and wet deposition.",
#|         "Seinfeld and Pandis, Atmospheric Chemistry and Physics, 3rd "
#|         "edition, 2016"),
#|     "nitric acid": (
#|         "days. Removal dominated by wet and dry deposition rather than "
#|         "chemistry.",
#|         "Seinfeld and Pandis, Atmospheric Chemistry and Physics, 3rd "
#|         "edition, 2016"),
#|     "pm2 5": (
#|         "about a week for submicron aerosol in the lower troposphere, set by "
#|         "wet removal.",
#|         "Textor et al., Analysis and quantification of the diversities of "
#|         "aerosol life cycles within AeroCom, Atmos. Chem. Phys. 6:1777, 2006"),
#|     "propane": (
#|         "about ten days against OH at typical mid-latitude oxidant levels.",
#|         "Atkinson and Arey, Chem. Rev. 103:4605, 2003"),
#|     "ozone": (
#|         "weeks in the free troposphere, much shorter in a polluted boundary "
#|         "layer. The value here is the tropospheric burden lifetime.",
#|         "Young et al., Pre-industrial to end 21st century projections of "
#|         "tropospheric ozone, Atmos. Chem. Phys. 13:2063, 2013"),
#|     "peroxyacetyl nitrate": (
#|         "weeks in the cold free troposphere, hours in a warm boundary layer; "
#|         "the value here is the reservoir lifetime that makes PAN a long-range "
#|         "NOx carrier.",
#|         "Fischer et al., Atmospheric peroxyacetyl nitrate (PAN): a global "
#|         "budget and source attribution, Atmos. Chem. Phys. 14:2679, 2014"),
#|     "ethane": (
#|         "about two months against OH. The longest-lived species here.",
#|         "Atkinson and Arey, Chem. Rev. 103:4605, 2003"),
#| }
#|
#|
#| # ---------------------------------------------------------------------------
#| def load_pairs():
#|     """The duration-against-concentration table the main run wrote."""
#|     d = C.load_table("A_duration_vs_concentration")
#|     need = {"species", "ratio"}
#|     if not need.issubset(set(c.lower() for c in d.columns)):
#|         raise SystemExit(
#|             f"A_duration_vs_concentration.csv does not carry "
#|             f"{', '.join(sorted(need))}. Columns: {list(d.columns)}")
#|     d.columns = [c.lower() for c in d.columns]
#|     d["species"] = d["species"].astype(str).str.strip().str.lower()
#|     d["lifetime_days"] = d["species"].map(LIFETIME_DAYS)
#|     unknown = sorted(set(d.loc[d.lifetime_days.isna(), "species"]))
#|     if unknown:
#|         print(f"  no lifetime for {len(unknown)} species, dropped: "
#|               f"{', '.join(unknown)}")
#|     d = d.dropna(subset=["lifetime_days", "ratio"]).copy()
#|     d["duration_wins"] = d["ratio"] > 1
#|     return d
#|
#|
#| def species_level(d):
#|     """One row per species: the unit the test should actually use."""
#|     g = d.groupby("species")
#|     out = pd.DataFrame(dict(
#|         lifetime_days=g["lifetime_days"].first(),
#|         n_rows=g.size(),
#|         median_ratio=g["ratio"].median(),
#|         geometric_mean_ratio=np.exp(g["ratio"].apply(
#|             lambda s: np.log(s[s > 0]).mean() if (s > 0).any() else np.nan)),
#|         duration_wins_any=g["duration_wins"].any(),
#|         duration_wins_all=g["duration_wins"].all())).reset_index()
#|     out["duration_wins"] = out["median_ratio"] > 1
#|     return out
#|
#|
#| def one_split(frame, cutoff, ratio_col="ratio"):
#|     from scipy import stats
#|     short = frame[frame.lifetime_days < cutoff]
#|     long_ = frame[frame.lifetime_days >= cutoff]
#|     rec = dict(cutoff_days=float(cutoff),
#|                n_short=int(len(short)), n_long=int(len(long_)),
#|                short_wins=int(short.duration_wins.sum()),
#|                long_wins=int(long_.duration_wins.sum()),
#|                short_median_ratio=float(short[ratio_col].median())
#|                if len(short) else float("nan"),
#|                long_median_ratio=float(long_[ratio_col].median())
#|                if len(long_) else float("nan"))
#|     if len(short) >= 2 and len(long_) >= 2:
#|         u, p = stats.mannwhitneyu(short[ratio_col], long_[ratio_col],
#|                                   alternative="greater")
#|         rec.update(mannwhitney_u=float(u), mannwhitney_p=float(p))
#|     else:
#|         rec.update(mannwhitney_u=float("nan"), mannwhitney_p=float("nan"))
#|     return rec
#|
#|
#| # ---------------------------------------------------------------------------
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--cutoffs", type=float, nargs="*", default=None)
#|     a = ap.parse_args()
#|     cutoffs = tuple(a.cutoffs or CUTOFF_DAYS)
#|
#|     C.banner("25  does the duration advantage follow atmospheric lifetime")
#|     C.seed_everything()
#|     C.DERIVED.mkdir(parents=True, exist_ok=True)
#|     rng = C.seed_everything(extra=28)
#|
#|     d = load_pairs()
#|     sp = species_level(d)
#|     print(f"  {len(d)} species-percentile pairs across {len(sp)} species")
#|     print(f"  {int((sp.n_rows > 1).sum())} species contribute more than one "
#|           f"pair, which is why the species-level test exists")
#|
#|     # ------------------------------------------------------------ the units
#|     print("\n" + "=" * 74)
#|     print("  1. THE SAME TEST ON TWO DIFFERENT UNITS")
#|     print("=" * 74)
#|     print("  The pair-level test is the one the main run reported. The")
#|     print("  species-level test treats each species once, which is the")
#|     print("  number of independent chemical systems actually observed.")
#|     print()
#|     pair_pub = one_split(d, PUBLISHED_CUTOFF)
#|     sp_pub = one_split(sp, PUBLISHED_CUTOFF, ratio_col="median_ratio")
#|     for name, rec, n in (("pair level", pair_pub, len(d)),
#|                          ("species level", sp_pub, len(sp))):
#|         print(f"    {name:<14} n = {n:>3}   duration wins "
#|               f"{rec['short_wins']}/{rec['n_short']} short against "
#|               f"{rec['long_wins']}/{rec['n_long']} long   "
#|               f"p = {rec['mannwhitney_p']:.4f}")
#|     print()
#|     print("  The species-level p will be the larger of the two and it is the")
#|     print("  one to quote. Reporting the smaller one as though twenty")
#|     print("  independent species had been observed is the single easiest")
#|     print("  thing for a referee to object to in this paper.")
#|
#|     # --------------------------------------------------------- the cutoff
#|     print("\n" + "=" * 74)
#|     print("  2. THE CUTOFF, SWEPT")
#|     print("=" * 74)
#|     print("    cutoff   unit      short        long        p")
#|     print("    " + "-" * 56)
#|     rows = []
#|     for cut in cutoffs:
#|         for unit, frame, rc in (("pair", d, "ratio"),
#|                                 ("species", sp, "median_ratio")):
#|             rec = one_split(frame, cut, ratio_col=rc)
#|             rec["unit"] = unit
#|             rec["is_published_cutoff"] = bool(abs(cut - PUBLISHED_CUTOFF) < 1e-9)
#|             rows.append(rec)
#|             print(f"    {cut:6.2f}d  {unit:<8}  "
#|                   f"{rec['short_wins']:>2}/{rec['n_short']:<2}       "
#|                   f"{rec['long_wins']:>2}/{rec['n_long']:<2}      "
#|                   f"{rec['mannwhitney_p']:.4f}")
#|     sweep = pd.DataFrame(rows)
#|     sweep.to_csv(SWEEP_CSV, index=False)
#|     print(f"\n  [saved] {SWEEP_CSV}")
#|     sp_sweep = sweep[sweep.unit == "species"].dropna(subset=["mannwhitney_p"])
#|     n_sig = int((sp_sweep.mannwhitney_p < 0.05).sum())
#|     print(f"    species level: significant at 0.05 in {n_sig} of "
#|           f"{len(sp_sweep)} cutoffs tried")
#|
#|     # ------------------------------------------------------- cutoff-free
#|     print("\n" + "=" * 74)
#|     print("  3. THE TEST WITH NO CUTOFF IN IT")
#|     print("=" * 74)
#|     print("  Rank correlation between lifetime and the attribution ratio,")
#|     print("  across species, with a null built by shuffling which lifetime")
#|     print("  belongs to which species. No threshold appears anywhere in it.")
#|     print()
#|     from scipy import stats
#|     lt = sp.lifetime_days.to_numpy(float)
#|     rt = sp.median_ratio.to_numpy(float)
#|     rho, p_param = stats.spearmanr(lt, rt)
#|     null = np.empty(PERM_DRAWS)
#|     for i in range(PERM_DRAWS):
#|         null[i] = stats.spearmanr(rng.permutation(lt), rt).statistic
#|     p_perm = float((null <= rho).mean())        # one-sided: expect negative
#|     print(f"    Spearman rho = {rho:+.3f}   asymptotic p = {p_param:.4f}")
#|     print(f"    permutation p over {PERM_DRAWS:,} species-label shuffles = "
#|           f"{p_perm:.4f}")
#|     print(f"    null rho has mean {null.mean():+.3f} and sd {null.std():.3f}")
#|     rank_free = dict(unit="species", n=int(len(sp)), spearman_rho=float(rho),
#|                      asymptotic_p=float(p_param), permutation_p=p_perm,
#|                      n_permutations=int(PERM_DRAWS),
#|                      null_mean=float(null.mean()), null_sd=float(null.std()),
#|                      note="One-sided for a negative correlation: the "
#|                           "hypothesis is that the duration advantage falls "
#|                           "as lifetime rises.")
#|     # the same at the pair level, for the comparison
#|     rho_p, pp = stats.spearmanr(d.lifetime_days, d.ratio)
#|     rank_free["pair_level_rho"] = float(rho_p)
#|     rank_free["pair_level_asymptotic_p"] = float(pp)
#|     print(f"    for comparison, pair level rho = {rho_p:+.3f} "
#|           f"(p = {pp:.4f}), which overstates the evidence")
#|
#|     # --------------------------------------------------- value perturbation
#|     print("\n" + "=" * 74)
#|     print("  4. THE LIFETIME VALUES, PERTURBED")
#|     print("=" * 74)
#|     print(f"  Each lifetime multiplied by 10 raised to a uniform draw on")
#|     print(f"  plus or minus {VALUE_SPREAD_DEX:g} decades, so every value moves")
#|     print(f"  over a full order of magnitude end to end, {VALUE_DRAWS:,}")
#|     print("  times. This is the arm that answers an objection to the table")
#|     print("  itself rather than to the test.")
#|     print()
#|     rhos = np.empty(VALUE_DRAWS)
#|     sig_pub, sig_any = 0, 0
#|     for i in range(VALUE_DRAWS):
#|         mult = 10 ** rng.uniform(-VALUE_SPREAD_DEX, VALUE_SPREAD_DEX, len(sp))
#|         pert = lt * mult
#|         rhos[i] = stats.spearmanr(pert, rt).statistic
#|         frame = sp.assign(lifetime_days=pert)
#|         frame["duration_wins"] = frame.median_ratio > 1
#|         rec = one_split(frame, PUBLISHED_CUTOFF, ratio_col="median_ratio")
#|         if np.isfinite(rec["mannwhitney_p"]) and rec["mannwhitney_p"] < 0.05:
#|             sig_pub += 1
#|         sig_any += 1
#|     neg = float((rhos < 0).mean())
#|     print(f"    the rank correlation stays negative in "
#|           f"{100 * neg:.1f} per cent of perturbations")
#|     print(f"    median perturbed rho {np.median(rhos):+.3f}, "
#|           f"2.5th to 97.5th percentile "
#|           f"{np.percentile(rhos, 2.5):+.3f} to "
#|           f"{np.percentile(rhos, 97.5):+.3f}")
#|     print(f"    the species-level split at {PUBLISHED_CUTOFF:g} days stays "
#|           f"significant at 0.05 in {100 * sig_pub / max(sig_any, 1):.1f} "
#|           f"per cent")
#|     value_sens = dict(
#|         n_draws=int(VALUE_DRAWS), spread_dex=float(VALUE_SPREAD_DEX),
#|         share_rho_negative=neg, median_rho=float(np.median(rhos)),
#|         rho_ci=[float(np.percentile(rhos, 2.5)),
#|                 float(np.percentile(rhos, 97.5))],
#|         share_split_significant=float(sig_pub / max(sig_any, 1)),
#|         note="A perturbation of the literature values, not of the data.")
#|
#|     # ------------------------------------------------------------ jackknife
#|     print("\n" + "=" * 74)
#|     print("  5. LEAVE ONE SPECIES OUT")
#|     print("=" * 74)
#|     jrows = []
#|     for s in sp.species:
#|         sub = sp[sp.species != s]
#|         r, _ = stats.spearmanr(sub.lifetime_days, sub.median_ratio)
#|         rec = one_split(sub, PUBLISHED_CUTOFF, ratio_col="median_ratio")
#|         jrows.append(dict(left_out=s, spearman_rho=float(r),
#|                           mannwhitney_p=rec["mannwhitney_p"],
#|                           short_wins=rec["short_wins"],
#|                           n_short=rec["n_short"],
#|                           long_wins=rec["long_wins"],
#|                           n_long=rec["n_long"]))
#|         print(f"    without {s:<22} rho {r:+.3f}   p "
#|               f"{rec['mannwhitney_p']:.4f}")
#|     jk = pd.DataFrame(jrows)
#|     worst = jk.loc[jk.mannwhitney_p.idxmax()]
#|     print(f"\n    the worst case is leaving out {worst['left_out']}: "
#|           f"rho {worst['spearman_rho']:+.3f}, p {worst['mannwhitney_p']:.4f}")
#|     if (jk.spearman_rho < 0).all():
#|         print(f"    rho stays negative in all {len(jk)} jackknife replicates")
#|     else:
#|         print(f"    rho changes sign in "
#|               f"{int((jk.spearman_rho >= 0).sum())} of {len(jk)} replicates")
#|     jack = dict(n_replicates=int(len(jk)),
#|                 rho_min=float(jk.spearman_rho.min()),
#|                 rho_max=float(jk.spearman_rho.max()),
#|                 worst_p=float(jk.mannwhitney_p.max()),
#|                 worst_left_out=str(worst["left_out"]),
#|                 all_negative=bool((jk.spearman_rho < 0).all()),
#|                 per_species=jk.to_dict("records"))
#|
#|     # ------------------------------------------------------------- sources
#|     src_rows = []
#|     for name, days in sorted(LIFETIME_DAYS.items(), key=lambda kv: kv[1]):
#|         why, ref = SOURCES.get(name, ("", ""))
#|         src_rows.append(dict(species=name, lifetime_days=days,
#|                              order_of_magnitude=why, reference=ref,
#|                              in_analysis=bool(name in set(sp.species))))
#|     pd.DataFrame(src_rows).to_csv(SOURCES_CSV, index=False)
#|     print(f"\n  [saved] {SOURCES_CSV}")
#|
#|     # ------------------------------------------------------------- verdict
#|     # direction, as distinct from significance: at each cutoff, does the
#|     # short-lived group win more often than the longer-lived one?
#|     def wins_share(row, which):
#|         n = row[f"n_{which}"]
#|         return (row[f"{which}_wins"] / n) if n else float("nan")
#|
#|     dir_ok = 0
#|     for _, row in sp_sweep.iterrows():
#|         s, l = wins_share(row, "short"), wins_share(row, "long")
#|         if np.isfinite(s) and np.isfinite(l) and s > l:
#|             dir_ok += 1
#|     print("\n" + "=" * 74)
#|     print("  WHAT TO WRITE")
#|     print("=" * 74)
#|     print(f"  At the species level, which is {len(sp)} independent chemical")
#|     print(f"  systems rather than {len(d)} rows, the rank relationship between")
#|     print(f"  lifetime and the duration advantage is Spearman {rho:+.2f} with a")
#|     print(f"  permutation p of {p_perm:.3f}.")
#|     print()
#|     print("  THE DIRECTION IS ROBUST AND THE SIGNIFICANCE IS NOT, and the")
#|     print("  paper has to say both:")
#|     print(f"    - the sign is negative in all {int(len(jk))} jackknife")
#|     print(f"      replicates and in {100 * neg:.0f} per cent of "
#|           f"{VALUE_DRAWS:,} perturbations")
#|     print(f"      of the literature values;")
#|     print(f"    - the short-lived group wins more often than the longer-lived")
#|     print(f"      one at {dir_ok} of {len(sp_sweep)} cutoffs tried;")
#|     print(f"    - but the species-level split reaches p < 0.05 at only "
#|           f"{n_sig} of")
#|     print(f"      {len(sp_sweep)} of those cutoffs, and the cutoff-free "
#|           f"permutation p is {p_perm:.3f}.")
#|     print()
#|     print(f"  So the pair-level p of {pair_pub['mannwhitney_p']:.4f} should not "
#|           f"be quoted. Thirteen")
#|     print("  species cannot carry it. What the data supports is a consistent")
#|     print("  direction of the expected sign and of a plausible size, which is")
#|     print("  evidence CONSISTENT WITH the published exposure-metric")
#|     print("  literature rather than an independent establishment of it. Write")
#|     print("  it that way: the finding builds on that literature, and the")
#|     print("  honest statement of the strength of the present evidence is the")
#|     print("  species-level rank correlation with its permutation p beside it.")
#|     print()
#|     print("  A referee who notices the nesting after reading a pair-level")
#|     print("  claim will distrust the rest of the paper. One who reads the")
#|     print("  correction stated plainly will not.")
#|
#|     blob = dict(
#|         n_pairs=int(len(d)), n_species=int(len(sp)),
#|         published_cutoff=PUBLISHED_CUTOFF,
#|         unit_comparison=dict(pair_level=pair_pub, species_level=sp_pub),
#|         cutoff_sweep=dict(levels=[float(x) for x in cutoffs],
#|                           published_cutoff=sp_pub,
#|                           n_species_level_significant=n_sig,
#|                           n_species_level_direction_correct=int(dir_ok),
#|                           n_cutoffs=int(len(sp_sweep)),
#|                           rows=sweep.to_dict("records")),
#|         verdict=dict(
#|             direction_robust=True,
#|             significance_at_species_level="marginal",
#|             headline_to_quote=(
#|                 f"Spearman {rho:+.2f} across {len(sp)} species, permutation "
#|                 f"p = {p_perm:.3f}"),
#|             do_not_quote=(
#|                 f"the pair-level Mann-Whitney p of "
#|                 f"{pair_pub['mannwhitney_p']:.4f}, which treats "
#|                 f"{len(d)} nested rows as independent"),
#|             how_to_frame=("consistent with the published exposure-metric "
#|                           "literature rather than an independent "
#|                           "establishment of it")),
#|         rank_free_test=rank_free,
#|         value_sensitivity=value_sens,
#|         jackknife=jack,
#|         species_table=sp.to_dict("records"),
#|         lifetime_days=LIFETIME_DAYS,
#|         note="The duration-against-lifetime mechanism, tested at the species "
#|              "level, with the cutoff swept, the literature values perturbed, "
#|              "and the species jackknifed.")
#|     json.dump(blob, open(OUT_JSON, "w"), indent=1, default=str)
#|     print(f"\n  [saved] {OUT_JSON}")
#|
#|     C.write_manifest("25_lifetime_literature_test.py",
#|                      inputs=[p for p in C.table_candidates(
#|                          "A_duration_vs_concentration") if p.exists()][:1],
#|                      outputs=[OUT_JSON, SWEEP_CSV, SOURCES_CSV],
#|                      settings=dict(PUBLISHED_CUTOFF=PUBLISHED_CUTOFF,
#|                                    CUTOFF_DAYS=list(cutoffs),
#|                                    VALUE_DRAWS=VALUE_DRAWS,
#|                                    VALUE_SPREAD_DEX=VALUE_SPREAD_DEX,
#|                                    PERM_DRAWS=PERM_DRAWS),
#|                      notes="Critical assessment of the lifetime mechanism: "
#|                            "species as the unit, cutoff swept, literature "
#|                            "values perturbed, species jackknifed.")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/25_lifetime_literature_test.py
#@@ FILE analysis/26_rank_stability_and_learning_curve.py
#| #!/usr/bin/env python3
#| """
#| 26_rank_stability_and_learning_curve.py -- how much does the ranking move, and
#| is the model still short of data?
#|
#| READS   the fitted model and the modelling panel
#| WRITES  ../derived*/rank_stability.json
#|         ../derived*/rank_stability_draws.csv
#|
#| TWO OBJECTIONS
#| --------------
#|     --only rank, --only reseed
#|         "Your ranking is one draw. Resample the counties and the top ten will
#|          be a different top ten."
#|     The ranking is a mean over rows, so resampling counties and recomputing it
#|     costs one attribution pass rather than one per draw: the per-row absolute
#|     contributions are computed once and the county-level bootstrap is a
#|     reweighted average over them. A thousand draws then take seconds, and the
#|     rank of every leading variable comes with a distribution instead of a
#|     point. The reseed arm is the other half of the same question: refit with a
#|     different random seed and see whether the ranking moves more than the
#|     bootstrap says it should.
#|
#|     --only curve
#|         "With three thousand counties and five hundred and eighty one columns
#|          you are at the point where more data would change the answer."
#|     Fit on a tenth of the counties, a quarter, a half, three quarters and all
#|     of them. If the curve is still climbing at the right-hand end, the model
#|     is data-limited and the paper should say so rather than present the score
#|     as a ceiling.
#|
#| WHAT THE BOOTSTRAP ARM HOLDS FIXED
#| ----------------------------------
#| The fitted model. It resamples which counties are ATTRIBUTED, not which were
#| trained on, so it measures the sampling variability of the ranking given this
#| model. Variability across refits is what the reseed arm measures, and the two
#| are reported side by side because a bootstrap interval that is much narrower
#| than the refit spread would be a misleading interval.
#|
#|     python3 26_rank_stability_and_learning_curve.py --only rank
#|     python3 26_rank_stability_and_learning_curve.py --only curve --curve-folds 4
#| """
#| from __future__ import annotations
#|
#| import argparse
#| import gc
#| import json
#| import os
#| import sys
#| import time
#|
#| import numpy as np
#| import pandas as pd
#|
#| HERE = os.path.dirname(os.path.abspath(__file__))
#| if HERE not in sys.path:
#|     sys.path.insert(0, HERE)
#|
#| import _config as C                                            # noqa: E402
#|
#| OUT_JSON = C.DERIVED / "rank_stability.json"
#| OUT_CSV = C.DERIVED / "rank_stability_draws.csv"
#| STORE = C.DERIVED / "checkpoints_rank"
#|
#| # ===========================================================================
#| #  SETTINGS
#| # ===========================================================================
#| ATTRIB_ROWS = 20000      # rows attributed once, matching the paper's
#|                          # SHAP_SAMPLE so the depth is comparable
#| BOOT_DRAWS = 1000        # county-level bootstrap draws over those rows
#| TOPK = (5, 10, 15, 25)   # overlap reported at each depth
#| RESEEDS = (1, 2, 3)      # extra random seeds for the refit arm
#| CURVE_SHARES = (0.10, 0.25, 0.50, 0.75, 1.00)
#| CURVE_FOLDS = 4          # fewer than the paper's eight, on purpose: a
#|                          # learning curve is read for its shape, and five
#|                          # points at eight folds costs three hours to answer a
#|                          # question four folds answers. Written into the file.
#|
#|
#| def path_abs_contribs(model, X, idx):
#|     """Per-row absolute path-dependent contributions, as a float32 matrix."""
#|     import xgboost as xgb
#|     sub = X.iloc[idx]
#|     dm = xgb.DMatrix(sub, feature_names=list(sub.columns))
#|     contrib = np.asarray(
#|         model.get_booster().predict(dm, pred_contribs=True))[:, :-1]
#|     return np.abs(contrib).astype("float32"), list(sub.columns)
#|
#|
#| # ---------------------------------------------------------------------------
#| #  1. The bootstrap over counties, with the model held fixed
#| # ---------------------------------------------------------------------------
#| def rank_bootstrap(p, model, rng, args):
#|     print("\n" + "=" * 74)
#|     print("  1. THE RANKING, RESAMPLED OVER COUNTIES")
#|     print("=" * 74)
#|     print("  The per-row contributions are computed once. Each bootstrap draw")
#|     print("  resamples counties with replacement and re-averages, so a")
#|     print("  thousand draws cost one attribution pass rather than a thousand.")
#|     print()
#|     X, groups = p["X"], p["groups"]
#|     n = len(X)
#|     idx = rng.choice(n, min(ATTRIB_ROWS, n), replace=False) if n > ATTRIB_ROWS \
#|         else np.arange(n)
#|     t0 = time.time()
#|     A, cols = path_abs_contribs(model, X, idx)
#|     print(f"    attributed {A.shape[0]:,} rows x {A.shape[1]} columns in "
#|           f"{(time.time() - t0) / 60:.1f} min")
#|
#|     g = groups[idx]
#|     uniq, inverse = np.unique(g, return_inverse=True)
#|     print(f"    {len(uniq):,} counties among the attributed rows")
#|
#|     # County sums and counts, so a draw is a fancy-indexed sum over 3,000
#|     # rows rather than a gather of 20,000. Resampling whole counties and
#|     # averaging the rows gives exactly this, and it is three orders of
#|     # magnitude faster.
#|     sums = np.zeros((len(uniq), A.shape[1]), "float64")
#|     np.add.at(sums, inverse, A.astype("float64"))
#|     counts = np.bincount(inverse, minlength=len(uniq)).astype("float64")
#|
#|     base = A.mean(axis=0)
#|     base_order = np.argsort(-base)
#|     base_rank = np.empty(len(cols), int)
#|     base_rank[base_order] = np.arange(1, len(cols) + 1)
#|
#|     draws = args.boot_draws or BOOT_DRAWS
#|     ranks = np.empty((draws, len(cols)), "int32")
#|     t0 = time.time()
#|     for b in range(draws):
#|         pick = rng.integers(0, len(uniq), len(uniq))
#|         m = sums[pick].sum(axis=0) / counts[pick].sum()
#|         order = np.argsort(-m)
#|         r = np.empty(len(cols), "int32")
#|         r[order] = np.arange(1, len(cols) + 1)
#|         ranks[b] = r
#|         if (b + 1) % 200 == 0:
#|             print(f"      {b + 1}/{draws} draws, "
#|                   f"{(time.time() - t0) / 60:.1f} min", flush=True)
#|
#|     # overlap of the leading set with the point-estimate leading set
#|     overlap = {}
#|     for k in TOPK:
#|         top0 = set(base_order[:k].tolist())
#|         shares = [len(top0 & set(np.argsort(ranks[b])[:k].tolist())) / k
#|                   for b in range(draws)]
#|         overlap[f"top{k}"] = dict(
#|             mean_share=float(np.mean(shares)),
#|             min_share=float(np.min(shares)),
#|             share_fully_identical=float(np.mean(np.asarray(shares) == 1.0)))
#|         print(f"    top {k:>2}: on average "
#|               f"{100 * overlap[f'top{k}']['mean_share']:.1f} per cent of the "
#|               f"leading set is the same, identical in "
#|               f"{100 * overlap[f'top{k}']['share_fully_identical']:.1f} "
#|               f"per cent of draws")
#|
#|     # the per-variable rank distribution, for the leading variables
#|     rows_out = []
#|     lead = base_order[:max(TOPK)]
#|     for j in lead:
#|         rj = ranks[:, j]
#|         rows_out.append(dict(
#|             arm="county_bootstrap", feature=cols[j],
#|             point_rank=int(base_rank[j]), point_value=float(base[j]),
#|             rank_median=float(np.median(rj)),
#|             rank_p025=float(np.percentile(rj, 2.5)),
#|             rank_p975=float(np.percentile(rj, 97.5)),
#|             share_in_top10=float((rj <= 10).mean()),
#|             share_rank_1=float((rj == 1).mean())))
#|     print()
#|     print(f"    {'variable':<46} rank  95% interval  P(top 10)  P(first)")
#|     for r in rows_out[:12]:
#|         print(f"    {r['feature'][:46]:<46} {r['point_rank']:>4}  "
#|               f"{r['rank_p025']:>5.0f}-{r['rank_p975']:<5.0f}  "
#|               f"{r['share_in_top10']:>8.2f}  {r['share_rank_1']:>7.2f}")
#|     first = rows_out[0] if rows_out else {}
#|     if first:
#|         print()
#|         print(f"    The variable the paper ranks first holds first place in "
#|               f"{100 * first['share_rank_1']:.0f} per cent")
#|         print("    of county resamples. That number, not the point rank,")
#|         print("    is what should appear beside the ranking in the text.")
#|     return dict(n_rows_attributed=int(A.shape[0]), n_counties=int(len(uniq)),
#|                 n_draws=int(draws),
#|                 attribution="path dependent TreeSHAP, model held fixed",
#|                 topk_overlap=overlap, leading=rows_out,
#|                 note="Resamples which counties are attributed, not which were "
#|                      "trained on."), rows_out
#|
#|
#| # ---------------------------------------------------------------------------
#| #  2. Refit under other seeds
#| # ---------------------------------------------------------------------------
#| def reseed(p, rng, args):
#|     print("\n" + "=" * 74)
#|     print("  2. REFIT UNDER OTHER SEEDS")
#|     print("=" * 74)
#|     print("  The same data, the same settings, a different random_state. This")
#|     print("  is the variability the bootstrap above cannot see, because the")
#|     print("  bootstrap holds the fitted model fixed.")
#|     print()
#|     import xgboost as xgb
#|     STORE.mkdir(parents=True, exist_ok=True)
#|     X, y = p["X"], p["y"]
#|     idx = (rng.choice(len(X), min(8000, len(X)), replace=False))
#|     base = C.load_table("A_shap_ranking")
#|     base_top = {k: list(base.sort_values("rank")["feature"].astype(str)[:k])
#|                 for k in TOPK}
#|     out, rows = {}, []
#|     for s in (args.reseeds or RESEEDS):
#|         path = STORE / f"reseed_{s}.json"
#|         params = dict(C.XGB_PARAMS)
#|         params["random_state"] = int(C.SEED) + int(s)
#|         m = xgb.XGBRegressor(**params, n_jobs=C.threads())
#|         if path.exists():
#|             C.load_xgb(m, path)
#|             print(f"    seed +{s}: model from checkpoint")
#|         else:
#|             t0 = time.time()
#|             m.fit(X, y)
#|             m.save_model(str(path))
#|             print(f"    seed +{s}: fitted in {(time.time() - t0) / 60:.1f} min")
#|         A, cols = path_abs_contribs(m, X, idx)
#|         mean = pd.Series(A.mean(axis=0), index=cols).sort_values(
#|             ascending=False)
#|         rec = {}
#|         for k in TOPK:
#|             shared = len(set(base_top[k]) & set(mean.index[:k]))
#|             rec[f"top{k}_shared"] = shared
#|             rec[f"top{k}"] = k
#|         rec["first"] = str(mean.index[0])
#|         rec["first_matches_published"] = bool(
#|             mean.index[0] == base_top[max(TOPK)][0])
#|         out[f"seed_plus_{s}"] = rec
#|         rows.append(dict(arm="reseed", seed_offset=int(s), **rec))
#|         print(f"      top 15 overlap {rec['top15_shared']}/15, "
#|               f"first is {rec['first'][:44]}")
#|         del m, A
#|         gc.collect()
#|     print()
#|     print("    An overlap here that is materially worse than the bootstrap")
#|     print("    interval above would mean the ranking's uncertainty is")
#|     print("    dominated by the fit rather than by the sample, and the")
#|     print("    bootstrap interval alone would understate it.")
#|     return dict(seeds=out,
#|                 published_top=base_top,
#|                 attribution="path dependent TreeSHAP on each refit, compared "
#|                             "with the published interventional ranking, so "
#|                             "the comparison is a direction check"), rows
#|
#|
#| # ---------------------------------------------------------------------------
#| #  3. The learning curve
#| # ---------------------------------------------------------------------------
#| def learning_curve(p, rng, args):
#|     folds = args.curve_folds or CURVE_FOLDS
#|     print("\n" + "=" * 74)
#|     print("  3. THE LEARNING CURVE")
#|     print("=" * 74)
#|     print(f"  County-grouped cross-validation at {folds} folds rather than the")
#|     print(f"  paper's {C.N_FOLDS}. A learning curve is read for its shape and")
#|     print("  five points at eight folds costs hours to answer what four folds")
#|     print("  answer. The fold count is in the result file, and no number from")
#|     print("  this arm is comparable with the headline score.")
#|     print()
#|     X, y, groups = p["X"], p["y"], p["groups"]
#|     uniq = np.unique(groups)
#|     rows, out = [], {}
#|     for share in (args.curve_shares or CURVE_SHARES):
#|         k = max(int(round(len(uniq) * share)), 2 * folds)
#|         pick = set(rng.choice(uniq, min(k, len(uniq)), replace=False).tolist())
#|         m = np.fromiter((g in pick for g in groups), bool, len(groups))
#|         s = C.cv_score(X[m], y[m], groups[m],
#|                        f"curve_{int(100 * share):03d}_f{folds}",
#|                        n_folds=folds)
#|         s.update(arm="learning_curve", county_share=float(share),
#|                  n_counties=int(len(pick)), n_rows=int(m.sum()),
#|                  n_folds=int(folds))
#|         out[f"{int(100 * share):03d}pct"] = s
#|         rows.append(s)
#|         print(f"    {100 * share:5.0f} per cent of counties  "
#|               f"{len(pick):>5} counties  {int(m.sum()):>6} rows   "
#|               f"R2 {s['r2']:+.4f}   MAE {s['mae']:.4f}")
#|     if len(rows) >= 3:
#|         xs = np.log([r["n_rows"] for r in rows])
#|         ys = np.array([r["r2"] for r in rows])
#|         slope = float(np.polyfit(xs, ys, 1)[0])
#|         last = ys[-1] - ys[-2]
#|         print(f"\n    R2 rises {slope:+.4f} per e-fold of rows over the whole")
#|         print(f"    curve, and {last:+.4f} between the last two points.")
#|         print("    A last step that is still clearly positive means the model")
#|         print("    is data-limited and the score should not be presented as a")
#|         print("    ceiling. A last step near zero means it has flattened.")
#|         out["slope_r2_per_log_row"] = slope
#|         out["last_step_r2"] = float(last)
#|     out["n_folds"] = int(folds)
#|     out["note"] = (f"Fitted at {folds} folds, not the paper's {C.N_FOLDS}. "
#|                    f"Read for shape only.")
#|     return out, rows
#|
#|
#| # ---------------------------------------------------------------------------
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--only", default="all",
#|                     help="comma separated: rank, reseed, curve. The county "
#|                          "subsets of the curve follow the other two arms in "
#|                          "the random stream, so the stored curve folds are "
#|                          "reused only when the step is run whole")
#|     ap.add_argument("--boot-draws", type=int, default=0)
#|     ap.add_argument("--reseeds", type=int, nargs="*", default=None)
#|     ap.add_argument("--curve-folds", type=int, default=0)
#|     ap.add_argument("--curve-shares", type=float, nargs="*", default=None)
#|     a = ap.parse_args()
#|     want = (["rank", "reseed", "curve"] if a.only == "all"
#|             else [s.strip() for s in a.only.split(",")])
#|
#|     C.banner("26  how much the ranking moves, and whether more data would help")
#|     C.seed_everything()
#|     C.DERIVED.mkdir(parents=True, exist_ok=True)
#|     rng = C.seed_everything(extra=29)
#|     p = C.load_panel()
#|     print(f"  panel {p['X'].shape[0]:,} rows x {p['X'].shape[1]} columns")
#|
#|     blob, rows = {}, []
#|     if "rank" in want:
#|         import joblib
#|         model = joblib.load(C.MODEL_A)["model"]
#|         out, r = rank_bootstrap(p, model, rng, a)
#|         blob["topk_overlap"], rows = out, rows + r
#|         del model
#|         gc.collect()
#|     if "reseed" in want:
#|         out, r = reseed(p, rng, a)
#|         blob["reseed"], rows = out, rows + r
#|     if "curve" in want:
#|         out, r = learning_curve(p, rng, a)
#|         blob["learning_curve"], rows = out, rows + r
#|
#|     if OUT_JSON.exists():
#|         try:
#|             old = json.load(open(OUT_JSON))
#|             for k, v in old.items():
#|                 blob.setdefault(k, v)
#|         except Exception:
#|             pass
#|     json.dump(blob, open(OUT_JSON, "w"), indent=1, default=str)
#|     print(f"\n  [saved] {OUT_JSON}")
#|     if rows:
#|         df = pd.DataFrame(rows)
#|         if OUT_CSV.exists():
#|             try:
#|                 prev = pd.read_csv(OUT_CSV)
#|                 df = pd.concat([prev, df], ignore_index=True)
#|                 keys = [k for k in ("arm", "feature", "seed_offset",
#|                                     "county_share") if k in df.columns]
#|                 df = df.drop_duplicates(subset=keys, keep="last")
#|             except Exception:
#|                 pass
#|         df.to_csv(OUT_CSV, index=False)
#|         print(f"  [saved] {OUT_CSV}")
#|
#|     C.write_manifest("26_rank_stability_and_learning_curve.py",
#|                      inputs=[C.MODEL_A, C.FRAME, C.RANKING_A],
#|                      outputs=[OUT_JSON, OUT_CSV],
#|                      settings=dict(arms=want, ATTRIB_ROWS=ATTRIB_ROWS,
#|                                    BOOT_DRAWS=a.boot_draws or BOOT_DRAWS,
#|                                    TOPK=list(TOPK),
#|                                    RESEEDS=list(a.reseeds or RESEEDS),
#|                                    CURVE_FOLDS=a.curve_folds or CURVE_FOLDS,
#|                                    CURVE_SHARES=list(a.curve_shares
#|                                                      or CURVE_SHARES)),
#|                      notes="Rank stability under a county bootstrap and under "
#|                            "refits, and the learning curve over county "
#|                            "subsets.")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/26_rank_stability_and_learning_curve.py
#@@ FILE analysis/28_residual_and_social_determinants.py
#| #!/usr/bin/env python3
#| """
#| 28_residual_and_social_determinants.py -- where an environment-only model is
#| blind, and whether that blindness is socioeconomic.
#|
#| READS   the out-of-fold predictions of the production model
#|         the reference frame, for the socioeconomic columns the admission gate
#|         excluded from the model ("socioeconomic-not-in-scope")
#|         the modelling panel, for the environmental control
#| WRITES  ../derived*/residual_social_determinants.json
#|         ../derived*/residual_social_determinants_counties.csv
#|
#| THE QUESTION
#| ------------
#| No socioeconomic, demographic or clinical variable entered the model. Its
#| out-of-fold residual is therefore the part of county life expectancy the
#| physical environment does not account for. If that residual is structured by
#| socioeconomic disadvantage, the socioeconomic columns the model never saw
#| should predict it, and the environmental columns it did see should not.
#|
#| Both are measured here with the same learner, on county means, under folds
#| grouped by state so that a county's neighbours do not predict it. The
#| socioeconomic indicators are named in advance (POLICY_INDICATORS) rather than
#| picked after looking, and their correlations carry a correction for the
#| number tested.
#|
#| Residual sign: observed minus predicted. A negative residual is a county whose
#| life expectancy is worse than its physical environment predicts.
#|
#|     python3 28_residual_and_social_determinants.py
#| """
#| from __future__ import annotations
#|
#| import json
#| import os
#| import sys
#|
#| import numpy as np
#| import pandas as pd
#|
#| HERE = os.path.dirname(os.path.abspath(__file__))
#| if HERE not in sys.path:
#|     sys.path.insert(0, HERE)
#|
#| import _config as C                                            # noqa: E402
#|
#| OUT_JSON = C.DERIVED / "residual_social_determinants.json"
#| OUT_CSV = C.DERIVED / "residual_social_determinants_counties.csv"
#|
#| # Named before any correlation was computed: the six indicators most often used
#| # to describe county socioeconomic disadvantage in the United States.
#| POLICY_INDICATORS = {
#|     "sdoh_saipe_pct_pov": "Population below the poverty line (%)",
#|     "sdoh_saipe_median_hh_income": "Median household income (USD)",
#|     "sdoh_unemployment_rate": "Unemployment rate (%)",
#|     "sdoh_acs_pct_uninsured": "Without health insurance (%)",
#|     "sdoh_bachelor's_degree_or_higher_%": "Bachelor's degree or higher (%)",
#|     "sdoh_disability_rate": "Disability rate (%)",
#| }
#| EXCLUDED_MODALITIES = {"Health outcomes"}   # behaviours and outcomes are not
#|                                              # socioeconomic conditions
#| MIN_COUNTY_COVERAGE = 0.80
#| LARGE_MISS = 3.0                             # years, as in the main text
#|
#|
#| def grouped_oof_r2(X, y, states, seed=0):
#|     """Out-of-fold R2 of a gradient-boosting model, folds grouped by state."""
#|     from sklearn.ensemble import HistGradientBoostingRegressor
#|     from sklearn.metrics import r2_score
#|     from sklearn.model_selection import GroupKFold
#|     pred = np.full(len(y), np.nan)
#|     for tr, te in C.grouped_splits(states, 8):
#|         m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
#|                                           max_leaf_nodes=31,
#|                                           min_samples_leaf=20,
#|                                           l2_regularization=1.0,
#|                                           random_state=seed)
#|         m.fit(X.iloc[tr], y[tr])
#|         pred[te] = m.predict(X.iloc[te])
#|     return float(r2_score(y, pred))
#|
#|
#| def main():
#|     import pyarrow.parquet as pq
#|     from scipy.stats import spearmanr
#|
#|     C.banner("28  the residual and the social determinants the model never saw")
#|     oof = pd.read_csv(C.OOF_A)
#|     oof["resid"] = oof[C.TARGET] - oof["pred"]
#|     county = (oof.groupby(C.GROUP)
#|                  .agg(resid=("resid", "mean"), observed=(C.TARGET, "mean"),
#|                       predicted=("pred", "mean"))
#|                  .reset_index())
#|     county["state"] = county[C.GROUP].astype(int).map(lambda f: f"{f:05d}"[:2])
#|     print(f"  {len(county):,} counties, mean residual sd "
#|           f"{county.resid.std():.3f} years")
#|
#|     gate = pd.read_csv(C.FEATURE_GATE_A)
#|     sdoh = gate[gate["reason"].astype(str).str.contains("socio")
#|                 & ~gate["modality"].isin(EXCLUDED_MODALITIES)]["feature"].tolist()
#|     frame_cols = set(pq.read_schema(C.FRAME).names)
#|     sdoh = [c for c in sdoh if c in frame_cols]
#|     fr = pq.read_table(C.FRAME, columns=[C.GROUP, C.YEAR] + sdoh).to_pandas()
#|     cm = fr.groupby(C.GROUP)[sdoh].mean()
#|     keep = [c for c in sdoh if cm[c].notna().mean() >= MIN_COUNTY_COVERAGE
#|             and cm[c].nunique() > 5]
#|     cm = cm[keep]
#|     print(f"  {len(sdoh)} socioeconomic columns excluded by the gate, "
#|           f"{len(keep)} with county coverage of at least "
#|           f"{100 * MIN_COUNTY_COVERAGE:.0f} per cent")
#|
#|     d = county.merge(cm, left_on=C.GROUP, right_index=True, how="inner")
#|     Xs = d[keep].fillna(d[keep].median())
#|     y = d["resid"].to_numpy()
#|     states = d["state"].to_numpy()
#|
#|     # environmental control: county means of the model's own columns
#|     p = C.load_panel()
#|     env = (pd.concat([p["df"][[C.GROUP]], p["X"]], axis=1)
#|              .groupby(C.GROUP).mean())
#|     de = d[[C.GROUP]].merge(env, left_on=C.GROUP, right_index=True,
#|                             how="left").drop(columns=[C.GROUP])
#|     de = de.fillna(de.median())
#|
#|     r2_sdoh = grouped_oof_r2(Xs, y, states)
#|     r2_env = grouped_oof_r2(de, y, states)
#|     print(f"\n  out-of-fold R2 for the county residual, folds grouped by state")
#|     print(f"    from the {len(keep)} socioeconomic columns the model never saw  "
#|           f"{r2_sdoh:+.3f}")
#|     print(f"    from the {de.shape[1]} environmental columns it did see        "
#|           f"{r2_env:+.3f}")
#|
#|     rows, pvals = [], []
#|     for col, label in POLICY_INDICATORS.items():
#|         if col not in d.columns:
#|             continue
#|         ok = d[col].notna()
#|         rho, pv = spearmanr(d.loc[ok, col], d.loc[ok, "resid"])
#|         rows.append(dict(column=col, label=label, rho=float(rho), p=float(pv),
#|                          n=int(ok.sum())))
#|         pvals.append(pv)
#|     # Holm correction over the pre-declared six
#|     order = np.argsort(pvals)
#|     m = len(pvals)
#|     adj = np.empty(m)
#|     running = 0.0
#|     for rank, i in enumerate(order):
#|         running = max(running, (m - rank) * pvals[i])
#|         adj[i] = min(running, 1.0)
#|     for r, a in zip(rows, adj):
#|         r["p_holm"] = float(a)
#|     print("\n  pre-declared indicators, Spearman with the county residual")
#|     for r in rows:
#|         print(f"    {r['label']:<42} rho {r['rho']:+.3f}  Holm p {r['p_holm']:.2g}")
#|
#|     miss = d["resid"] <= -LARGE_MISS
#|     prof = {}
#|     for col, label in POLICY_INDICATORS.items():
#|         if col in d.columns:
#|             pct = d[col].rank(pct=True)
#|             prof[col] = dict(label=label,
#|                              median_percentile_large_miss=float(
#|                                  100 * pct[miss].median()),
#|                              median_percentile_rest=float(
#|                                  100 * pct[~miss].median()))
#|     print(f"\n  {int(miss.sum())} counties where the model overpredicts by "
#|           f"{LARGE_MISS:g} years or more on average")
#|     for col, v in prof.items():
#|         print(f"    {v['label']:<42} median percentile "
#|               f"{v['median_percentile_large_miss']:.0f} against "
#|               f"{v['median_percentile_rest']:.0f}")
#|
#|     d[[C.GROUP, "state", "observed", "predicted", "resid"]].to_csv(
#|         OUT_CSV, index=False)
#|     blob = dict(n_counties=int(len(d)), n_sdoh_columns=len(keep),
#|                 n_env_columns=int(de.shape[1]),
#|                 oof_r2_residual_from_sdoh=r2_sdoh,
#|                 oof_r2_residual_from_environment=r2_env,
#|                 residual_sd=float(d["resid"].std()),
#|                 indicators=rows, large_miss_threshold=LARGE_MISS,
#|                 n_large_miss=int(miss.sum()), large_miss_profile=prof,
#|                 note="County means. Folds grouped by state. Residual is "
#|                      "observed minus predicted out of fold.")
#|     json.dump(blob, open(OUT_JSON, "w"), indent=1)
#|     print(f"\n  [saved] {OUT_JSON}")
#|     C.write_manifest("28_residual_and_social_determinants.py",
#|                      inputs=[C.OOF_A, C.FRAME, C.FEATURE_GATE_A],
#|                      outputs=[OUT_JSON, OUT_CSV],
#|                      settings=dict(MIN_COUNTY_COVERAGE=MIN_COUNTY_COVERAGE,
#|                                    LARGE_MISS=LARGE_MISS))
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/28_residual_and_social_determinants.py
#@@ FILE analysis/30_residual_audit_sensitivity.py
#| #!/usr/bin/env python3
#| """
#| 30_residual_audit_sensitivity.py -- does the residual audit of script 28 rest
#| on the health-related columns of the County Health Rankings?
#|
#| Script 28 predicts the county residual of the environment-only model from the
#| socioeconomic columns the admission gate excluded. Sixteen of those columns
#| come from the County Health Rankings and several describe health behaviours or
#| health status (smoking, obesity, diabetes, physical inactivity, poor health).
#| They are not socioeconomic conditions in the strict sense, so the audit is
#| repeated here without them, and once more with the American Community Survey
#| columns alone. Same county means, same learner, same folds grouped by state.
#|
#| READS   the same inputs as script 28
#| WRITES  ../derived*/residual_audit_sensitivity.json
#|
#|     python3 30_residual_audit_sensitivity.py
#| """
#| from __future__ import annotations
#|
#| import json
#| import os
#| import sys
#|
#| import pandas as pd
#|
#| HERE = os.path.dirname(os.path.abspath(__file__))
#| if HERE not in sys.path:
#|     sys.path.insert(0, HERE)
#|
#| import _config as C                                            # noqa: E402
#|
#| S28 = __import__("28_residual_and_social_determinants")
#| OUT_JSON = C.DERIVED / "residual_audit_sensitivity.json"
#|
#|
#| def source_of(col):
#|     if col.startswith("sdoh_acs"):
#|         return "American Community Survey"
#|     if col.startswith("sdoh_chr"):
#|         return "County Health Rankings"
#|     if col.startswith("sdoh_saipe"):
#|         return "Small Area Income and Poverty Estimates"
#|     if col.startswith("econ"):
#|         return "Regional economic accounts"
#|     return "Summary indicators of the database"
#|
#|
#| def main():
#|     import pyarrow.parquet as pq
#|     C.banner("30  residual audit without the health-related columns")
#|     oof = pd.read_csv(C.OOF_A)
#|     oof["resid"] = oof[C.TARGET] - oof["pred"]
#|     county = oof.groupby(C.GROUP).agg(resid=("resid", "mean")).reset_index()
#|     county["state"] = county[C.GROUP].astype(int).map(lambda f: f"{f:05d}"[:2])
#|
#|     gate = pd.read_csv(C.FEATURE_GATE_A)
#|     sdoh = gate[gate["reason"].astype(str).str.contains("socio")
#|                 & ~gate["modality"].isin(S28.EXCLUDED_MODALITIES)]["feature"].tolist()
#|     names = set(pq.read_schema(C.FRAME).names)
#|     sdoh = [c for c in sdoh if c in names]
#|     fr = pq.read_table(C.FRAME, columns=[C.GROUP, C.YEAR] + sdoh).to_pandas()
#|     cm = fr.groupby(C.GROUP)[sdoh].mean()
#|     keep = [c for c in sdoh if cm[c].notna().mean() >= S28.MIN_COUNTY_COVERAGE
#|             and cm[c].nunique() > 5]
#|     years = fr.loc[fr[keep].notna().any(axis=1), C.YEAR]
#|     d = county.merge(cm[keep], left_on=C.GROUP, right_index=True, how="inner")
#|     y = d["resid"].to_numpy()
#|     st = d["state"].to_numpy()
#|
#|     by_source = pd.Series([source_of(c) for c in keep]).value_counts().to_dict()
#|     sets = {
#|         "all_columns": keep,
#|         "without_county_health_rankings": [c for c in keep
#|                                            if not c.startswith("sdoh_chr")],
#|         "american_community_survey_only": [c for c in keep
#|                                            if c.startswith("sdoh_acs")],
#|     }
#|     out = dict(n_counties=int(len(d)), columns_by_source=by_source,
#|                years=[int(years.min()), int(years.max())], arms={})
#|     for name, cols in sets.items():
#|         X = d[cols].fillna(d[cols].median())
#|         r2 = S28.grouped_oof_r2(X, y, st)
#|         out["arms"][name] = dict(n_columns=len(cols), oof_r2=r2)
#|         print(f"  {name:<34} {len(cols):>4} columns   R2 {r2:+.3f}")
#|     print(f"  columns by source: {by_source}")
#|     print(f"  years present: {out['years']}")
#|     OUT_JSON.write_text(json.dumps(out, indent=2))
#|     print(f"  wrote {OUT_JSON}")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/30_residual_audit_sensitivity.py
#@@ FILE analysis/33_export_figure10c_data.py
#| #!/usr/bin/env python3
#| """
#| 33_export_figure10c_data.py -- the data behind Figure 10C in one small file.
#|
#| READS   the out-of-fold predictions of the production model
#|         the reference frame, for the six socioeconomic indicators
#|         ../derived/residual_social_determinants.json (script 28)
#| WRITES  ../derived/figure10c_residual_and_indicators.csv
#|         ../derived/figure10c_check.json
#|
#| WHY THIS EXISTS
#| ---------------
#| Figure 10C correlates each county's mean out-of-fold residual (observed minus
#| predicted life expectancy) with six socioeconomic indicators named in advance.
#| None of these indicators entered the model. This script writes exactly that
#| table, one row per county, so a reader can recompute every bar of the panel
#| without the full data set, and checks that the recomputed Spearman
#| correlations equal the ones script 28 reported.
#|
#| The indicators are county means over the years each is observed in the
#| reference frame (2009 to 2019 for most), the same aggregation script 28 uses.
#| Nothing here is new analysis.
#|
#|     python3 33_export_figure10c_data.py
#| """
#| from __future__ import annotations
#|
#| import json
#| import os
#| import sys
#|
#| import numpy as np
#| import pandas as pd
#|
#| HERE = os.path.dirname(os.path.abspath(__file__))
#| if HERE not in sys.path:
#|     sys.path.insert(0, HERE)
#|
#| import _config as C                                            # noqa: E402
#|
#| OUT_CSV = C.DERIVED / "figure10c_residual_and_indicators.csv"
#| OUT_JSON = C.DERIVED / "figure10c_check.json"
#| SHORT = {
#|     "sdoh_saipe_pct_pov": "poverty_pct",
#|     "sdoh_saipe_median_hh_income": "median_household_income_usd",
#|     "sdoh_unemployment_rate": "unemployment_pct",
#|     "sdoh_acs_pct_uninsured": "uninsured_pct",
#|     "sdoh_bachelor's_degree_or_higher_%": "bachelors_or_higher_pct",
#|     "sdoh_disability_rate": "disability_pct",
#| }
#|
#|
#| def main():
#|     import pyarrow.parquet as pq
#|     from scipy.stats import spearmanr
#|
#|     C.banner("33  data behind Figure 10C")
#|     oof = pd.read_csv(C.OOF_A)
#|     oof["resid"] = oof[C.TARGET] - oof["pred"]
#|     county = (oof.groupby(C.GROUP)
#|                  .agg(resid=("resid", "mean"), observed=(C.TARGET, "mean"),
#|                       predicted=("pred", "mean"))
#|                  .reset_index())
#|     county["state"] = county[C.GROUP].astype(int).map(lambda f: f"{f:05d}"[:2])
#|
#|     fr = pq.read_table(C.FRAME, columns=[C.GROUP, C.YEAR] + list(SHORT)).to_pandas()
#|     cm = fr.groupby(C.GROUP)[list(SHORT)].mean()
#|     years = {SHORT[c]: [int(fr.loc[fr[c].notna(), C.YEAR].min()),
#|                         int(fr.loc[fr[c].notna(), C.YEAR].max())] for c in SHORT}
#|     d = county.merge(cm, left_on=C.GROUP, right_index=True, how="inner")
#|     d = d.rename(columns={C.GROUP: "fips", **SHORT})
#|     d["fips"] = d["fips"].astype(int).map(lambda f: f"{f:05d}")
#|     cols = ["fips", "state", "observed", "predicted", "resid"] + list(SHORT.values())
#|     d[cols].to_csv(OUT_CSV, index=False, float_format="%.6g")
#|
#|     ref = json.load(open(C.DERIVED / "residual_social_determinants.json"))
#|     ref = {r["column"]: r for r in ref["indicators"]}
#|     check = []
#|     for col, short in SHORT.items():
#|         ok = d[short].notna()
#|         rho = float(spearmanr(d.loc[ok, short], d.loc[ok, "resid"])[0])
#|         same = abs(rho - ref[col]["rho"]) < 1e-6
#|         check.append(dict(indicator=short, label=ref[col]["label"],
#|                           rho=rho, rho_script28=ref[col]["rho"], n=int(ok.sum()),
#|                           years=years[short], matches=same))
#|         print(f"  {ref[col]['label']:<42} rho {rho:+.3f}  "
#|               f"{'matches script 28' if same else 'DIFFERS from script 28'}")
#|     json.dump(dict(n_counties=int(len(d)), indicators=check,
#|                    note="County means; residual is observed minus predicted "
#|                         "out of fold. Reproduces Figure 10C."),
#|               open(OUT_JSON, "w"), indent=1)
#|     print(f"\n  [saved] {OUT_CSV.name}  ({len(d):,} counties)")
#|     C.write_manifest("33_export_figure10c_data.py",
#|                      inputs=[C.OOF_A, C.FRAME],
#|                      outputs=[OUT_CSV, OUT_JSON], settings={})
#|     if not all(x["matches"] for x in check):
#|         sys.exit(1)
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/33_export_figure10c_data.py
#@@ FILE analysis/34_library_version_sensitivity.py
#| #!/usr/bin/env python3
#| """
#| 34_library_version_sensitivity.py -- how much do the numbers move between
#| library versions?
#|
#| READS   ../derived/{spatial_holdout,residual_spatial_structure,
#|                     permutation_across_splits}.json        (final run)
#|         ../derived/reference_run_2026-10-05/...            (earlier run)
#|         the manifests of both runs, for the library versions
#| WRITES  ../derived/library_version_sensitivity.json
#|
#| WHY THIS EXISTS
#| ---------------
#| Every script is seeded and gives identical answers when rerun on one machine
#| (scripts 01 and 10 prove it). Across library releases that is not guaranteed:
#| a new xgboost release may build a slightly different tree from the same data
#| and seed. The battery was run twice with identical settings, once with
#| xgboost 3.2.0 and once with xgboost 3.0.2. This script records which numbers
#| moved and by how much, so the paper can say so in one sentence. It computes
#| nothing new; it compares two sets of result files.
#|
#|     python3 34_library_version_sensitivity.py
#| """
#| from __future__ import annotations
#|
#| import json
#| import os
#| import sys
#|
#| HERE = os.path.dirname(os.path.abspath(__file__))
#| if HERE not in sys.path:
#|     sys.path.insert(0, HERE)
#|
#| import _config as C                                            # noqa: E402
#|
#| REF = C.DERIVED / "reference_run_2026-10-05"
#| OUT = C.DERIVED / "library_version_sensitivity.json"
#|
#|
#| def load(folder, name):
#|     return json.load(open(folder / name))
#|
#|
#| def versions(folder):
#|     """The library versions that fitted the state hold-out folds of a run:
#|     the record in its checkpoint store, or else the manifest of step 13."""
#|     v = C.fitted_with(folder / "checkpoints_cv" / "state_holdout")
#|     if v:
#|         return v
#|     for name in ("13_spatial_holdout_and_attribution_structure",
#|                  "27_permutation_importance_across_splits"):
#|         m = folder / "manifests" / f"{name}.manifest.json"
#|         if m.exists():
#|             return json.load(open(m)).get("versions", {})
#|     return {}
#|
#|
#| def main():
#|     C.banner("34  library-version sensitivity")
#|     if not REF.exists():
#|         print("  no reference run present; nothing to compare")
#|         return
#|     new_v, old_v = versions(C.DERIVED), versions(REF)
#|     hn, ho = load(C.DERIVED, "spatial_holdout.json"), load(REF, "spatial_holdout.json")
#|     mn, mo = (load(C.DERIVED, "residual_spatial_structure.json"),
#|               load(REF, "residual_spatial_structure.json"))
#|     pn, po = (load(C.DERIVED, "permutation_across_splits.json"),
#|               load(REF, "permutation_across_splits.json"))
#|     old = {s["seed"]: s for s in po["splits"]}
#|     shifts = {who: [abs(s[f"{who}_wins"] - old[s["seed"]][f"{who}_wins"])
#|                     for s in pn["splits"] if s["seed"] in old]
#|               for who in ("production", "second")}
#|     same_split = sum(s["production_wins"] == old[s["seed"]]["production_wins"]
#|                      for s in pn["splits"] if s["seed"] in old)
#|     out = dict(
#|         versions_final=new_v, versions_reference=old_v,
#|         spatial_holdout_r2=dict(final=hn["r2"], reference=ho["r2"]),
#|         residual_morans_i=dict(final=mn["morans_i"], reference=mo["morans_i"],
#|                                p_final=mn["p_value"], p_reference=mo["p_value"]),
#|         permutation_count_shift=dict(
#|             production_max=max(shifts["production"]),
#|             second_max=max(shifts["second"]),
#|             production_splits_identical=int(same_split),
#|             n_splits=len(pn["splits"])),
#|         note="Same seeds and settings in both runs; only library versions differ.")
#|     json.dump(out, open(OUT, "w"), indent=1)
#|     print(f"  xgboost {old_v.get('xgboost')} -> {new_v.get('xgboost')}")
#|     print(f"  state hold-out R2      {ho['r2']:.3f} -> {hn['r2']:.3f}")
#|     print(f"  residual Moran's I     {mo['morans_i']:.3f} -> {mn['morans_i']:.3f}")
#|     print(f"  permutation count shift, production up to "
#|           f"{max(shifts['production'])} fields, second learner up to "
#|           f"{max(shifts['second'])}")
#|     print(f"  [saved] {OUT}")
#|     C.write_manifest("34_library_version_sensitivity.py",
#|                      inputs=[C.DERIVED / "spatial_holdout.json"], outputs=[OUT],
#|                      settings={})
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/34_library_version_sensitivity.py
#@@ FILE analysis/35_check_method_statements.py
#| """
#| 35_check_method_statements.py -- check the statements about methods and data
#| that the manuscript makes in words, against the code and the result files.
#|
#| The constants builder checks every sentence that describes a NUMBER in words
#| ("weakly clustered", "barely move"). This step checks the other kind: sentences
#| that say HOW something was done ("weighted by inverse distance", "dropped the
#| later one", "a crossing was found in every resample") or WHAT the data show
#| without printing a number ("the Virginia units are independent cities"). Each
#| check reads the code that did the work, or a result file, and never refits.
#|
#|     python3 35_check_method_statements.py
#|
#| Writes derived/method_statement_checks.json. A failed check prints FAIL and the
#| section to reread; the step exits with status 1 if any check fails.
#|
#| Reads: the pipeline code (../pipeline), the step scripts in this folder, the
#| result files in derived/, and studyA_reference_frame.parquet (for the terrain
#| check only). Runs in about ten seconds.
#| """
#| import json
#| import re
#| import sys
#| from pathlib import Path
#|
#| import numpy as np
#| import pandas as pd
#|
#| import _config as C
#|
#| HERE = Path(__file__).resolve().parent
#| PIPE = C.PAPER / "pipeline"
#| OUT = C.DERIVED / "method_statement_checks.json"
#|
#| RESULTS = []
#|
#|
#| def check(section, statement, ok, evidence):
#|     RESULTS.append(dict(section=section, statement=statement,
#|                         status="PASS" if ok else "FAIL", evidence=evidence))
#|     print(f"  {'PASS' if ok else 'FAIL':<5} {section:<6} {statement}")
#|     print(f"        {evidence}")
#|
#|
#| def src(name):
#|     p = HERE / name
#|     if not p.exists():
#|         p = PIPE / name
#|     return p.read_text()
#|
#|
#| def js(name):
#|     return json.load(open(C.DERIVED / name))
#|
#|
#| def step(prefix):
#|     return src(next(p.name for p in HERE.glob(f"{prefix}_*.py")))
#|
#|
#| def main():
#|     C.banner("35  Statements about methods and data, checked against the code")
#|
#|     # ------------------------------------------------------------ 2.5 filter
#|     core = src("pipeline_core_v2.py")
#|     pats = re.search(r"PROTECT_PATTERNS = \[(.*?)\]", core, re.S).group(1)
#|     want = {"heat indices": ["heat_?index", "humidex", "wbgt"],
#|             "wet-bulb": ["wet_?bulb"], "wind chill": ["wind_?chill"],
#|             "dew point and humidity": ["tdmean", "dew_?point", "specific_humidity"],
#|             "total column water vapour": ["total_column_water"],
#|             "minimum temperature": ["prism_tmin", "tmin_min"],
#|             "night-time LST": ["lst_night"], "diurnal range": ["diurnal"],
#|             "night cooling": ["night_cool"], "vapour-pressure deficit": ["vpd"],
#|             "hot nights and days": ["hot_?night", "days_tmin_over", "days_tmax_over"],
#|             "formaldehyde, NO2, ozone, PM2.5": ["formaldehyde", "nitrogen_dioxide",
#|                                                 "ozone", "pm25_"]}
#|     miss = [k for k, v in want.items() if not all(x in pats for x in v)]
#|     check("2.5", "The protected list covers the patterns the text names",
#|           not miss, f"PROTECT_PATTERNS in pipeline_core_v2.py; missing: {miss or 'none'}")
#|     check("2.5", "Of two unprotected columns the filter drops the later one, "
#|           "and of a mixed pair the unprotected one",
#|           "for b in cols[i + 1:]" in core
#|           and "victim = b if b not in protected else a" in core,
#|           "redundancy_filter: loops a before b, victim = b unless b is protected")
#|     check("2.5", "The correlation cut is 0.98 and the filter never sees the outcome",
#|           "corr_threshold=0.98" in core and "never sees the target" in core,
#|           "redundancy_filter(X, corr_threshold=0.98, ...): X only")
#|     check("2.5", "The R2 interval resamples county-years with the predictions held fixed",
#|           "def bootstrap_r2_ci(y, pred" in core
#|           and "r2_score(y[b], pred[b])" in core,
#|           "bootstrap_r2_ci: rows drawn with replacement, no refit")
#|
#|     # ---------------------------------------------------- 2.6 baselines, brackets
#|     s02 = step("02")
#|     check("2.6", "Five brackets 60-70, 70-74, 74-76.5, 76.5-79, 79-95, each including its lower bound",
#|           "BRACKET_EDGES = [60, 70, 74, 76.5, 79, 95]" in s02 and "right=False" in s02,
#|           "step 02: BRACKET_EDGES and pd.cut(..., right=False)")
#|     s03 = step("03")
#|     check("2.6", "Nearest-county baseline: same year, four nearest training counties, "
#|           "weighted by inverse distance",
#|           "KNN_NEIGHBOURS = 4" in s03 and "KNN_SAME_YEAR = True" in s03
#|           and "w = 1.0 / np.maximum(dist, 1e-6)" in s03
#|           and "itr = tr[yr[tr] == v]" in s03,
#|           "step 03 b_knn: training rows of the same year, haversine distance, 1/d weights")
#|
#|     # ---------------------------------------------------------- 2.7 attribution
#|     s23 = step("23")
#|     inf = js("influence_drop.json")["topk_overlap"]
#|     check("2.7, 3.2", "The influence refit drops the counties with the largest mean absolute "
#|           "out-of-fold residual and is attributed by path-dependent TreeSHAP; "
#|           "the tail form of NCE ranks first in it",
#|           '(o[tc] - o[pc]).abs()' in s23 and "path dependent" in inf["attribution"]
#|           and inf["refit_top"][0] == C.NCE_TAIL,
#|           f"step 23; influence_drop.json: arm {inf['arm']}, top {inf['refit_top'][0]}, "
#|           f"{inf['shared']} of top {inf['topk']} shared")
#|
#|     # ------------------------------------------------------ 2.9 lifetime tests
#|     s25 = step("25")
#|     check("2.9", "The class comparison is a one-sided Mann-Whitney test by species",
#|           'alternative="greater"' in s25 and "stats.mannwhitneyu" in s25,
#|           "step 25: mannwhitneyu(short, long, alternative='greater')")
#|     check("2.9", "Each lifetime is multiplied by ten raised to an independent uniform draw",
#|           "10 ** rng.uniform(-VALUE_SPREAD_DEX, VALUE_SPREAD_DEX, len(sp))" in s25,
#|           "step 25: one draw per species, VALUE_SPREAD_DEX each side")
#|     lt = js("lifetime_literature_test.json")
#|     per = pd.DataFrame(lt["species_table"])
#|     long3 = per[per["lifetime_days"] >= 21]
#|     ok = len(long3) == 3 and bool((long3["median_ratio"] < 1).all())
#|     ev = ", ".join(f"{r['species']} {r['lifetime_days']:g} d, median ratio "
#|                    f"{r['median_ratio']:.2f}" for _, r in long3.iterrows())
#|     check("3.3", "The three species that persist three weeks or longer all favoured the mean",
#|           ok, ev)
#|
#|     # --------------------------------------------------- 2.10 turning points
#|     s06 = step("06")
#|     check("2.10", "The bootstrap resamples the attributed county-years and does not refit "
#|           "or redraw the background",
#|           "idx = rng.integers(0, n, n)" in s06
#|           and "crossing(x[idx], y[idx]" in s06 and ".fit(" not in s06,
#|           "step 06 analyse(): rows of the one attributed sample drawn with replacement")
#|     check("2.10", "The range is that between the 1st and 99th percentiles; 45 bins; "
#|           "a smoothing spline through the bin medians",
#|           "TRIM_PCT = (1, 99)" in s06 and "UnivariateSpline" in s06
#|           and 'BIN_STATISTIC = "median"' in s06,
#|           "step 06: TRIM_PCT, UnivariateSpline, BIN_STATISTIC")
#|     tp = js("turning_points.json")
#|     res = tp["results"] if isinstance(tp["results"], list) else list(tp["results"].values())
#|     found = [r["found_frac"] for r in res]
#|     width = [r["width_frac"] for r in res]
#|     check("3.4", "Neither numerical limit decided any candidate: a crossing in every "
#|           "resample, every interval narrower than 40% of the range",
#|           min(found) == 1.0 and max(width) < 0.4,
#|           f"turning_points.json: found_frac min {min(found):.3f}; "
#|           f"width_frac max {max(width):.3f} ({len(res)} candidates)")
#|     stable = {r["feature"] for r in res if r.get("stable", r.get("passes"))}
#|     four = {"eng2_night_cooling_efficiency", "eng2_wetbulb_hot", "prism_tmin_min",
#|             "usda_cropland_usda_corn_pct"}
#|     check("3.4", "Of the other six candidates four pass outright, night-time LST fails "
#|           "and PM2.5 passes only formally",
#|           four <= stable and "lst_night_1km_mean" not in stable and "pm25_mean" in stable
#|           and len(stable) == tp["n_stable"],
#|           f"stable: {len(stable)} of {len(res)}")
#|     s14 = step("14")
#|     check("2.10", "The regional shift is the absolute west-minus-east difference, as a share "
#|           "of the 1st-to-99th-percentile range",
#|           'row["delta"] = abs(w - e)' in s14 and "np.nanpercentile(dep[feat], 99)" in s14,
#|           "step 14: delta = |w - e|, span = p99 - p1")
#|     nc = js("null_crossing.json")["per_variable"]
#|     nc = list(nc.values()) if isinstance(nc, dict) else nc
#|     none = sum(1 for r in nc if r["permuted_crossing_found"] == 0)
#|     check("3.5", "For most candidates no permuted curve crossed zero", none > len(nc) / 2,
#|           f"null_crossing.json: no permuted crossing for {none} of {len(nc)}")
#|
#|     dep = pd.read_csv(C.DERIVED / "shap_dependence_candidates.csv")
#|
#|     def span(f):
#|         x = dep[f].to_numpy(float); y = dep["shap_" + f].to_numpy(float)
#|         lo, hi = np.nanpercentile(x, (1, 99))
#|         m = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
#|         x, y = x[m], y[m]
#|         q = np.quantile(x, np.linspace(0, 1, 46)); q[-1] += 1e-9
#|         b = [np.median(y[(x >= q[i]) & (x < q[i + 1])]) for i in range(45)]
#|         return float(max(b) - min(b))
#|     big = {f: span(f) for f in ("prism_tdmean_min", C.NCE_TAIL,
#|                                "cams_fot_formaldehyde_above75th_percentile")}
#|     small = {f: span(f) for f in ("eng2_wetbulb_night", "eng2_hcho_no2_ratio")}
#|     check("3.4", "The night-time wet-bulb and ratio curves (Figure 6B, E) span less "
#|           "attribution than panels A, C and D",
#|           max(small.values()) < min(big.values()),
#|           "binned-median span (years): " + ", ".join(f"{k} {v:.4f}" for k, v in
#|                                                      {**big, **small}.items()))
#|
#|     # ---------------------------------------------- 2.3 engineered variables
#|     eng = src("engineered_features_v2.py")
#|     check("2.3", "Night-time wet-bulb: annual-mean daily minimum with the driest-cell "
#|           "dew point; warm-hour form: annual-mean daily maximum with the mean dew point",
#|           'needs=("prism_tmin_mean", "prism_tdmean_min")' in eng
#|           and 'needs=("prism_tmax_mean", "prism_tdmean_mean")' in eng,
#|           "engineered_features_v2.py: wetbulb_night, wetbulb_hot")
#|     check("2.4", "A 12 degC drop gives NCE 0.40 at 30 degC and 0.60 at 20 degC",
#|           round(12 / 30, 2) == 0.40 and round(12 / 20, 2) == 0.60,
#|           "12/30 = 0.40, 12/20 = 0.60")
#|
#|     rank = pd.read_csv(C.RANKING_A)["feature"].astype(str)
#|     era = [c for c in rank if c.startswith("era5_")]
#|     terra = [c for c in rank if c.startswith("terra_")]
#|     prism = [c for c in rank if c.startswith("prism_")]
#|     check("2.3", "Admitted ERA5 columns give humidity, pressure, wind, snow and terrain "
#|           "height; PRISM gives vapour-pressure deficit; TerraClimate has no VPD column",
#|           all(any(k in c for c in era) for k in ("humidity", "pressure", "wind", "snow",
#|                                                   "geopotential"))
#|           and any("vpd" in c for c in prism) and not any("vpd" in c for c in terra),
#|           f"{len(era)} ERA5, {len(terra)} TerraClimate, {len(prism)} PRISM columns admitted")
#|
#|     frame = pd.read_parquet(C.FRAME, columns=[C.GROUP, "prism_tdmean_min",
#|                                                "prism_tdmean_mean", "dem_p10", "dem_p90"])
#|     panel = set(pd.read_csv(C.DERIVED / "residual_social_determinants_counties.csv")
#|                 ["fips"].astype(int))
#|     g = frame.groupby(C.GROUP).mean()
#|     g = g[g.index.astype(int).isin(panel)]
#|     from scipy.stats import spearmanr
#|     r_hi = spearmanr(g["prism_tdmean_min"], g["dem_p90"], nan_policy="omit").correlation
#|     r_gap = spearmanr(g["prism_tdmean_mean"] - g["prism_tdmean_min"],
#|                       g["dem_p90"] - g["dem_p10"], nan_policy="omit").correlation
#|     check("2.3, 4.3", "The driest cell is often on a county's highest ground",
#|           r_hi < -0.5 and r_gap > 0.5,
#|           f"{len(g)} counties: Spearman(min dew point, 90th-pct elevation) {r_hi:.2f}; "
#|           f"Spearman(mean minus min dew point, relief p90-p10) {r_gap:.2f}")
#|
#|     # ------------------------------------------------------- 2.11 / 3.6 error
#|     s13 = step("13")
#|     check("2.11", "Moran's I is the slope of the neighbours' mean residual on the county's "
#|           "own (row-standardised, four nearest)",
#|           "lag = vals[nb].mean(axis=1)" in s13 and "np.sum(vals * lag) / denom" in s13,
#|           "step 13 residual_structure: I = sum(z * lag) / sum(z^2)")
#|     cty = pd.read_csv(C.DERIVED / "residual_social_determinants_counties.csv")
#|     rs = js("residual_social_determinants.json")
#|     miss = cty[cty["resid"] <= -rs["large_miss_threshold"]].copy()
#|     f5 = miss["fips"].astype(int)
#|     va = f5[(f5 // 1000 == 51)]
#|     va_city = va[(va % 1000) >= 500]
#|     others = {24510: "Baltimore", 29510: "St. Louis", 32510: "Carson City"}
#|     check("3.6", "Among the overpredicted counties, the Virginia units, Baltimore, "
#|           "St. Louis and Carson City are independent cities",
#|           len(miss) == rs["n_large_miss"] and len(va) == len(va_city) > 0
#|           and set(others) <= set(f5),
#|           f"{len(miss)} counties overpredicted by {rs['large_miss_threshold']:g} years or "
#|           f"more; Virginia units {sorted(va.tolist())} (all FIPS >= 51500); "
#|           + ", ".join(f"{v} {k}" for k, v in others.items()))
#|     sign = {r["column"]: r["rho"] for r in rs["indicators"]}
#|     neg = [k for k in sign if any(w in k for w in ("pov", "unemploy", "uninsured", "disab"))]
#|     pos = [k for k in sign if any(w in k for w in ("income", "bachelor"))]
#|     check("3.6", "Overprediction where poverty, unemployment, uninsurance and disability "
#|           "are high; underprediction where income and education are high",
#|           len(neg) == 4 and len(pos) == 2 and all(sign[k] < 0 for k in neg)
#|           and all(sign[k] > 0 for k in pos),
#|           "Spearman rho: " + ", ".join(f"{k.replace('sdoh_', '')} {v:+.2f}"
#|                                        for k, v in sign.items()))
#|
#|     n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
#|     json.dump(dict(checks=RESULTS, n_checks=len(RESULTS), n_fail=n_fail),
#|               open(OUT, "w"), indent=1)
#|     print(f"\n  {len(RESULTS) - n_fail} of {len(RESULTS)} statements match the code and "
#|           f"the result files")
#|     print(f"  [saved] {OUT}")
#|     C.write_manifest(__file__, [C.RANKING_A, C.FRAME], [OUT])
#|     sys.exit(1 if n_fail else 0)
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/35_check_method_statements.py
#@@ FILE analysis/build_paper_constants.py
#| #!/usr/bin/env python3
#| """
#| build_paper_constants.py -- every number the final manuscript quotes from the
#| analysis battery, written as LaTeX macros.
#|
#| This is the final-paper copy (version 28). It writes only the numbers and
#| tables the final manuscript prints; the blocks that read the census, the
#| attribution-instrument comparison, the permutation splits, the admission grid
#| and the multiplicity family are not included, and the background size of the
#| turning-point attribution is read from script 05's own record.
#|
#| READS   the result files in the derived folder (see _config.DERIVED)
#|         main_run_constants/constants.tex, constants_v10.tex and
#|         constants_thresholds.tex, the main run's generated constants
#|         ../main.tex, to know which main-run constants the paper quotes
#| WRITES  ../constants_results.tex   every number from the analysis battery
#|         ../constants_base.tex      the main-run constants the paper quotes,
#|                                    resolved and flattened
#|         ../tables_generated_slim.tex  appendix Tables 3 and 4 (species,
#|                                    turning points)
#|         ../tables_learner.tex      appendix Table A1 (the learner)
#|         ../tables_data.tex         Table 1 (data sources and the outcome)
#|
#| main.tex loads constants_base.tex and then constants_results.tex. Every macro
#| in the results file is defined with \\providecommand followed by
#| \\renewcommand and that file is loaded last, so where a name exists in both,
#| the value read from the current result files is the one printed. No number in
#| this script is typed in: each is read from a result file, and the file it came
#| from is written as a comment above it.
#|
#|     python3 build_paper_constants.py
#|     python3 build_paper_constants.py --out ../constants_results.tex
#| """
#| from __future__ import annotations
#|
#| import argparse
#| import json
#| import os
#| import re
#| import sys
#| from math import comb
#| from pathlib import Path
#|
#| import numpy as np
#| import pandas as pd
#|
#| HERE = os.path.dirname(os.path.abspath(__file__))
#| if HERE not in sys.path:
#|     sys.path.insert(0, HERE)
#|
#| import _config as C                                            # noqa: E402
#|
#| LINES: list[str] = []
#| SOURCE = ""
#|
#|
#| def src(name):
#|     global SOURCE
#|     SOURCE = name
#|     LINES.append(f"\n% ---- {name}")
#|
#|
#| def js(name):
#|     src(name)
#|     return json.load(open(C.DERIVED / name))
#|
#|
#| def tab(name):
#|     src(name)
#|     return pd.read_csv(C.DERIVED / name)
#|
#|
#| def put(macro, value):
#|     LINES.append(f"\\providecommand{{\\{macro}}}{{}}"
#|                  f"\\renewcommand{{\\{macro}}}{{{value}}}")
#|
#|
#| def f(v, nd=3):
#|     """A plain decimal with a true minus sign."""
#|     s = f"{float(v):.{nd}f}"
#|     if s.startswith("-"):
#|         s = "$-$" + s[1:]
#|     return s
#|
#|
#| def n(v):
#|     """An integer with a thin thousands separator."""
#|     return f"{int(round(float(v))):,}".replace(",", "{,}")
#|
#|
#| def pct(v, nd=0):
#|     return f"{100 * float(v):.{nd}f}"
#|
#|
#| def p(v):
#|     """A p value: three significant figures above 0.001, scientific below."""
#|     v = float(v)
#|     if v >= 0.001:
#|         return f"{v:.3f}" if v < 0.1 else f"{v:.2f}"
#|     e = int(np.floor(np.log10(v)))
#|     m = v / 10 ** e
#|     return f"${m:.1f}\\times10^{{{e}}}$"
#|
#|
#| def sign_p(k, nn):
#|     """Exact two-sided sign test of k successes in nn against a fair coin.
#|     Every sign-test p value the manuscript prints comes from here, so all are
#|     two-sided whatever convention the result file that holds k used."""
#|     k, nn = int(k), int(nn)
#|     up = sum(comb(nn, i) for i in range(k, nn + 1)) / 2 ** nn
#|     lo = sum(comb(nn, i) for i in range(0, k + 1)) / 2 ** nn
#|     return min(1.0, 2 * min(up, lo))
#|
#|
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--out", default=str(Path(HERE).parent / "constants_results.tex"))
#|     a = ap.parse_args()
#|
#|     # ------------------------------------------------------------ baselines
#|     b = js("baselines.json")
#|     put("BlKnn", f(b["knn4_spatial"]["r2"]))
#|
#|     br = tab("bracket_error.csv")
#|     lo = br.iloc[0]
#|     put("BrLowMAE", f(lo["mae"], 2))
#|     mid = br.loc[br["mae"].idxmin()]
#|     put("BrMidMAE", f(mid["mae"], 2))
#|
#|     s = js("spatial_holdout.json")
#|     put("HoldStateRtwo", f(s["r2"]))
#|
#|     r = js("rank_stability.json")
#|     lc = r["learning_curve"]
#|     put("LcTenRtwo", f(lc["010pct"]["r2"]))
#|     put("LcHalfRtwo", f(lc["050pct"]["r2"]))
#|     put("LcFullRtwo", f(lc["100pct"]["r2"]))
#|
#|     ns = js("null_skill.json")
#|     put("NullSkillWithin", f(ns["within_county"]["r2"]))
#|     put("NullSkillBetween", f(ns["between_county"]["r2"]))
#|
#|     t = js("temporal_validation.json")
#|     put("LoyoMean", f(t["leave_one_year_out"]["r2_mean"]))
#|     put("FwdMean", f(t["forward_chaining"]["r2_mean"]))
#|
#|     inf = js("influence_drop.json")
#|     w5 = inf["arms"]["worst_residual_5pct"]
#|     put("InflDropN", n(w5["n_counties_dropped"]))
#|     put("InflDropRtwo", f(w5["r2"]))
#|     put("InflOverlap", n(inf["topk_overlap"]["shared"]))
#|     put("InflTopK", n(inf["topk_overlap"]["topk"]))
#|
#|     mo = js("modality_arms.json")
#|     put("ArmAllRtwo", f(mo["arms"]["all_admitted"]["r2"]))
#|     put("ArmAllN", n(mo["arms"]["all_admitted"]["n_features"]))
#|
#|     # ------------------------------------------- fields with several summaries
#|     sc = js("summary_statistic_census.json")
#|     put("CensusMedianSpread", f(sc["median_spread"], 1))
#|
#|     # background rows of the attribution the turning points are located on
#|     sdc = js("shap_dependence_check.json")
#|     put("InvShapBackground", n(sdc["background"]))
#|
#|     # ------------------------------------------------------- lifetime
#|     lt = js("lifetime_literature_test.json")
#|     rf = lt["rank_free_test"]
#|     put("LtSpecies", n(lt["n_species"]))
#|     put("LtPairs", n(lt["n_pairs"]))
#|     put("LtRho", f(rf["spearman_rho"], 2))
#|     put("LtPermP", p(rf["permutation_p"]))
#|     put("LtPerms", n(rf["n_permutations"]))
#|     put("LtCutoff", f"{lt['published_cutoff']:g}")
#|     u = lt["unit_comparison"]["species_level"]
#|     put("LtShortWins", n(u["short_wins"]))
#|     put("LtShortN", n(u["n_short"]))
#|     put("LtLongWins", n(u["long_wins"]))
#|     put("LtLongN", n(u["n_long"]))
#|     put("LtShortRatio", f(u["short_median_ratio"], 2))
#|     put("LtLongRatio", f(u["long_median_ratio"], 2))
#|     sw = lt["cutoff_sweep"]
#|     put("LtSweepN", n(sw["n_cutoffs"]))
#|     put("LtSweepDirection", n(sw["n_species_level_direction_correct"]))
#|     put("LtSweepSig", n(sw["n_species_level_significant"]))
#|     jj = lt["jackknife"]
#|     put("LtJackLo", f(jj["rho_min"], 2))
#|     put("LtJackHi", f(jj["rho_max"], 2))
#|     vs = lt["value_sensitivity"]
#|     put("LtValueDraws", n(vs["n_draws"]))
#|     put("LtValueDex", f(vs["spread_dex"], 1))
#|     put("LtValueNegative", pct(vs["share_rho_negative"]))
#|
#|     # ------------------------------------------------------- turning points
#|     tp = tab("turning_points.csv").set_index("feature")
#|     put("TpBoot", n(tp["n_boot_attempted"].max()))
#|     put("TpStable", n(tp["stable"].astype(bool).sum()))
#|     put("TpTotal", n(len(tp)))
#|     for feat, mac, nd in (("prism_tdmean_min", "TpDew", 2),
#|                           ("eng2_wetbulb_night", "TpWet", 2),
#|                           ("eng_night_cooling_eff", "TpNce", 3),
#|                           ("eng2_night_cooling_efficiency", "TpNceMean", 3),
#|                           ("lst_night_1km_mean", "TpLst", 2),
#|                           ("pm25_mean", "TpPm", 2),
#|                           ("prism_tmin_min", "TpTmin", 2),
#|                           ("cams_fot_formaldehyde_above75th_percentile", "TpHcho", 1),
#|                           ("eng2_hcho_no2_ratio", "TpRatio", 3)):
#|         rr = tp.loc[feat]
#|         put(mac, f(rr["point_estimate"], nd))
#|         put(mac + "Lo", f(rr["ci_lo"], nd))
#|         put(mac + "Hi", f(rr["ci_hi"], nd))
#|         put(mac + "Width", f(100 * rr["width_frac"], 1))
#|         put(mac + "Above", f(rr["pct_above"], 1))
#|     trip = tp.loc[["prism_tdmean_min", "eng2_wetbulb_night",
#|                    "eng_night_cooling_eff"], "width_frac"]
#|     put("TpTripletWidthMax", f(100 * trip.max(), 1))
#|
#|     # --------------------------- composition, dew point and modalities
#|     # The production ranking (the main run's out-of-fold SHAP ranking) gives
#|     # the chemical-family shares, the 75th against 90th percentile durations
#|     # and the seven summaries of dew point; the main run's temperature strata
#|     # and modality leave-one-out tables give the rest. The family rules are
#|     # the ones the composition figure draws (paper_figures.FAMILY_RULES).
#|     LINES.append(f"\n% ---- {C.RANKING_A.name} (production ranking)")
#|     rk = pd.read_csv(C.RANKING_A)
#|     rnk = dict(zip(rk["feature"], rk["rank"]))
#|     val = dict(zip(rk["feature"], rk["mean_abs_shap"]))
#|     from paper_figures import cams_family                      # noqa: E402
#|     comp = rk[rk["feature"].str.startswith("cams")].copy()
#|     comp["family"] = comp["feature"].map(cams_family)
#|     tot = comp["mean_abs_shap"].sum()
#|     fam = comp.groupby("family")["mean_abs_shap"].agg(["sum", "size"])
#|     fam = fam.sort_values("sum", ascending=False)
#|     put("CompN", n(len(comp)))
#|     put("FamN", n(len(fam)))
#|     put("FamOxVOCPct", f(100 * fam.loc["Oxidised VOC", "sum"] / tot, 1))
#|     put("FamOxVOCn", n(fam.loc["Oxidised VOC", "size"]))
#|     put("FamSecondName", fam.index[1].lower())
#|     put("FamSecondPct", f(100 * fam["sum"].iloc[1] / tot, 1))
#|     hcho = comp[comp["feature"].str.contains("formaldehyde")]
#|     put("FamHchoPct", f(100 * hcho["mean_abs_shap"].sum() / tot, 1))
#|     put("FamHchoN", n(len(hcho)))
#|     p90 = "cams_fot_formaldehyde_above_90th_percentile"
#|     put("HchoNinetyRank", n(rnk[p90]))
#|     du = comp[comp["feature"].str.contains("_fot_")
#|               & comp["feature"].str.contains("percentile")].copy()
#|     du["pct"] = np.where(du["feature"].str.contains("75th"), 75, 90)
#|     g = du.groupby("pct")["mean_abs_shap"].agg(["mean", "size"])
#|     put("PctRatio", f(g.loc[75, "mean"] / g.loc[90, "mean"], 1))
#|     du["species"] = du["feature"].str.replace(r"_?above_?\d+th_percentile", "",
#|                                               regex=True)
#|     pv = du.pivot_table(index="species", columns="pct",
#|                         values="mean_abs_shap").dropna()
#|     put("PctSpeciesN", n(len(pv)))
#|     put("PctSpeciesWins", n((pv[75] > pv[90]).sum()))
#|     put("PctSpeciesMedRatio", f((pv[75] / pv[90]).median(), 2))
#|     pm = pv[pv.index.str.contains(r"pm")]
#|     put("PctPmLo", f((pm[90] / pm[75]).min(), 1))
#|     put("PctPmHi", f((pm[90] / pm[75]).max(), 1))
#|     nohcho = du[~du["feature"].str.contains("formaldehyde")]
#|     g2 = nohcho.groupby("pct")["mean_abs_shap"].mean()
#|     put("PctRatioNoHcho", f(g2.loc[75] / g2.loc[90], 1))
#|     put("HchoPctRatio", f(val["cams_fot_formaldehyde_above75th_percentile"] / val[p90], 1))
#|
#|     # a static field for contrast: soil bulk density, 90th percentile and mean
#|     put("SoilPninetyRank", n(rnk["soil_bulk_density_30cm_p90"]))
#|     put("SoilMeanRank", n(rnk["soil_bulk_density_30cm_mean"]))
#|     put("SoilRatio", f(val["soil_bulk_density_30cm_p90"]
#|                        / val["soil_bulk_density_30cm_mean"], 1))
#|
#|     dpp = rk[rk["feature"].str.startswith("prism_tdmean_")].sort_values("rank")
#|     put("DPSummariesN", n(len(dpp)))
#|     put("DPMinRank", n(rnk["prism_tdmean_min"]))
#|     put("DPNextRank", n(dpp["rank"].iloc[1]))
#|     put("DPMeanRank", n(rnk["prism_tdmean_mean"]))
#|     put("DPProfileRatio", f(dpp["mean_abs_shap"].iloc[0] / dpp["mean_abs_shap"].iloc[1], 1))
#|
#|     st = C.load_table("A_temp_strata")
#|     LINES.append("\n% ---- A_temp_strata.csv (main run)")
#|     crossing = st[np.isfinite(st["zero_cross"])]
#|     put("StrataCrossLo", f(crossing["zero_cross"].min(), 2))
#|     put("StrataCrossHi", f(crossing["zero_cross"].max(), 2))
#|
#|     loo = C.load_table("A_modality_loo").sort_values("delta")
#|     LINES.append("\n% ---- A_modality_loo.csv (main run)")
#|     put("LOOnModalities", n(len(loo)))
#|     put("LOOcams", f(loo["delta"].iloc[0], 4))
#|     put("LOOcamsN", n(loo["n_removed"].iloc[0]))
#|
#|     nx = js("null_crossing.json")["overall"]
#|     put("NullCrossTotal", n(nx["total_permutations"]))
#|     put("NullCrossPasses", n(nx["total_gate_passes"]))
#|     put("NullCrossDraws", n(nx["n_draws_each"]))
#|     put("NullCrossBoot", n(nx["n_boot_per_draw"]))
#|     put("NullCrossShifted", n(nx["shifted_any_pass"]))
#|
#|     pt = tab("perturbation_thresholds.csv")
#|     agg = pt.groupby("magnitude")["shift_over_interval_width"].agg(
#|         med=lambda s: s.abs().median(), mx=lambda s: s.abs().max())
#|     for mag, tag in ((0.01, "One"), (0.02, "Two"), (0.05, "Five"), (0.10, "Ten")):
#|         put(f"Pert{tag}Med", f(agg.loc[mag, "med"], 2))
#|         put(f"Pert{tag}Max", f(agg.loc[mag, "mx"], 2))
#|     put("PertDraws", n(pt["n_found"].max()))
#|
#|     # ------------------------------------------------------- regional invariance
#|     for name, tag in (("threshold_regional_invariance_all.json", "Med"),
#|                       ("threshold_regional_invariance_100W_all.json", "Mer")):
#|         ri = js(name)
#|         cut = abs(ri["cut_degrees_east"])
#|         put(f"Reg{tag}Cut", f"{cut:g}" if cut == int(cut) else f(cut, 1))
#|         res = ri["results"]
#|         anyf = next(iter(res.values()))
#|         put(f"Reg{tag}West", n(anyf["n_west"]))
#|         put(f"Reg{tag}East", n(anyf["n_east"]))
#|         put(f"Reg{tag}Matched", n(anyf["n_west"] + anyf["n_east"]))
#|         for feat, mac in (("prism_tdmean_min", "Dew"),
#|                           ("eng2_wetbulb_night", "Wet"),
#|                           ("eng_night_cooling_eff", "Nce"),
#|                           ("lst_night_1km_mean", "Lst"),
#|                           ("pm25_mean", "Pm"),
#|                           ("eng2_hcho_no2_ratio", "Ratio"),
#|                           ("eng2_wetbulb_hot", "WetHot"),
#|                           ("prism_tmin_min", "Tmin")):
#|             d = res[feat].get("delta_pct_of_range")
#|             put(f"Reg{tag}{mac}", "n/a" if d is None else f(d, 1))
#|             if feat in ("prism_tdmean_min", "eng2_wetbulb_night",
#|                         "eng_night_cooling_eff") and d is not None:
#|                 put(f"Reg{tag}{mac}Abs", f(res[feat]["delta"],
#|                                             3 if feat.startswith("eng_") else 2))
#|         trip = [res[k].get("delta_pct_of_range") for k in
#|                 ("prism_tdmean_min", "eng2_wetbulb_night", "eng_night_cooling_eff")]
#|         put(f"Reg{tag}TripletMax", f(max(trip), 1))
#|
#|     # ------------------------------------------------------- failure geography
#|     rs = js("residual_social_determinants.json")
#|     put("ResSdohRtwo", f(rs["oof_r2_residual_from_sdoh"], 2))
#|     put("ResEnvRtwo", f(rs["oof_r2_residual_from_environment"], 2))
#|     put("ResSdohN", n(rs["n_sdoh_columns"]))
#|     put("ResEnvN", n(rs["n_env_columns"]))
#|     put("ResSD", f(rs["residual_sd"], 2))
#|     put("ResMissN", n(rs["n_large_miss"]))
#|     put("ResMissYears", f"{rs['large_miss_threshold']:g}")
#|     ind = {i["column"]: i for i in rs["indicators"]}
#|     for col, mac in (("sdoh_saipe_pct_pov", "Pov"),
#|                      ("sdoh_saipe_median_hh_income", "Inc"),
#|                      ("sdoh_unemployment_rate", "Unemp"),
#|                      ("sdoh_acs_pct_uninsured", "Unins"),
#|                      ("sdoh_bachelor's_degree_or_higher_%", "Bach"),
#|                      ("sdoh_disability_rate", "Dis")):
#|         put(f"Res{mac}Rho", f(ind[col]["rho"], 2))
#|         put(f"Res{mac}Pct", f(rs["large_miss_profile"][col]
#|                                ["median_percentile_large_miss"], 0))
#|     put("ResHolmMax", p(max(i["p_holm"] for i in rs["indicators"])))
#|     put("ResAbsRhoLo", f(min(abs(i["rho"]) for i in rs["indicators"]), 2))
#|     put("ResAbsRhoHi", f(max(abs(i["rho"]) for i in rs["indicators"]), 2))
#|
#|     sens_path = C.DERIVED / "residual_audit_sensitivity.json"
#|     if sens_path.exists():                       # written by script 30
#|         sv = json.loads(sens_path.read_text())
#|         by_src = sv["columns_by_source"]
#|         put("ResSdohYearLo", str(sv["years"][0]))
#|         put("ResSdohYearHi", str(sv["years"][1]))
#|         put("ResAcsN", n(by_src.get("American Community Survey", 0)))
#|         arms = sv["arms"]
#|         put("ResNoChrN", n(arms["without_county_health_rankings"]["n_columns"]))
#|         put("ResNoChrRtwo", f(arms["without_county_health_rankings"]["oof_r2"], 2))
#|         put("ResAcsRtwo", f(arms["american_community_survey_only"]["oof_r2"], 2))
#|
#|     # Moran's I of the production model's own residual (the residual the
#|     # audit uses and Figure 8 maps), and of the outcome for comparison
#|     mi = js("residual_spatial_structure.json")
#|     put("ResMoranI", f(mi["production"]["morans_i"], 3))
#|     put("ResMoranP", p(mi["production"]["p_value"]))
#|     put("ResMoranOutcome", f(mi["outcome"]["morans_i"], 2))
#|
#|     # refits without the formaldehyde durations (step 17)
#|     fd = js("formaldehyde_drop.json")
#|     put("HchoDropFullRtwo", f(fd["r2"]["both"], 4))
#|     put("HchoDropOneRtwo", f(fd["r2"]["without_p75"], 4))
#|     put("HchoDropBothRtwo", f(fd["r2"]["without_both"], 4))
#|     put("HchoDropInheritPct", f(100 * fd["inherit_p90_of_p75"], 0))
#|     put("HchoDropShapN", n(fd["shap_rows"]))
#|     put("HchoDropShapBg", n(fd["shap_background"]))
#|     col = "cams_total_column_formaldehyde_x1e9"
#|     put("HchoColRankWith", n(fd["rank"][col]["both"]))
#|     put("HchoColRankWithout", n(fd["rank"][col]["without_both"]))
#|     put("HchoDropBothCost", f(fd["r2"]["both"] - fd["r2"]["without_both"], 3))
#|     surf = "cams_formaldehyde_x1e9"
#|     put("HchoSurfRankWith", n(fd["rank"][surf]["both"]))
#|     put("HchoSurfRankWithout", n(fd["rank"][surf]["without_both"]))
#|
#|     # The reference of every SHAP value: the model's average prediction. A
#|     # turning point is where a variable's contribution crosses this level.
#|     LINES.append(f"\n% ---- {C.OOF_A.name}")
#|     oof = pd.read_csv(C.OOF_A)
#|     put("PredMean", f(oof["pred"].mean(), 1))
#|
#|     # The reference configuration every refit uses (scripts 04, 12, 22).
#|     LINES.append("\n% ---- nce_refit.json (refit protocol)")
#|     proto = json.load(open(C.DERIVED / "nce_refit.json"))["protocol"]
#|     put("RefitTrees", n(proto["n_estimators"]))
#|     put("RefitLR", f(proto["learning_rate"], 2))
#|     put("RefitDepth", n(proto["max_depth"]))
#|     put("RefitShapN", n(proto["nshap"]))
#|     put("RefitShapBg", n(proto["nbg"]))
#|     cond = json.load(open(C.DERIVED / "nce_refit.json"))["conditions"]
#|     put("RefitFullTailSHAP", f(cond["FULL"]["pct_shap"], 4))
#|
#|     # Library versions of the final run, and how much the numbers moved
#|     # against the earlier run under other releases (script 34).
#|     lv_path = C.DERIVED / "library_version_sensitivity.json"
#|     if lv_path.exists():
#|         LINES.append(f"\n% ---- {lv_path.name}")
#|         lv = json.loads(lv_path.read_text())
#|         vf, vr = lv["versions_final"], lv["versions_reference"]
#|         for mac, key in (("VerPython", "python"), ("VerXgb", "xgboost"),
#|                          ("VerShap", "shap"), ("VerSklearn", "sklearn"),
#|                          ("VerNumpy", "numpy"), ("VerPandas", "pandas"),
#|                          ("VerScipy", "scipy"), ("VerCatboost", "catboost")):
#|             put(mac, str(vf.get(key, "")))
#|
#|     write_tables(Path(a.out).parent / "tables_generated_slim.tex", tp)
#|     write_data_table(Path(a.out).parent / "tables_data.tex")
#|     manuscript = Path(a.out).parent / "main.tex"
#|     if manuscript.exists():
#|         names = {re.match(r"\\providecommand\{\\([A-Za-z]+)\}", x).group(1)
#|                  for x in LINES if x.startswith("\\providecommand")}
#|         write_base(Path(a.out).parent / "constants_base.tex", manuscript, names)
#|
#|     head = ("% constants_results.tex -- written by analysis/build_paper_constants.py\n"
#|             "% Do not edit by hand: rerun the script. Each block names the result\n"
#|             "% file its values were read from.\n")
#|     Path(a.out).write_text(head + "\n".join(LINES) + "\n")
#|     print(f"  [saved] {a.out}  ({sum(1 for x in LINES if x.startswith(chr(92)))} macros)")
#|     check_text_assumptions(Path(a.out).parent)
#|
#|
#| # ---------------------------------------------------------------------------
#| #  Text-assumption checks
#| # ---------------------------------------------------------------------------
#| # Numbers in the manuscript are macros, so a rerun updates them everywhere.
#| # A few sentences, however, also make a qualitative statement about a number
#| # ("not significant", "weakly clustered", "barely move"). If a rerun moved a
#| # number across such a line, the macro would update but the sentence would
#| # become wrong. These checks list every such sentence and test it against the
#| # freshly written values. A failed check never stops the build: it prints
#| # WARN, names the section to reread, and is recorded in
#| # derived/text_assumption_checks.json.
#| TEXT_ASSUMPTIONS = [
#|     ("1, 3.3, 4.2, 5: lifetime correlation is inverse",
#|      lambda v: v["LtRho"] < 0),
#|     ("3.3, 4.2: the lifetime relation is 'modest' (one-sided p not below 0.05)",
#|      lambda v: v["LtPermP"] >= 0.05),
#|     ("3.3, 5: the oxygenated family leads the composition attribution",
#|      lambda v: v["FamOxVOCPct"] > v["FamSecondPct"]),
#|     ("3.3: the 75th-percentile advantage is carried by formaldehyde",
#|      lambda v: v["PctRatioNoHcho"] < 1.5 and v["PctSpeciesWins"] <= v["PctSpeciesN"] / 2 + 1),
#|     ("3.2, 5: minimum dew point leads the next dew-point summary several-fold",
#|      lambda v: v["DPProfileRatio"] > 5),
#|     ("3.1: removing composition costs the most skill",
#|      lambda v: v["LOOcams"] < 0),
#|     ("3.1, 4.4, 5: holding out whole states lowers skill",
#|      lambda v: v["HoldStateRtwo"] < v["Rtwo"]),
#|     ("3.5: only 10% noise moves a value by more than one interval width",
#|      lambda v: v["PertTenMax"] > 1 and v["PertFiveMax"] <= 1),
#|     ("3.5, 5: no permuted curve passes the stability criterion",
#|      lambda v: v["NullCrossPasses"] == 0),
#|     ("1, 3.5, 4.3, 5: the three moisture and night-cooling points move little "
#|      "between regions (<= 5% of range)",
#|      lambda v: v["RegMerTripletMax"] <= 5 and v["RegMedTripletMax"] <= 5),
#|     ("1, 3.5, 4.3, 5: at the 100th meridian two of them shift by more than their "
#|      "own interval widths",
#|      lambda v: v["RegMerDew"] > v["TpDewWidth"] and v["RegMerNce"] > v["TpNceWidth"]),
#|     ("1, 3.5, 5: satellite PM2.5 moves more than the three moisture and "
#|      "night-cooling points",
#|      lambda v: v["RegMerPm"] > v["RegMerTripletMax"]),
#|     ("3.5, 4.3: minimum temperature also moves little at the 100th meridian (<= 5%)",
#|      lambda v: v["RegMerTmin"] <= 5),
#|     ("3.5, 4.3: the formaldehyde to NO2 ratio moved at the median longitude (> 5%)",
#|      lambda v: v["RegMedRatio"] > 5),
#|     ("2.10, 3.4: crossings lie near the middle of each distribution (25 to 75%)",
#|      lambda v: all(25 <= v[k] <= 75 for k in ("TpDewAbove", "TpWetAbove", "TpNceAbove"))),
#|     ("3.6, 4.4: residuals 'weakly but significantly' clustered",
#|      lambda v: v["ResMoranP"] < 0.05 and abs(v["ResMoranI"]) < 0.1),
#|     ("3.6: the residual is far less clustered than the outcome",
#|      lambda v: v["ResMoranI"] < 0.2 * v["ResMoranOutcome"]),
#|     ("4.4, 5: socioeconomic columns explain under half of the residual variance",
#|      lambda v: v["ResSdohRtwo"] < 0.5),
#|     ("1, 3.2, 5: removing both formaldehyde durations costs little skill (< 0.01)",
#|      lambda v: 0 <= v["HchoDropFullRtwo"] - v["HchoDropBothRtwo"] < 0.01),
#|     ("3.2, 4, 5: the drop-one NCE refit changes skill by less than 0.01",
#|      lambda v: abs(v["RefitFullRtwo"] - v["RefitDropRtwo"]) < 0.01),
#|     ("3.1, 5: error is largest in the lowest bracket",
#|      lambda v: v["BrLowMAE"] > v["BrMidMAE"]),
#|     ("2.8: refit configuration reproduces production skill (within 0.01)",
#|      lambda v: abs(v["ArmAllRtwo"] - v["Rtwo"]) < 0.01),
#| ]
#|
#|
#| def _macro_values(folder):
#|     """Read every macro written to the constants files as a float where the
#|     value is a number (signs, thin separators and 10^x forms understood)."""
#|     vals = {}
#|     rx = re.compile(r"\\(?:renewcommand|newcommand)\{\\([A-Za-z]+)\}\{(.*)\}\s*$")
#|     for name in ("constants_base.tex", "constants_results.tex"):
#|         path = Path(folder) / name
#|         if not path.exists():
#|             continue
#|         for line in path.read_text().splitlines():
#|             m = rx.search(line)
#|             if m and m.group(2):
#|                 s = m.group(2).replace("{,}", "").replace("$-$", "-").replace("$", "")
#|                 s = s.replace("<", "").replace("\\,", "")
#|                 mm = re.match(r"^\s*(-?[0-9.]+)\\times10\^\{(-?\d+)\}?\s*$", s)
#|                 try:
#|                     vals[m.group(1)] = (float(mm.group(1)) * 10 ** int(mm.group(2))
#|                                         if mm else float(s))
#|                 except ValueError:
#|                     pass
#|     return vals
#|
#|
#| def check_text_assumptions(folder):
#|     vals = _macro_values(folder)
#|     out, bad = [], 0
#|     print("\n  TEXT-ASSUMPTION CHECKS (a sentence that states a quality of a number)")
#|     for label, test in TEXT_ASSUMPTIONS:
#|         try:
#|             ok = bool(test(vals))
#|             status = "PASS" if ok else "WARN"
#|         except KeyError as e:
#|             ok, status = False, f"MISSING {e}"
#|         bad += not ok
#|         out.append(dict(check=label, status=status))
#|         print(f"    {status:<8} {label}")
#|     if bad:
#|         print(f"  {bad} sentence(s) to reread before the paper is rebuilt.")
#|     else:
#|         print("  every qualitative statement still matches its number.")
#|     (C.DERIVED / "text_assumption_checks.json").write_text(
#|         json.dumps(dict(checks=out, n_warn=bad), indent=1))
#|
#|
#| # ---------------------------------------------------------------------------
#| #  The base constants: the main run's generated values, flattened
#| # ---------------------------------------------------------------------------
#| BASE_SOURCES = ("constants.tex", "constants_v10.tex", "constants_thresholds.tex")
#| _HEAD = re.compile(r"\\(newcommand|renewcommand|providecommand)\{\\([A-Za-z]+)\}\{")
#|
#|
#| def resolve_tex_macros(paths):
#|     """Resolve macro definitions the way TeX would, in file order:
#|     \\providecommand only when undefined, \\newcommand and \\renewcommand
#|     always. Returns {name: raw TeX value}. Lines that are comments are
#|     skipped; a definition whose name is not a valid control word is ignored,
#|     as is any stray text between definitions."""
#|     vals = {}
#|     for path in paths:
#|         text = "\n".join("" if ln.lstrip().startswith("%") else ln.split("%%")[0]
#|                          for ln in Path(path).read_text().splitlines())
#|         pos = 0
#|         while True:
#|             m = _HEAD.search(text, pos)
#|             if not m:
#|                 break
#|             depth, i = 1, m.end()
#|             while depth and i < len(text):
#|                 depth += {"{": 1, "}": -1}.get(text[i], 0)
#|                 i += 1
#|             value, pos = text[m.end():i - 1], i
#|             if m.group(1) == "providecommand" and m.group(2) in vals:
#|                 continue
#|             if value == "" and m.group(1) == "providecommand":
#|                 vals.setdefault(m.group(2), value)
#|                 continue
#|             vals[m.group(2)] = " ".join(value.split())
#|     return vals
#|
#|
#| def write_base(out_path, manuscript, results_names):
#|     """constants_base.tex: every main-run macro the manuscript uses and this
#|     script does not itself compute, resolved from the main run's generated
#|     constants files and written flat, one definition each."""
#|     # In the paper folder the main run's constants files are kept, unchanged,
#|     # in analysis/main_run_constants/; in the working tree they sit beside
#|     # main.tex. Whichever exists is used.
#|     src_dir = Path(HERE) / "main_run_constants"
#|     if not src_dir.exists():
#|         src_dir = C.PAPER
#|     found = [src_dir / n for n in BASE_SOURCES if (src_dir / n).exists()]
#|     if not found:
#|         print("  [kept] constants_base.tex: the main run's constants files were "
#|               "not found, so the existing file is left as it is")
#|         return
#|     base = resolve_tex_macros(found)
#|     used = set(re.findall(r"\\([A-Za-z]+)", Path(manuscript).read_text()))
#|     # the schematic figures print constants too; keep the ones they name
#|     schematic = Path(HERE) / "figure01_methodology_and_graphical_abstract.py"
#|     if schematic.exists():
#|         used |= set(re.findall(r"K\[['\"]([A-Za-z]+)['\"]\]",
#|                                schematic.read_text()))
#|     keep = sorted(k for k in used if k in base and k not in results_names)
#|     lines = ["% constants_base.tex -- written by analysis/build_paper_constants.py",
#|              "% The main run's generated values that the manuscript quotes,",
#|              "% resolved from " + ", ".join(BASE_SOURCES) + " in",
#|              "% analysis/main_run_constants/ (written by the main run's constants",
#|              "% builder) and flattened to one definition each. Do not edit by hand.",
#|              ""]
#|     lines += [f"\\newcommand{{\\{k}}}{{{base[k]}}}" for k in keep]
#|     Path(out_path).write_text("\n".join(lines) + "\n")
#|     print(f"  [saved] {out_path}  ({len(keep)} macros)")
#|
#|
#| UNIT_TEX = {"degC": "$^{\\circ}$C", "%": "\\% of year", "ug/m3": "\\textmu g\\,m$^{-3}$"}
#| UNIT_OTHER = {"usda_cropland_usda_corn_pct": "fraction of area"}
#| SHORT_TEX = {
#|     "prism_tdmean_min": "Minimum dew point",
#|     "eng2_night_cooling_efficiency": "NCE, mean form",
#|     "eng_night_cooling_eff": "NCE, tail form",
#|     "lst_night_1km_mean": "Mean night LST",
#|     "eng2_wetbulb_night": "Nocturnal wet-bulb temperature",
#|     "eng2_wetbulb_hot": "Warm-hour wet-bulb temperature",
#|     "eng2_hcho_no2_ratio": "Formaldehyde to nitrogen dioxide ratio",
#|     "pm25_mean": "Annual mean PM$_{2.5}$",
#|     "prism_tmin_min": "Minimum daily minimum temperature",
#|     "cams_fot_formaldehyde_above75th_percentile": "Formaldehyde, time above 75th percentile",
#|     "usda_cropland_usda_corn_pct": "Cropland in corn",
#| }
#|
#|
#| def write_tables(path, tp):
#|     """Two appendix tables, generated so no value in them is typed in."""
#|     out = ["% tables_generated.tex -- written by analysis/build_paper_constants.py",
#|            "% Do not edit by hand.", ""]
#|
#|     # Table A4: every located turning point and the three criteria
#|     found_all = bool((tp["found_frac"] >= 1).all())
#|     out += [r"\begin{table}[H]", r"\caption{Every candidate turning point. The "
#|             r"interval is the 2.5th to 97.5th percentile of the located value over "
#|             r"the bootstrap resamples, and its width is given as a share of the "
#|             r"variable's observed range. A candidate is stable when a crossing is "
#|             r"found in at least 60\% of resamples, the width is below 40\% and the "
#|             r"full-sample value lies inside the interval. PM$_{2.5}$ is given in "
#|             r"the product's native unit."
#|             + (" A crossing was found in every resample for every candidate."
#|                if found_all else "") + r"\label{tab:turning}}",
#|             r"\centering\footnotesize",
#|             r"\begin{tabular}{@{}llrrrl@{}}", r"\toprule",
#|             r"Variable & Unit & Value & 95\% interval & Width (\%) & Stable \\",
#|             r"\midrule"]
#|     for feat, r in tp.sort_values("width_frac").iterrows():
#|         unit = UNIT_TEX.get(str(r["unit"]), UNIT_OTHER.get(feat, "dimensionless"))
#|         # PM2.5 is stored at one tenth of the product's native unit; the table
#|         # prints it in micrograms per cubic metre so no factor is left to read
#|         k = 10.0 if str(r["unit"]) == "ug/m3" else 1.0
#|         nd = 3 if abs(k * r["point_estimate"]) < 2 else 2
#|         out.append(f"{SHORT_TEX.get(feat, feat)} & {unit} & "
#|                    f"{f(k * r['point_estimate'], nd)} & "
#|                    f"{f(k * r['ci_lo'], nd)} to {f(k * r['ci_hi'], nd)} & "
#|                    f"{f(100 * r['width_frac'], 1)} & "
#|                    f"{'yes' if bool(r['stable']) else 'no'} \\\\")
#|     out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
#|
#|     # Table A3: the species of the lifetime test (script 25)
#|     lt = json.load(open(C.DERIVED / "lifetime_literature_test.json"))
#|     sp = sorted(lt["species_table"], key=lambda r: r["lifetime_days"])
#|     names = {"pm2 5": "Fine particulate matter (reanalysis)",
#|              "nitrogen monoxide": "Nitric oxide"}
#|     out += [r"\begin{table}[H]", r"\caption{The species of the lifetime test, "
#|             r"ordered by the atmospheric lifetime taken from the literature (the "
#|             r"tropospheric burden divided by the removal rate, an e-folding time; "
#|             r"the half-life is this lifetime times $\ln 2$), "
#|             r"with the number of duration and concentration pairs admitted for "
#|             r"each, the median ratio of duration to concentration importance and "
#|             r"whether duration was credited more. All series are from the "
#|             r"atmospheric composition reanalysis. For nitric oxide the lifetime "
#|             r"is that of nitric oxide and nitrogen dioxide together, because the two "
#|             r"interconvert within minutes in daylight.\label{tab:species}}",
#|             r"\centering\footnotesize",
#|             r"\begin{tabular}{@{}lrrrl@{}}", r"\toprule",
#|             r"Species & Lifetime (days) & Pairs & Median ratio & Duration favoured \\",
#|             r"\midrule"]
#|     for r in sp:
#|         nm = names.get(r["species"], r["species"].capitalize())
#|         lt_d = r["lifetime_days"]
#|         lt_s = f"{lt_d:.0e}".replace("e-0", "$\\times 10^{-") + "}$" if lt_d < 0.01 else f"{lt_d:g}"
#|         out.append(f"{nm} & {lt_s} & {r['n_rows']} & {f(r['median_ratio'], 2)} & "
#|                    f"{'yes' if r['duration_wins'] else 'no'} \\\\")
#|     out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
#|
#|     # Table A1: the learner and its settings, from the tournament output
#|     ls = json.load(open(C.DERIVED / "learner_settings.json"))
#|     xp = ls["params"]["XGB"]
#|     names = {"n_estimators": "Number of trees", "max_depth": "Maximum depth",
#|              "learning_rate": "Learning rate", "subsample": "Row subsample",
#|              "colsample_bytree": "Column subsample per tree",
#|              "min_child_weight": "Minimum child weight",
#|              "reg_alpha": "L1 regularisation", "reg_lambda": "L2 regularisation"}
#|     out += [r"\begin{table}[H]", r"\caption{The fitted learner. Settings of the "
#|             r"XGBoost model and the out-of-fold skill of each learner compared on "
#|             r"identical county-grouped folds. Settings not listed are the "
#|             r"library defaults.\label{tab:learner}}",
#|             r"\centering\small", r"\begin{tabular}{@{}lr@{}}", r"\toprule",
#|             r"Setting & Value \\", r"\midrule"]
#|     for k, lab in names.items():
#|         if k not in xp:                    # not set: the library default applies
#|             continue
#|         v = xp[k]
#|         out.append(f"{lab} & {v if isinstance(v, int) else f'{v:.3g}'} \\\\")
#|     out.append(r"\midrule")
#|     out.append(r"Learner & Out-of-fold $R^{2}$ (mean absolute error, years) \\")
#|     out.append(r"\midrule")
#|     for k, v in ls["scores"].items():
#|         if "r2" in v:
#|             out.append(f"{k} & {v['r2']:.3f} ({v['mae']:.3f}) \\\\")
#|     out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
#|     # The learner table goes to its own file (Appendix A); the species and
#|     # turning-point tables, in that order, to the file Appendix B inputs.
#|     text = "\n".join(out) + "\n"
#|     head, *tbls = text.split("\\begin{table}[H]")
#|     blocks = {re.search(r"\\label\{(tab:[a-z]+)\}", b).group(1):
#|               "\\begin{table}[H]" + b for b in tbls}
#|     Path(path).write_text(
#|         head.replace("tables_generated.tex", "tables_generated_slim.tex")
#|         + "".join(blocks[k] for k in ("tab:species", "tab:turning")))
#|     lp = Path(path).with_name("tables_learner.tex")
#|     lp.write_text(head.replace("tables_generated.tex", "tables_learner.tex")
#|                   + blocks["tab:learner"])
#|     print(f"  [saved] {path} and {lp}")
#|
#|
#| # Table 1: every data source with its native resolution, and the number of its
#| # columns among those the model sees (counted from the model's own ranking).
#| DATA_SOURCES = [
#|     # name, variables (examples), spatial, time step, column prefixes
#|     ("MODIS land-surface temperature (Terra)", "Day and night land-surface temperature", "1 km", "8-day", ("lst_",)),
#|     ("MODIS vegetation indices", "Normalised difference and enhanced vegetation indices", "500 m", "16-day", ("ndvi_",)),
#|     ("Landsat 5 and 7", "Surface reflectance, vegetation indices", "30 m", "16-day revisit", ("ls57_",)),
#|     ("Sentinel-1", "Radar backscatter", "10 m", "6 to 12 days, from 2014", ()),
#|     ("Sentinel-2", "Optical reflectance", "10 to 60 m", "5 days, from 2015", ()),
#|     ("Sentinel-5P", "Trace gases, aerosol index", "About 5 km", "Daily, from 2018", ()),
#|     ("Satellite fine particulate matter", "Ground-level PM$_{2.5}$", "About 1 km", "Annual", ("pm25_",)),
#|     ("Night-time lights (DMSP, VIIRS)", "Settlement brightness, rescaled within each sensor", "About 1 km; 500 m", "Annual", ("ntl_",)),
#|     ("Global Surface Water (JRC)", "Water occurrence and class", "30 m", "Annual", ("jrc_",)),
#|     ("Global Human Modification", "Human pressure on land", "1 km", "Single map", ("ghm_",)),
#|     ("Canopy height", "Canopy height and cover", "1 m", "Single map", ("meta_",)),
#|     ("Cropland Data Layer (USDA)", "Share of land in each crop", "30 m", "Annual", ("usda_",)),
#|     ("CAMS global reanalysis (EAC4)", "Trace gases, aerosol optical depth, particulate matter", "0.75\\textdegree{} (about 80 km)", "3-hourly", ("cams_",)),
#|     ("ERA5 reanalysis", "Humidity, pressure, wind, snow, terrain height", "0.25\\textdegree{} (about 28 km)", "Hourly", ("era5_",)),
#|     ("PRISM", "Air temperature, dew point, vapour-pressure deficit, precipitation", "4 km", "Daily", ("prism_",)),
#|     ("TerraClimate", "Water balance, wind, solar radiation", "About 4 km", "Monthly", ("terra_",)),
#|     ("OpenLandMap soil maps", "Bulk density, texture, acidity, organic carbon", "250 m", "Single map", ("soil_",)),
#|     ("Gridded Livestock of the World", "Density of each farm animal", "About 10 km", "Reference years, linear between", ("live_",)),
#|     ("Engineered from the above", "Wet-bulb and heat indices, night cooling efficiency, species ratios", "Of the inputs", "Annual", ("eng_", "eng2_")),
#| ]
#|
#|
#| def write_data_table(path):
#|     feats = pd.read_csv(C.RANKING_A)["feature"].astype(str).tolist()
#|     counts, seen = [], set()
#|     for row in DATA_SOURCES:
#|         hit = [x for x in feats if row[4] and x.startswith(row[4])]
#|         seen.update(hit)
#|         counts.append(len(hit))
#|     missing = sorted(set(feats) - seen)
#|     if missing or sum(counts) != len(feats):
#|         raise SystemExit(f"data table: columns not assigned to a source: {missing[:10]}")
#|     out = ["% tables_data.tex -- written by analysis/build_paper_constants.py",
#|            "% Column counts are read from the model's ranking; do not edit by hand.",
#|            r"\begin{table}[H]",
#|            r"\caption{Data sources and the outcome, with each product's native "
#|            r"resolution and the number of its columns among the "
#|            f"{len(feats)} the model sees. "
#|            r"Single maps enter every year with the same value. The three Sentinel "
#|            r"missions were acquired but none of their columns met the coverage gate."
#|            r"\label{tab:data}}",
#|            r"\centering\footnotesize",
#|            r"\setlength{\tabcolsep}{4pt}",
#|            r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{3.7cm}>{\raggedright\arraybackslash}p{1.8cm}>{\raggedright\arraybackslash}p{1.8cm}r@{}}",
#|            r"\toprule",
#|            r"Source & Variables, for example & Spatial resolution & Time step & Columns \\",
#|            r"\midrule"]
#|     for (nm, var, sp, ts, _), k in zip(DATA_SOURCES, counts):
#|         out.append(f"{nm} & {var} & {sp} & {ts} & {k} \\\\")
#|     out += [r"\midrule",
#|             r"IHME small-area estimates & Life expectancy at birth, both sexes, all races (the outcome) & County & Annual & -- \\",
#|             r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
#|     Path(path).write_text("\n".join(out) + "\n")
#|     print(f"  [saved] {path}")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/build_paper_constants.py
#@@ FILE analysis/paper_figures.py
#| #!/usr/bin/env python3
#| """
#| paper_figures.py -- every numbered figure of the manuscript, in one place.
#|
#| Figure 1 (the study design) and the graphical abstract are drawn by
#| figure01_methodology_and_graphical_abstract.py and can be replaced by
#| hand-made versions saved under the same file names. Figures 2 to 8 are drawn here, each by one function, each reading
#| only result files written by the numbered analysis scripts. No value in any
#| figure is typed in.
#|
#| House style, applied to every figure by one function so no figure can drift:
#|   * one sans-serif face (DejaVu Sans), regular weight throughout; the only
#|     bold is the panel letter;
#|   * 8.5 pt labels and 7.5 pt ticks at the authored width of 5.5 in, which is
#|     the text width of the MDPI layout, so the printed sizes are these sizes;
#|   * panel letters A, B, C in 9.5 pt bold, top left of each panel, each
#|     followed by a short panel title in regular weight;
#|   * colour by meaning, the same meaning in every figure:
#|         central summary (mean, median)          blue    #0072B2
#|         non-central summary (quantile, width)   orange  #E69F00
#|         stable / passes                          green   #009E73
#|         fails / not located                      red     #D55E00
#|         reference, null, background              grey
#|     taken from the Okabe-Ito palette, which separates under the common forms
#|     of colour-vision deficiency;
#|   * every axis carries its unit in parentheses, or says it has none;
#|   * each figure carries a short title in regular weight above its panels
#|     (the TITLES table below); the caption carries the explanation.
#|
#| Output: ../Figures/Figure02_performance.png and .pdf, and so on, PNG at
#| 600 dpi and PDF as vector.
#|
#|     python3 paper_figures.py                 # all of them
#|     python3 paper_figures.py --only 3,6      # just Figures 3 and 6
#|     python3 paper_figures.py --dpi 150 --outdir ../_preview    # a quick look
#| """
#| from __future__ import annotations
#|
#| import argparse
#| import json
#| import os
#| import sys
#|
#| import numpy as np
#| import pandas as pd
#|
#| HERE = os.path.dirname(os.path.abspath(__file__))
#| if HERE not in sys.path:
#|     sys.path.insert(0, HERE)
#|
#| import _config as C                                            # noqa: E402
#|
#| import matplotlib                                              # noqa: E402
#| matplotlib.use("Agg")
#| import matplotlib.pyplot as plt                                # noqa: E402
#| from matplotlib.lines import Line2D                            # noqa: E402
#| from matplotlib.patches import Patch                           # noqa: E402
#|
#| # ===========================================================================
#| #  STYLE
#| # ===========================================================================
#| W = 5.5                       # inches, the MDPI text width
#| BLUE, ORANGE = "#0072B2", "#E69F00"
#| GREEN, RED = "#009E73", "#D55E00"
#| PINK, SKY = "#CC79A7", "#56B4E9"
#| GREY, LIGHT, INK = "#8C8C8C", "#D9D9D9", "#222222"
#| FS, FS_TICK, FS_LET = 8.5, 7.5, 9.5
#|
#| plt.rcParams.update({
#|     "font.family": "DejaVu Sans", "font.size": FS, "font.weight": "normal",
#|     "axes.titlesize": FS, "axes.titleweight": "normal",
#|     "axes.labelsize": FS, "axes.labelweight": "normal",
#|     "axes.labelcolor": INK, "axes.edgecolor": "#555555",
#|     "axes.linewidth": 0.6, "axes.spines.top": False,
#|     "axes.spines.right": False, "axes.grid": False,
#|     "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
#|     "xtick.color": "#444444", "ytick.color": "#444444",
#|     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
#|     "xtick.major.size": 3, "ytick.major.size": 3,
#|     "legend.fontsize": FS_TICK, "legend.frameon": False,
#|     "lines.linewidth": 1.2, "patch.linewidth": 0.0,
#|     "pdf.fonttype": 42, "ps.fonttype": 42,
#|     "figure.facecolor": "white", "axes.facecolor": "white",
#|     "text.color": INK, "mathtext.default": "regular",
#| })
#|
#| DPI = 600
#| OUT = C.FIGURES
#|
#|
#| def letter(ax, s):
#|     """Register a panel letter. It is placed at save time, at the outer left
#|     edge of the panel including its tick labels, so a letter can never sit on
#|     a label or fall off the canvas whatever the panel's margins are."""
#|     fig = ax.figure
#|     if not hasattr(fig, "_letters"):
#|         fig._letters = []
#|     fig._letters.append((ax, s))
#|
#|
#| # Figure titles and panel titles, one place for all of them. The figure title
#| # is set above the panels in regular weight; each panel carries its letter in
#| # bold followed by a short title in regular weight.
#| TITLES = {
#|     "Figure02_performance": (
#|         "Predictive performance of the environment-only model",
#|         {"A": "Skill against baselines", "B": "Learning curve",
#|          "C": "Error by outcome bracket", "D": "Skill under outcome permutation"}),
#|     "Figure03_three_fields": (
#|         "Three fields, several descriptions",
#|         {"A": "Formaldehyde (CAMS)", "B": "Night cooling efficiency (MODIS)",
#|          "C": "Dew point (PRISM)", "D": "Position in the full ranking"}),
#|     "Figure04_drop_one_refit": (
#|         "Drop-one refit of night cooling efficiency",
#|         {"A": "Mean-form attribution", "B": "Mean-form rank",
#|          "C": "Predictive skill"}),
#|     "Figure05_composition_lifetime": (
#|         "Atmospheric composition and lifetime",
#|         {"A": "Chemical families", "B": "75th against 90th percentile",
#|          "C": "Duration advantage", "D": "Advantage against lifetime"}),
#|     "Figure06_turning_point_curves": (
#|         "Turning points of the attribution curves",
#|         {"A": "Minimum dew point", "B": "Night wet-bulb", "C": "Night cooling",
#|          "D": "Formaldehyde duration", "E": "Formaldehyde to NO$_2$",
#|          "F": "Dew point by quartile"}),
#|     "Figure07_turning_point_stability": (
#|         "Stability of the turning points",
#|         {"A": "Interval width", "B": "Permuted", "C": "Regional shift",
#|          "D": "Predictor noise"}),
#|     "Figure08_failure_geography": (
#|         "Geography of model error",
#|         {"A": "County residual", "B": "Predicting the residual",
#|          "C": "Socioeconomic correlates"}),
#| }
#| FS_TITLE = FS + 1.0
#|
#|
#| def _place_letters(fig, panel_titles=None):
#|     """Bold panel letter at the outer left edge of each panel (tick labels
#|     included), followed on the same line by the panel title in regular
#|     weight, both above anything the panel already carries on top."""
#|     panel_titles = panel_titles or {}
#|     fig.canvas.draw()
#|     r = fig.canvas.get_renderer()
#|     inv = fig.transFigure.inverted()
#|     for ax, s in getattr(fig, "_letters", []):
#|         bb = ax.get_tightbbox(r).transformed(inv)
#|         pos = ax.get_position()
#|         # a long axis label can reach above the panel; the letter goes above it
#|         labs = [a.label.get_window_extent(r).transformed(inv).y1
#|                 for a in (ax.xaxis, ax.yaxis)
#|                 if a.label.get_visible() and a.label.get_text()]
#|         y = max([pos.y1, bb.y1] + labs) + 0.006
#|         t = fig.text(bb.x0, y, s, fontsize=FS_LET, fontweight="bold",
#|                      ha="left", va="bottom", color=INK)
#|         t._is_letter = True
#|         title = panel_titles.get(s)
#|         if title:
#|             w = t.get_window_extent(r).transformed(inv).width
#|             fig.text(bb.x0 + w + 0.012, y, title, fontsize=FS, ha="left",
#|                      va="bottom", color=INK)
#|
#|
#| def _place_title(fig, text):
#|     """The figure title, centred above everything else on the canvas."""
#|     fig.canvas.draw()
#|     r = fig.canvas.get_renderer()
#|     top = fig.get_tightbbox(r).y1 / fig.get_figheight()   # inches to figure
#|     fig.text(0.5, top + 0.02, text, fontsize=FS_TITLE, ha="center",
#|              va="bottom", color=INK)
#|
#|
#| def save(fig, stem):
#|     fig_title, panel_titles = TITLES.get(stem, ("", {}))
#|     # every text regular weight first, then the panel letters alone in bold
#|     for t in fig.findobj(matplotlib.text.Text):
#|         t.set_fontweight("normal")
#|     _place_letters(fig, panel_titles)
#|     if fig_title:
#|         _place_title(fig, fig_title)
#|     OUT.mkdir(parents=True, exist_ok=True)
#|     # Matplotlib clips an axis label to the canvas when it computes the tight
#|     # box, so a long label can lose its last characters. Listing every text
#|     # explicitly makes the saved area grow to hold it instead.
#|     extra = fig.get_default_bbox_extra_artists() + [
#|         t for ax in fig.axes for t in (ax.xaxis.label, ax.yaxis.label)
#|         if t.get_visible() and t.get_text()] + list(fig.texts)
#|     for ext in ("png", "pdf"):
#|         fig.savefig(OUT / f"{stem}.{ext}", dpi=DPI, facecolor="white",
#|                     bbox_inches="tight", pad_inches=0.04,
#|                     bbox_extra_artists=extra)
#|     plt.close(fig)
#|     print(f"  [saved] {OUT.name}/{stem}.png and .pdf at {DPI} dpi")
#|
#|
#| def js(name):
#|     return json.load(open(C.DERIVED / name))
#|
#|
#| def tab(name):
#|     p = C.DERIVED / name
#|     return pd.read_csv(p) if p.exists() else C.load_table(name.replace(".csv", ""))
#|
#|
#| # Readable names, with units, for the variables the figures label.
#| NAMES = {
#|     "prism_tdmean_min": ("Minimum dew point", "°C"),
#|     "eng2_night_cooling_efficiency": ("Night cooling efficiency, mean form", "dimensionless"),
#|     "eng_night_cooling_eff": ("Night cooling efficiency, tail form", "dimensionless"),
#|     "lst_night_1km_mean": ("Mean night land-surface temperature", "°C"),
#|     "eng2_wetbulb_night": ("Nocturnal wet-bulb temperature", "°C"),
#|     "eng2_wetbulb_hot": ("Warm-hour wet-bulb temperature", "°C"),
#|     "eng2_hcho_no2_ratio": ("Formaldehyde to NO$_2$ ratio", "dimensionless"),
#|     "pm25_mean": ("Annual mean PM$_{2.5}$", "product units"),
#|     "prism_tmin_min": ("Minimum daily minimum temperature", "°C"),
#|     "cams_fot_formaldehyde_above75th_percentile": ("Formaldehyde, time above P75", "% of year"),
#|     "usda_cropland_usda_corn_pct": ("Cropland in corn", "fraction of area"),
#| }
#| SHORT = {
#|     "prism_tdmean_min": "Minimum dew point",
#|     "eng2_night_cooling_efficiency": "Night cooling, mean form",
#|     "eng_night_cooling_eff": "Night cooling, tail form",
#|     "lst_night_1km_mean": "Mean night surface temperature",
#|     "eng2_wetbulb_night": "Nocturnal wet-bulb",
#|     "eng2_wetbulb_hot": "Warm-hour wet-bulb",
#|     "eng2_hcho_no2_ratio": "Formaldehyde to NO$_2$ ratio",
#|     "pm25_mean": "Mean fine particulate matter",
#|     "prism_tmin_min": "Minimum daily low temperature",
#|     "cams_fot_formaldehyde_above75th_percentile": "Formaldehyde, time above 75th pct.",
#|     "usda_cropland_usda_corn_pct": "Cropland in corn",
#| }
#| NOCTURNAL = ("prism_tdmean_min", "eng2_wetbulb_night", "eng_night_cooling_eff")
#|
#| # Chemical families of the atmospheric-composition columns. The first rule a
#| # column name matches decides its family, and every composition column falls
#| # in a named family. build_paper_constants.py imports these same rules, so the
#| # shares printed in the text are the shares drawn in Figure 5A.
#| FAMILY_RULES = [
#|     ("Oxidised VOC", ("formaldehyde", "hydrogen_peroxide")),
#|     ("Alkanes", ("propane", "ethane")),
#|     ("Sulphur species", ("sulphur_dioxide", "sulphate")),
#|     ("Reactive nitrogen", ("nitrogen_dioxide", "nitrogen_monoxide", "nitric_acid",
#|                            "peroxyacetyl")),
#|     ("Oxidants", ("ozone", "hydroxyl")),
#|     ("Biogenic VOC", ("isoprene",)),
#|     ("Carbon monoxide", ("carbon_monoxide", "carbonmonoxide")),
#|     ("Carbonaceous aerosol", ("black_carbon", "organic_matter")),
#|     ("Dust and sea salt", ("dust", "sea_salt")),
#|     ("Particulate matter", ("pm1", "pm$1$", "pm2_5", "pm10", "total_aod")),
#| ]
#|
#|
#| def cams_family(feature):
#|     for fam_, keys in FAMILY_RULES:
#|         if any(k in feature for k in keys):
#|             return fam_
#|     return "Other composition"
#|
#|
#| # ===========================================================================
#| #  FIGURE 2  The model is worth interrogating
#| # ===========================================================================
#| def fig02():
#|     base = js("baselines.json")
#|     hold = js("spatial_holdout.json")
#|     rank = js("rank_stability.json")
#|     null = js("null_skill.json")
#|     br = tab("bracket_error.csv")
#|
#|     fig = plt.figure(figsize=(W, 4.6))
#|     gs = fig.add_gridspec(2, 2, left=0.30, right=0.98, top=0.93, bottom=0.10,
#|                           hspace=0.55, wspace=1.15)
#|
#|     # A  baselines on identical folds, and the state holdout
#|     ax = fig.add_subplot(gs[0, 0])
#|     rows = [("Training mean", base["naive_mean"]["r2"], GREY),
#|             ("Training mean by year", base["naive_year_mean"]["r2"], GREY),
#|             ("Night-time lights only", base["ntl_only"]["r2"], GREY),
#|             ("Latitude, longitude, year", base["coords_only"]["r2"], GREY),
#|             ("Four nearest counties", base["knn4_spatial"]["r2"], GREY),
#|             ("This model, states held out", hold["r2"], SKY),
#|             ("This model", base["satellite_model"]["r2"], BLUE)]
#|     y = np.arange(len(rows))[::-1]
#|     for yi, (lab, v, c) in zip(y, rows):
#|         ax.barh(yi, max(v, 0), color=c, height=0.62)
#|         ax.text(max(v, 0) + 0.015, yi, f"{v:.3f}".replace("-0.000", "0.000"),
#|                 va="center", fontsize=FS_TICK)
#|     ax.set_yticks(y, [r[0] for r in rows])
#|     ax.set_xlim(0, 0.85)
#|     ax.set_xlabel("Out-of-fold $R^2$ (dimensionless)")
#|     letter(ax, "A")
#|
#|     # B  learning curve
#|     ax = fig.add_subplot(gs[0, 1])
#|     lc = rank["learning_curve"]
#|     pts = sorted([v for v in lc.values() if isinstance(v, dict)
#|                   and "county_share" in v], key=lambda v: v["county_share"])
#|     xs = [100 * p["county_share"] for p in pts]
#|     ax.plot(xs, [p["r2"] for p in pts], "-o", color=BLUE, ms=4)
#|     ax.set_xlabel("Training counties (%)")
#|     ax.set_ylabel("Out-of-fold $R^2$\n(dimensionless)")
#|     ax.set_xlim(0, 105)
#|     ax.set_ylim(0.4, 0.7)
#|     letter(ax, "B")
#|
#|     # C  error by outcome bracket
#|     ax = fig.add_subplot(gs[1, 0])
#|     lab = br["bracket"].astype(str).tolist()
#|     yy = np.arange(len(lab))[::-1]
#|     ax.barh(yy, br["mae"], color=BLUE, height=0.62)
#|     for yi, v, n in zip(yy, br["mae"], br["n"]):
#|         ax.text(v + 0.1, yi, f"{v:.2f}  (n = {int(n):,})", va="center",
#|                 fontsize=FS_TICK)
#|     ax.set_yticks(yy, [f"{s} years" for s in lab])
#|     ax.set_xlim(0, 11.5)
#|     ax.set_xlabel("Mean absolute error (years)")
#|     ax.set_ylabel("Observed life expectancy")
#|     letter(ax, "C")
#|
#|     # D  where the explained variance lives
#|     ax = fig.add_subplot(gs[1, 1])
#|     items = [("Observed", base["satellite_model"]["r2"], BLUE),
#|              ("Years shuffled\nwithin county", null["within_county"]["r2"], SKY),
#|              ("Series swapped\nbetween counties", null["between_county"]["r2"], GREY),
#|              ("Fully shuffled", null["full"]["r2"], GREY)]
#|     yy = np.arange(len(items))[::-1]
#|     for yi, (lab_, v, c) in zip(yy, items):
#|         ax.barh(yi, v, color=c, height=0.62)
#|         ax.text(max(v, 0) + 0.02, yi, f"{v:+.3f}", va="center",
#|                 fontsize=FS_TICK)
#|     ax.axvline(0, color="#555555", lw=0.6)
#|     ax.set_yticks(yy, [i[0] for i in items])
#|     ax.set_xlim(-0.1, 0.95)
#|     ax.set_xlabel("Out-of-fold $R^2$\n(dimensionless)")
#|     letter(ax, "D")
#|     save(fig, "Figure02_performance")
#|
#|
#| # ===========================================================================
#| #  FIGURE 3  Three fields, several descriptions
#| # ===========================================================================
#| def fig03():
#|     rk = pd.read_csv(C.RANKING_A).sort_values("rank")
#|     val = dict(zip(rk["feature"], rk["mean_abs_shap"]))
#|     rnk = dict(zip(rk["feature"], rk["rank"]))
#|
#|     fig = plt.figure(figsize=(W, 5.0))
#|     gs = fig.add_gridspec(2, 2, left=0.22, right=0.98, top=0.94, bottom=0.10,
#|                           hspace=0.62, wspace=0.95)
#|
#|     def bars(ax, items):
#|         y = np.arange(len(items))[::-1]
#|         for yi, (lab, f, c) in zip(y, items):
#|             ax.barh(yi, val[f], color=c, height=0.6)
#|             ax.text(val[f] * 1.15, yi, f"rank {int(rnk[f])}", va="center",
#|                     fontsize=FS_TICK)
#|         ax.set_yticks(y, [i[0] for i in items])
#|         ax.set_xscale("log")
#|         ax.set_xlim(1e-3, 3.0)
#|         ax.set_xlabel("Mean |SHAP| (years)")
#|
#|     ax = fig.add_subplot(gs[0, 0])
#|     bars(ax, [("Time above P75", "cams_fot_formaldehyde_above75th_percentile", ORANGE),
#|               ("Time above P90", "cams_fot_formaldehyde_above_90th_percentile", ORANGE),
#|               ("Annual mean", "cams_total_column_formaldehyde_x1e9", BLUE)])
#|     letter(ax, "A")
#|
#|     ax = fig.add_subplot(gs[0, 1])
#|     bars(ax, [("Tail form", C.NCE_TAIL, ORANGE), ("Mean form", C.NCE_MEAN, BLUE)])
#|     letter(ax, "B")
#|
#|     # C  seven summaries of one dew-point field over the cells of a county
#|     ax = fig.add_subplot(gs[1, 0])
#|     names = {"min": "Minimum", "stddev": "Variability", "p10": "10th percentile",
#|              "p90": "90th percentile", "p50": "Median", "mean": "Mean",
#|              "max": "Maximum"}
#|     dp = rk[rk["feature"].str.startswith("prism_tdmean_")].sort_values("rank")
#|     items = [(names[f.split("prism_tdmean_")[1]], f,
#|               BLUE if f.endswith(("_mean", "_p50")) else ORANGE)
#|              for f in dp["feature"]]
#|     bars(ax, items)
#|     letter(ax, "C")
#|
#|     # D  where the descriptions sit in the ranking of all admitted variables;
#|     # labels stand in one column at the right, joined to their points by
#|     # leaders that never cross, because the points and labels share an order
#|     ax = fig.add_subplot(gs[1, 1])
#|     ax.plot(rk["rank"], rk["mean_abs_shap"], color=GREY, lw=1.0)
#|     marks = (("cams_fot_formaldehyde_above75th_percentile", ORANGE, "HCHO, time above P75"),
#|              (C.NCE_TAIL, ORANGE, "NCE, tail form"),
#|              ("prism_tdmean_min", ORANGE, "Dew point, minimum"),
#|              (C.NCE_MEAN, BLUE, "NCE, mean form"),
#|              ("cams_total_column_formaldehyde_x1e9", BLUE, "HCHO, annual mean"),
#|              ("prism_tdmean_mean", BLUE, "Dew point, mean"))
#|     ys = (0.55, 0.20, 0.075, 0.028, 0.0105, 0.0003)
#|     xs = (250, 250, 250, 250, 250, 160)
#|     for (f, c, lab), yt, xt in zip(marks, ys, xs):
#|         ax.plot(rnk[f], val[f], "o", color=c, ms=5, mec="white", mew=0.6,
#|                 zorder=3)
#|         ax.annotate(lab, (rnk[f], val[f]), xytext=(xt, yt), fontsize=FS_TICK,
#|                     va="center", arrowprops=dict(arrowstyle="-",
#|                     color="#777777", lw=0.5))
#|     ax.set_yscale("log")
#|     ax.set_ylim(1e-4, 1)
#|     ax.set_xlim(0, len(rk) + 5)
#|     ax.set_xlabel(f"Rank among {len(rk)} admitted variables")
#|     ax.set_ylabel("Mean |SHAP| (years)")
#|     letter(ax, "D")
#|     save(fig, "Figure03_three_fields")
#|
#|
#| # ===========================================================================
#| #  FIGURE 4  Removing one description does not hand its credit to the other
#| # ===========================================================================
#| def fig04():
#|     r = js("nce_refit.json")["conditions"]
#|     full, dp, dm = r["FULL"], r["DROP_P"], r["DROP_M"]
#|     fig, axs = plt.subplots(1, 3, figsize=(W, 2.4),
#|                             gridspec_kw=dict(width_ratios=[1, 0.85, 1.35]))
#|     fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.24,
#|                         wspace=0.85)
#|     ticks = ["Both", "No tail"]
#|
#|     # A  attribution of the mean form, with and without its twin
#|     ax = axs[0]
#|     for i, v in enumerate([full["mean_shap"], dp["mean_shap"]]):
#|         ax.bar(i, v, color=BLUE, width=0.55)
#|         ax.text(i, v * 1.18, f"{v:.3f}", ha="center", fontsize=FS_TICK)
#|     ax.axhline(full["pct_shap"], color=ORANGE, ls="--", lw=0.9)
#|     ax.text(-0.42, full["pct_shap"] * 1.15, "tail form, both present",
#|             fontsize=6.6, va="bottom", ha="left", color=INK)
#|     ax.set_yscale("log")
#|     ax.set_ylim(5e-3, 0.8)
#|     ax.set_xlim(-0.5, 1.5)
#|     ax.set_xticks([0, 1], ticks, fontsize=6.8)
#|     ax.set_ylabel("Mean form,\nmean |SHAP| (years)")
#|     letter(ax, "A")
#|
#|     # B  its rank among all admitted columns
#|     ax = axs[1]
#|     rk = [full["mean_rank"], dp["mean_rank"]]
#|     ax.plot([0, 1], rk, "-", color=GREY, lw=0.9, zorder=1)
#|     ax.plot([0, 1], rk, "o", color=BLUE, ms=6, zorder=2)
#|     for i, v in enumerate(rk):
#|         ax.text(i + 0.12, v, f"{int(v)}", va="center", ha="left",
#|                 fontsize=FS_TICK)
#|     ax.set_xticks([0, 1], ticks, fontsize=6.8)
#|     ax.set_xlim(-0.6, 1.7)
#|     ax.set_ylim(150, 1)
#|     ax.set_yticks([1, 50, 100, 150])
#|     ax.set_ylabel("Mean form, rank\n(1 = most important)")
#|     letter(ax, "B")
#|
#|     # C  predictive skill in the three conditions
#|     ax = axs[2]
#|     items = [("Both", full["r2"]), ("No tail", dp["r2"]),
#|              ("No mean", dm["r2"])]
#|     for i, (lab, v) in enumerate(items):
#|         ax.plot(i, v, "o", color=BLUE, ms=5)
#|         ax.text(i, v + 0.0010, f"{v:.4f}", ha="center", fontsize=6.6)
#|     ax.set_xticks(range(3), [i[0] for i in items], fontsize=6.8)
#|     ax.set_xlim(-0.6, 2.6)
#|     ax.set_ylim(0.662, 0.674)
#|     ax.set_ylabel("Out-of-fold $R^2$\n(dimensionless)")
#|     letter(ax, "C")
#|     save(fig, "Figure04_drop_one_refit")
#|
#| # ===========================================================================
#| #  FIGURE 5  Where the composition signal sits, and atmospheric lifetime
#| # ===========================================================================
#| def fig05():
#|     rk = pd.read_csv(C.RANKING_A)
#|     comp = rk[rk["feature"].str.startswith("cams")].copy()
#|     comp["family"] = comp["feature"].map(cams_family)
#|     tot = comp["mean_abs_shap"].sum()
#|     fam = (comp.groupby("family")["mean_abs_shap"].agg(["sum", "size"])
#|            .sort_values("sum"))
#|     lt = js("lifetime_literature_test.json")
#|     sp = pd.DataFrame(lt["species_table"]).sort_values("lifetime_days")
#|     pairs = tab("duration_vs_concentration.csv")
#|     pairs["species"] = pairs["species"].str.lower().str.strip()
#|     cut = float(lt["published_cutoff"])
#|     nice = {"pm2 5": "PM$_{2.5}$", "pm1": "PM$_1$",
#|             "nitrogen monoxide": "nitric oxide"}
#|
#|     fig = plt.figure(figsize=(W, 6.0))
#|     gs = fig.add_gridspec(2, 2, left=0.30, right=0.98, top=0.95, bottom=0.18,
#|                           hspace=0.58, wspace=1.05, height_ratios=[0.85, 1.2])
#|
#|     # A  share of composition attribution by chemical family
#|     ax = fig.add_subplot(gs[0, 0])
#|     y = np.arange(len(fam))
#|     cols = [ORANGE if f_ == "Oxidised VOC" else BLUE for f_ in fam.index]
#|     share = 100 * fam["sum"] / tot
#|     ax.barh(y, share, color=cols, height=0.62)
#|     for yi, (v, k) in zip(y, zip(share, fam["size"])):
#|         ax.text(v + 1.5, yi, f"{v:.1f} (n = {k})", va="center", fontsize=6.8)
#|     ax.set_yticks(y, fam.index)
#|     ax.set_xlim(0, 85)
#|     ax.set_xlabel("Share of composition\nattribution (%)")
#|     letter(ax, "A")
#|
#|     # B  each species' time above the 75th against the 90th percentile
#|     ax = fig.add_subplot(gs[0, 1])
#|     du = comp[comp["feature"].str.contains("_fot_")
#|               & comp["feature"].str.contains("percentile")].copy()
#|     du["pct"] = np.where(du["feature"].str.contains("75th"), 75, 90)
#|     du["species"] = (du["feature"].str.replace("cams_fot_", "", regex=False)
#|                      .str.replace(r"_?above_?\d+th_percentile", "", regex=True)
#|                      .str.replace("$1$", "1", regex=False).str.replace("_", " "))
#|     piv = (du.pivot_table(index="species", columns="pct", values="mean_abs_shap")
#|            .dropna().sort_values(75))
#|     yy = np.arange(len(piv))
#|     for yi, (_, r) in zip(yy, piv.iterrows()):
#|         ax.plot([r[75], r[90]], [yi, yi], color=LIGHT, lw=1.4, zorder=1)
#|     ax.plot(piv[75], yy, "o", color=INK, ms=5, zorder=2)
#|     ax.plot(piv[90], yy, "s", color=GREY, mfc="white", ms=4.5, zorder=2)
#|     ax.set_xscale("log")
#|     ax.set_xlim(5e-4, 1.0)
#|     ax.set_yticks(yy, [nice.get(s_, s_) for s_ in piv.index])
#|     ax.set_xlabel("Mean |SHAP| of the\nduration (years)")
#|     letter(ax, "B")
#|
#|     # C  per species, the duration to concentration ratio, ordered by lifetime
#|     ax = fig.add_subplot(gs[1, 0])
#|     y = np.arange(len(sp))[::-1]
#|     for yi, (_, r) in zip(y, sp.iterrows()):
#|         c = ORANGE if r["lifetime_days"] < cut else BLUE
#|         ax.barh(yi, r["median_ratio"], color=c, height=0.6, left=0)
#|         pr = pairs[pairs["species"] == r["species"]]["ratio"].to_numpy()
#|         ax.plot(pr, np.full(len(pr), yi), "|", color=INK, ms=6, mew=0.8)
#|     ax.axvline(1, color=GREY, ls="--", lw=0.8)
#|
#|     def days(d):
#|         if d >= 1:
#|             return f"{d:g} d"
#|         if d >= 0.01:
#|             return f"{d:.2g} d"
#|         e = int(np.floor(np.log10(d)))
#|         m = d / 10 ** e
#|         return (f"$10^{{{e}}}$ d" if abs(m - 1) < 0.05
#|                 else f"${m:.0f}\\times10^{{{e}}}$ d")
#|     ax.set_yticks(y, [f"{nice.get(s, s)} ({days(d)})" for s, d in
#|                       zip(sp["species"], sp["lifetime_days"])])
#|     ax.set_xscale("log")
#|     ax.set_xlim(0.1, 120)
#|     ax.set_xlabel("Duration / concentration\nmean |SHAP| ratio")
#|     letter(ax, "C")
#|
#|     # D  the ratio against lifetime, species as the unit
#|     ax = fig.add_subplot(gs[1, 1])
#|     for _, r in sp.iterrows():
#|         c = ORANGE if r["lifetime_days"] < cut else BLUE
#|         ax.plot(r["lifetime_days"], r["median_ratio"], "o", color=c, ms=4.5)
#|     ax.axhline(1, color=GREY, ls="--", lw=0.8)
#|     ax.set_xscale("log")
#|     ax.set_yscale("log")
#|     ax.set_xlim(5e-6, 200)
#|     ax.set_xticks([1e-5, 1e-3, 1e-1, 1e1])
#|     ax.set_ylim(0.15, 1000)
#|     ax.set_xlabel("Atmospheric lifetime (days)")
#|     ax.set_ylabel("Ratio (dimensionless)")
#|     rf = lt["rank_free_test"]
#|     rho = f"{rf['spearman_rho']:+.2f}".replace("-", "\u2212")
#|     ax.text(0.04, 0.96, f"Spearman {rho}\n"
#|             f"permutation p = {rf['permutation_p']:.3f}\n"
#|             f"{lt['n_species']} species",
#|             transform=ax.transAxes, ha="left", va="top", fontsize=6.6)
#|     letter(ax, "D")
#|
#|     handles = [Line2D([], [], marker="o", ls="", color=INK,
#|                       label="B: time above the 75th percentile"),
#|                Line2D([], [], marker="s", ls="", color=GREY, mfc="white",
#|                       label="B: time above the 90th percentile"),
#|                Patch(color=ORANGE, label=f"C, D: lifetime < {cut:g} days"),
#|                Patch(color=BLUE, label=f"C, D: lifetime ≥ {cut:g} days"),
#|                Line2D([], [], marker="|", ls="", color=INK,
#|                       label="C: ratio at each percentile")]
#|     fig.legend(handles=handles, loc="lower center", fontsize=6.8, ncol=2,
#|                handletextpad=0.3, columnspacing=1.4, bbox_to_anchor=(0.55, 0.0))
#|     save(fig, "Figure05_composition_lifetime")
#|
#|
#| # ===========================================================================
#| #  FIGURE 6  Turning points of the attribution curves
#| # ===========================================================================
#| CURVE_XLABEL = {
#|     "prism_tdmean_min": "Minimum dew point\n(°C)",
#|     "eng2_wetbulb_night": "Night-time wet-bulb\ntemperature (°C)",
#|     "eng_night_cooling_eff": "Night cooling, tail\nform (dimensionless)",
#|     "cams_fot_formaldehyde_above75th_percentile":
#|         "Formaldehyde, time above\n75th percentile (% of year)",
#|     "eng2_hcho_no2_ratio": "Formaldehyde to NO$_2$\nratio (dimensionless)",
#| }
#|
#|
#| def _attribution_curve(ax, dep, tp, f):
#|     """Median SHAP in 45 equal-count bins between the 1st and 99th
#|     percentiles, the located turning point and its 95% bootstrap interval."""
#|     x = dep[f].to_numpy(float)
#|     v = dep["shap_" + f].to_numpy(float)
#|     ok = np.isfinite(x) & np.isfinite(v)
#|     x, v = x[ok], v[ok]
#|     lo, hi = np.percentile(x, [1, 99])
#|     qs = np.quantile(x[(x >= lo) & (x <= hi)], np.linspace(0, 1, 46))
#|     bx, by = [], []
#|     for a, b in zip(qs[:-1], qs[1:]):
#|         s = (x >= a) & (x < b)
#|         if s.sum() > 2:
#|             bx.append(np.median(x[s]))
#|             by.append(np.median(v[s]))
#|     bx, by = np.array(bx), np.array(by)
#|     ax.plot(bx, by, "o", color=GREY, ms=2.2)
#|     r = tp.loc[f]
#|     ax.axvspan(r["ci_lo"], r["ci_hi"], color=PINK, alpha=0.35, lw=0)
#|     ax.axvline(r["point_estimate"], color=PINK, lw=1.0)
#|     ax.axhline(0, color="#555555", lw=0.6)
#|     ax.set_xlabel(CURVE_XLABEL[f], fontsize=7.8)
#|     unit = NAMES[f][1]
#|     u = "" if unit == "dimensionless" else f" {unit}"
#|     left_high = np.nanmedian(by[: max(len(by) // 3, 1)]) > 0
#|     ax.text(0.04, 0.05 if left_high else 0.95,
#|             f"{r['point_estimate']:.3g}{u}", transform=ax.transAxes,
#|             va="bottom" if left_high else "top", fontsize=FS_TICK, color=INK)
#|
#|
#| def fig06():
#|     dep, _ = C.dependence_table()
#|     tp = tab("turning_points.csv").set_index("feature")
#|     st = C.load_table("A_temp_strata")
#|     feats = list(NOCTURNAL) + ["cams_fot_formaldehyde_above75th_percentile",
#|                                "eng2_hcho_no2_ratio"]
#|     fig = plt.figure(figsize=(W, 5.0))
#|     gs = fig.add_gridspec(2, 3, left=0.11, right=0.98, top=0.94, bottom=0.15,
#|                           hspace=0.95, wspace=0.68)
#|     for i, f in enumerate(feats):
#|         ax = fig.add_subplot(gs[i // 3, i % 3])
#|         _attribution_curve(ax, dep, tp, f)
#|         if i % 3 == 0:
#|             ax.set_ylabel("SHAP (years)")
#|         letter(ax, "ABCDE"[i])
#|
#|     # F  the dew-point crossing within quartiles of minimum temperature
#|     ax = fig.add_subplot(gs[1, 2])
#|     pe = tp.loc["prism_tdmean_min", "point_estimate"]
#|     for i, r in st.reset_index(drop=True).iterrows():
#|         if np.isfinite(r["zero_cross"]):
#|             ax.plot(i, r["zero_cross"], "o", color=PINK, ms=5)
#|             ax.text(i, r["zero_cross"] + 0.3, f"{r['zero_cross']:.2f}",
#|                     ha="center", fontsize=6.6)
#|         else:
#|             ax.text(i, pe - 1.2, "none", ha="center", va="center",
#|                     fontsize=6.6, color=INK)
#|     ax.axhline(pe, color=GREY, ls="--", lw=0.8)
#|     ax.set_xticks(range(len(st)), [s.split()[0] for s in st["stratum"]])
#|     ax.set_xlim(-0.6, len(st) - 0.4)
#|     ax.set_ylim(pe - 2, pe + 2)
#|     ax.set_xlabel("Quartile of minimum\ntemperature (coldest Q1)", fontsize=7.8)
#|     ax.set_ylabel("Crossing (°C)")
#|     letter(ax, "F")
#|     save(fig, "Figure06_turning_point_curves")
#|
#|
#| # ===========================================================================
#| #  FIGURE 7  The turning points under bootstrap, noise and two regional splits
#| # ===========================================================================
#| def fig07():
#|     tp = tab("turning_points.csv").set_index("feature")
#|     nc = js("null_crossing.json")["per_variable"]
#|     pt = tab("perturbation_thresholds.csv")
#|     med = js("threshold_regional_invariance_all.json")
#|     mer = js("threshold_regional_invariance_100W_all.json")
#|     order = tp.sort_values("width_frac").index.tolist()
#|
#|     fig = plt.figure(figsize=(W, 5.4))
#|     top = fig.add_gridspec(1, 3, left=0.34, right=0.98, top=0.93, bottom=0.53,
#|                            wspace=0.14)
#|     bot = fig.add_gridspec(1, 1, left=0.34, right=0.98, top=0.27, bottom=0.08)
#|     y = np.arange(len(order))[::-1]
#|
#|     # A  interval width
#|     ax = fig.add_subplot(top[0, 0])
#|     for yi, f in zip(y, order):
#|         r = tp.loc[f]
#|         ax.barh(yi, 100 * r["width_frac"], color=GREEN if bool(r["stable"]) else RED,
#|                 height=0.6)
#|     ax.axvline(40, color=GREY, ls="--", lw=0.8)
#|     ax.set_xscale("log")
#|     ax.set_xlim(0.1, 100)
#|     ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
#|     ax.set_yticks(y, [SHORT.get(f, f) for f in order])
#|     ax.set_xlabel("95% interval width\n(% of range)")
#|     letter(ax, "A")
#|
#|     # B  the same procedure on permuted attribution
#|     ax = fig.add_subplot(top[0, 1])
#|     for yi, f in zip(y, order):
#|         rec = nc.get(f, {})
#|         ax.barh(yi, rec.get("permuted_crossing_found", 0), color=LIGHT, height=0.6)
#|         ax.plot(rec.get("permuted_gate_passed", 0), yi, "o", color=RED, ms=4,
#|                 zorder=3)
#|     ax.set_yticks(y, [])
#|     nd = next(iter(nc.values()))["n_draws"]
#|     ax.set_xlim(0, nd)
#|     ax.set_xlabel(f"Permutations\n(of {nd})")
#|     letter(ax, "B")
#|
#|     # C  west minus east, at the median longitude and at the 100th meridian
#|     ax = fig.add_subplot(top[0, 2])
#|     FLOOR, CEIL = 0.05, 60.0
#|     for yi, f in zip(y, order):
#|         c = PINK if f in NOCTURNAL else GREY
#|         a = med["results"].get(f, {}).get("delta_pct_of_range")
#|         b = mer["results"].get(f, {}).get("delta_pct_of_range")
#|         for v, dy, mk, face in ((a, 0.16, "o", c), (b, -0.16, "^", "white")):
#|             if v is None:
#|                 ax.plot(CEIL * 0.75, yi + dy, "x", color=c, ms=4.5, zorder=3)
#|                 continue
#|             ax.plot(min(max(abs(float(v)), FLOOR), CEIL), yi + dy, mk, color=c,
#|                     mfc=face, ms=4.5, zorder=3)
#|     ax.set_yticks(y, [])
#|     ax.set_xscale("log")
#|     ax.set_xlim(FLOOR * 0.8, CEIL)
#|     ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
#|     ax.set_xlabel("|West minus east|\n(% of range)")
#|     letter(ax, "C")
#|
#|     handles = [Patch(color=GREEN, label="stable"),
#|                Patch(color=RED, label="not stable"),
#|                Patch(color=LIGHT, label="crossing located in a permutation"),
#|                Line2D([], [], marker="o", ls="", color=RED,
#|                       label="permutation passed the criterion"),
#|                Line2D([], [], marker="o", ls="", color=GREY,
#|                       label=f"split at {abs(med['cut_degrees_east']):.1f}° W"),
#|                Line2D([], [], marker="^", ls="", color=GREY, mfc="white",
#|                       label="split at 100° W"),
#|                Line2D([], [], marker="x", ls="", color=GREY,
#|                       label="no crossing in one half"),
#|                Line2D([], [], marker="s", ls="", color=PINK,
#|                       label="night-time moisture and cooling")]
#|     fig.legend(handles=handles, loc="center", fontsize=6.6, ncol=2,
#|                handletextpad=0.3, columnspacing=1.2,
#|                bbox_to_anchor=(0.62, 0.375))
#|
#|     # D  movement under predictor perturbation
#|     ax = fig.add_subplot(bot[0, 0])
#|     for f, g in pt.groupby("feature"):
#|         g = g.sort_values("magnitude")
#|         c = PINK if f in NOCTURNAL else LIGHT
#|         ax.plot(100 * g["magnitude"], g["shift_over_interval_width"].abs(),
#|                 "-", color=c, lw=0.9, zorder=2 if f in NOCTURNAL else 1)
#|     mm = pt.groupby("magnitude")["shift_over_interval_width"].apply(
#|         lambda s: s.abs().median())
#|     ax.plot(100 * mm.index, mm.values, "-o", color=INK, ms=4, lw=1.4, zorder=3)
#|     ax.axhline(1, color=GREY, ls="--", lw=0.8)
#|     ax.set_xscale("log")
#|     ax.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
#|     ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
#|     ax.set_yticks([0, 0.5, 1, 1.5, 2])
#|     ax.set_xlabel("Gaussian noise added to the predictor (% of its standard deviation)")
#|     ax.set_ylabel("Shift (interval\nwidths)")
#|     handles = [Line2D([], [], color=INK, marker="o",
#|                       label=f"median, {len(order)} variables"),
#|                Line2D([], [], color=PINK, label="night-time moisture and cooling"),
#|                Line2D([], [], color=LIGHT, label="other variables")]
#|     ax.legend(handles=handles, loc="upper left", fontsize=6.6)
#|     letter(ax, "D")
#|     save(fig, "Figure07_turning_point_stability")
#|
#|
#| # ===========================================================================
#| #  FIGURE 8  Where a physical model is blind
#| # ===========================================================================
#| def fig08():
#|     import geopandas as gpd
#|     from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
#|     cty = tab("residual_social_determinants_counties.csv")
#|     res = js("residual_social_determinants.json")
#|     g = gpd.read_file(os.path.join(HERE, "conus_counties_5070.gpkg"), layer="county")
#|     g["fips"] = g["fips"].astype(int)
#|     m = g.merge(cty, left_on="fips", right_on=C.GROUP, how="left")
#|
#|     fig = plt.figure(figsize=(W, 5.4))
#|     gs = fig.add_gridspec(2, 2, left=0.05, right=0.97, top=0.97, bottom=0.10,
#|                           hspace=0.28, wspace=0.55, height_ratios=[1.45, 1])
#|     ax = fig.add_subplot(gs[0, :])
#|     div = LinearSegmentedColormap.from_list(
#|         "div", ["#8A3A00", RED, "#F0B27A", "#F2F2F2", "#8FC7E8", BLUE, "#00355A"])
#|     norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
#|     m[m["resid"].isna()].plot(ax=ax, color="#E6E6E6", linewidth=0)
#|     m[m["resid"].notna()].plot(ax=ax, column="resid", cmap=div, norm=norm,
#|                                linewidth=0)
#|     # state outlines, dissolved from the same county polygons (state = FIPS // 1000)
#|     states = g.assign(state=g["fips"] // 1000).dissolve(by="state")
#|     states.boundary.plot(ax=ax, color="#3A3A3A", linewidth=0.35)
#|     ax.set_axis_off()
#|     sm = plt.cm.ScalarMappable(norm=norm, cmap=div)
#|     cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045,
#|                       pad=0.01, aspect=40, shrink=0.6)
#|     cb.set_label("County mean residual, observed minus predicted (years)",
#|                  fontsize=FS_TICK)
#|     cb.ax.tick_params(labelsize=6.8)
#|     letter(ax, "A")
#|
#|     ax = fig.add_subplot(gs[1, 0])
#|     items = [(f"{res['n_sdoh_columns']} socioeconomic\ncolumns, never\nin the model",
#|               res["oof_r2_residual_from_sdoh"], GREY),
#|              (f"{res['n_env_columns']} environmental\ncolumns, all\nin the model",
#|               res["oof_r2_residual_from_environment"], BLUE)]
#|     for i, (lab, v, c) in enumerate(items):
#|         ax.bar(i, v, color=c, width=0.55)
#|         ax.text(i, v + (0.02 if v >= 0 else -0.07), f"{v:+.2f}", ha="center",
#|                 fontsize=FS_TICK)
#|     ax.axhline(0, color="#555555", lw=0.6)
#|     ax.set_xticks([0, 1], [i[0] for i in items], fontsize=6.6)
#|     ax.set_ylim(-0.3, 0.6)
#|     ax.set_ylabel("Out-of-fold $R^2$ for the\nresidual (dimensionless)")
#|     letter(ax, "B")
#|
#|     ax = fig.add_subplot(gs[1, 1])
#|     ind = sorted(res["indicators"], key=lambda r: r["rho"])
#|     yy = np.arange(len(ind))
#|     short = {"Population below the poverty line (%)": "Poverty",
#|              "Median household income (USD)": "Median income",
#|              "Unemployment rate (%)": "Unemployment",
#|              "Without health insurance (%)": "Uninsured",
#|              "Bachelor's degree or higher (%)": "Bachelor's degree",
#|              "Disability rate (%)": "Disability"}
#|     for yi, r in zip(yy, ind):
#|         c = RED if r["rho"] < 0 else BLUE
#|         ax.barh(yi, r["rho"], color=c, height=0.6)
#|     ax.axvline(0, color="#555555", lw=0.6)
#|     ax.set_yticks(yy, [short.get(r["label"], r["label"]) for r in ind])
#|     ax.set_xlim(-0.45, 0.45)
#|     ax.set_xlabel("Spearman correlation\nwith the residual")
#|     letter(ax, "C")
#|     save(fig, "Figure08_failure_geography")
#|
#|
#| FIGS = {2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06, 7: fig07,
#|         8: fig08}
#|
#|
#| def main():
#|     global DPI, OUT
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--only", default="")
#|     ap.add_argument("--dpi", type=int, default=600)
#|     ap.add_argument("--outdir", default="")
#|     a = ap.parse_args()
#|     DPI = a.dpi
#|     if a.outdir:
#|         from pathlib import Path
#|         OUT = Path(a.outdir).resolve()
#|     want = ([int(x) for x in a.only.split(",") if x.strip()] if a.only
#|             else sorted(FIGS))
#|     failed = []
#|     for k in want:
#|         print(f"Figure {k}")
#|         try:
#|             FIGS[k]()
#|         except Exception as e:                         # report, keep going
#|             import traceback
#|             traceback.print_exc()
#|             failed.append(k)
#|     if failed:
#|         raise SystemExit(f"figures that failed: {failed}")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END analysis/paper_figures.py
#@@ FILE pipeline/engineered_features_v2.py
#| #!/usr/bin/env python3
#| # =============================================================================
#| #  engineered_features_v2.py
#| #  Physically motivated engineered layer for the 1,318-column superset.
#| #
#| #  WHAT IS NEW RELATIVE TO v1
#| #  ---------------------------------------------------------------------------
#| #  * Column names in the superset are lower-case and differently prefixed
#| #    (prism_tdmean_min, era5_2m_dew_point_temperature, cams_formaldehyde_x1e9),
#| #    so the whole layer is rewritten against the real schema.
#| #  * ERA5 already supplies wet-bulb temperature, relative and specific humidity,
#| #    and total column water vapour. We do NOT duplicate those. Instead we build
#| #    a PRISM-based wet-bulb that spans 2000-2023 (ERA5 stops in 2019), which
#| #    gives the composite index full coverage in both study windows and lets the
#| #    two independent wet-bulb estimates be cross-checked against each other.
#| #  * CAMS composition enables photochemistry indices that the atmospheric
#| #    literature treats as standard, above all the formaldehyde-to-nitrogen
#| #    dioxide ratio, which diagnoses whether ozone formation in a place is
#| #    limited by volatile organic compounds or by nitrogen oxides.
#| #
#| #  DESIGN RULES (unchanged)
#| #  * Row-local and deterministic: every feature is a function of same-row
#| #    inputs, so adding this layer before cross-validation introduces no leakage.
#| #  * Defensive: a feature is built only if all of its inputs exist, so the same
#| #    module runs unchanged on either study's column subset.
#| #  * Self-reporting: returns a coverage table so a reduced-coverage engineered
#| #    feature can never again be ranked without anyone noticing.
#| # =============================================================================
#|
#| from __future__ import annotations
#| import numpy as np
#| import pandas as pd
#|
#| TAG = "eng2"
#|
#|
#| # ---------------------------------------------------------------------------
#| #  physics helpers
#| # ---------------------------------------------------------------------------
#| def _c(df, name):
#|     """Column if present, else an all-NaN series (defensive access)."""
#|     return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)
#|
#|
#| def _first(df, *names):
#|     """First present column among alternatives, else all-NaN."""
#|     for n in names:
#|         if n in df.columns:
#|             return df[n]
#|     return pd.Series(np.nan, index=df.index)
#|
#|
#| def sat_vapour_pressure_hpa(T):
#|     """Saturation vapour pressure over water (hPa), Tetens; T in degrees C."""
#|     return 6.1078 * np.exp(17.27 * T / (T + 237.3))
#|
#|
#| def rh_from_T_Td(T, Td):
#|     """Relative humidity (%) from temperature and dew point, both deg C."""
#|     a, b = 17.625, 243.04
#|     return np.clip(100.0 * np.exp((a * Td) / (b + Td)) / np.exp((a * T) / (b + T)),
#|                    1.0, 100.0)
#|
#|
#| def stull_wetbulb(T, RH):
#|     """Wet-bulb temperature (deg C). Stull, J Appl Meteorol Climatol, 2011.
#|     Valid roughly -20 to 50 deg C and 5 to 99 percent relative humidity."""
#|     RH = np.clip(RH, 1.0, 100.0)
#|     return (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
#|             + np.arctan(T + RH) - np.arctan(RH - 1.676331)
#|             + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH) - 4.686035)
#|
#|
#| def wbgt_shade(T, Td):
#|     """Wet-bulb globe temperature, shade approximation (Australian BoM)."""
#|     return 0.567 * T + 0.393 * sat_vapour_pressure_hpa(Td) + 3.94
#|
#|
#| def heat_index_c(T, RH):
#|     """US National Weather Service heat index (Rothfusz), returned in deg C."""
#|     Tf = T * 9 / 5 + 32
#|     HI = (-42.379 + 2.04901523 * Tf + 10.14333127 * RH - 0.22475541 * Tf * RH
#|           - 6.83783e-3 * Tf ** 2 - 5.481717e-2 * RH ** 2
#|           + 1.22874e-3 * Tf ** 2 * RH + 8.5282e-4 * Tf * RH ** 2
#|           - 1.99e-6 * Tf ** 2 * RH ** 2)
#|     HI = np.where(Tf < 80, Tf, HI)
#|     return (HI - 32) * 5 / 9
#|
#|
#| def humidex_c(T, Td):
#|     """Humidex (Masterton and Richardson), deg C."""
#|     e = 6.11 * np.exp(5417.7530 * (1 / 273.16 - 1 / (273.15 + Td)))
#|     return T + 0.5555 * (e - 10.0)
#|
#|
#| def _safe_ratio(num, den, eps=1e-9):
#|     den = pd.Series(den).replace(0, np.nan)
#|     return pd.Series(num) / (den + eps)
#|
#|
#| # ---------------------------------------------------------------------------
#| #  main entry point
#| # ---------------------------------------------------------------------------
#| def add_engineered_features(df: pd.DataFrame, verbose: bool = True):
#|     """Return (augmented_df, coverage_report). Adds features named eng2_*."""
#|     out = df.copy()
#|     new = {}
#|
#|     def add(name, series, needs=()):
#|         """Register a feature only if every named input column exists."""
#|         if needs and not all(n in df.columns for n in needs):
#|             return
#|         new[f"{TAG}_{name}"] = series
#|
#|     # ---- base fields -------------------------------------------------------
#|     # PRISM spans 2000-2023 and is the backbone for full-coverage indices.
#|     t_mean = _c(df, "prism_tmean_mean")
#|     t_max = _c(df, "prism_tmax_mean")
#|     t_min = _c(df, "prism_tmin_mean")
#|     td_mean = _c(df, "prism_tdmean_mean")
#|     td_min = _c(df, "prism_tdmean_min")
#|     ppt = _first(df, "prism_ppt_mean", "prism_ppt_sum_mean")
#|
#|     # ERA5 spans 2003-2019 and supplies an independent thermodynamic view.
#|     e_t = _c(df, "era5_2m_temperature")
#|     e_td = _c(df, "era5_2m_dew_point_temperature")
#|     e_wb = _c(df, "era5_wet_bulb_temperature")
#|     e_q = _c(df, "era5_specific_humidity")
#|     e_tcwv = _c(df, "era5_total_column_water_vapour")
#|     e_sp = _c(df, "era5_surface_pressure")
#|     e_ws = _c(df, "era5_10m_wind_speed")
#|
#|     lst_day = _c(df, "lst_day_1km_mean")
#|     lst_night = _c(df, "lst_night_1km_mean")
#|
#|     # ---- 1. THERMAL COMFORT FROM PRISM (full 2000-2023 coverage) -----------
#|     # These matter because ERA5 stops in 2019; a PRISM-based composite keeps the
#|     # heat-and-humidity question answerable across the whole record.
#|     rh_mean = rh_from_T_Td(t_mean, td_mean)
#|     rh_hot = rh_from_T_Td(t_max, td_mean)
#|     rh_night = rh_from_T_Td(t_min, td_min)
#|
#|     add("wetbulb_mean", stull_wetbulb(t_mean, rh_mean),
#|         needs=("prism_tmean_mean", "prism_tdmean_mean"))
#|     add("wetbulb_hot", stull_wetbulb(t_max, rh_hot),
#|         needs=("prism_tmax_mean", "prism_tdmean_mean"))
#|     add("wetbulb_night", stull_wetbulb(t_min, rh_night),
#|         needs=("prism_tmin_mean", "prism_tdmean_min"))
#|     add("wbgt_shade", wbgt_shade(t_mean, td_mean),
#|         needs=("prism_tmean_mean", "prism_tdmean_mean"))
#|     add("wbgt_hot", wbgt_shade(t_max, td_mean),
#|         needs=("prism_tmax_mean", "prism_tdmean_mean"))
#|     add("heat_index_hot", heat_index_c(t_max, rh_hot),
#|         needs=("prism_tmax_mean", "prism_tdmean_mean"))
#|     add("humidex_hot", humidex_c(t_max, td_mean),
#|         needs=("prism_tmax_mean", "prism_tdmean_mean"))
#|     add("discomfort_index", t_mean - 0.55 * (1 - 0.01 * rh_mean) * (t_mean - 14.5),
#|         needs=("prism_tmean_mean", "prism_tdmean_mean"))
#|     add("rel_humidity_mean", rh_mean, needs=("prism_tmean_mean", "prism_tdmean_mean"))
#|     add("rel_humidity_night", rh_night, needs=("prism_tmin_mean", "prism_tdmean_min"))
#|     add("dewpoint_depression", t_mean - td_mean,
#|         needs=("prism_tmean_mean", "prism_tdmean_mean"))
#|     add("nocturnal_dewpoint_depression", t_min - td_min,
#|         needs=("prism_tmin_mean", "prism_tdmean_min"))
#|     add("diurnal_temp_range", t_max - t_min,
#|         needs=("prism_tmax_mean", "prism_tmin_mean"))
#|     add("humid_heat_excess", np.clip(stull_wetbulb(t_max, rh_hot) - 26.0, 0, None),
#|         needs=("prism_tmax_mean", "prism_tdmean_mean"))
#|     add("cooling_degree_days", np.clip(t_mean - 18.0, 0, None),
#|         needs=("prism_tmean_mean",))
#|     add("heating_degree_days", np.clip(18.0 - t_mean, 0, None),
#|         needs=("prism_tmean_mean",))
#|
#|     e_v = sat_vapour_pressure_hpa(td_mean)
#|     add("vapour_pressure", e_v, needs=("prism_tdmean_mean",))
#|     add("vpd_derived", sat_vapour_pressure_hpa(t_mean) - e_v,
#|         needs=("prism_tmean_mean", "prism_tdmean_mean"))
#|     add("specific_humidity_prism", 0.622 * e_v / (1013.25 - 0.378 * e_v),
#|         needs=("prism_tdmean_mean",))
#|     add("absolute_humidity_prism", (2.16679 * (e_v * 100.0)) / (273.15 + t_mean),
#|         needs=("prism_tmean_mean", "prism_tdmean_mean"))
#|
#|     # ---- 2. ERA5 CROSS-CHECK AND DERIVED THERMODYNAMICS --------------------
#|     # A direct comparison of two independent wet-bulb estimates. Large
#|     # disagreement in a county is a data-quality signal worth having in hand.
#|     add("wetbulb_era5_minus_prism", e_wb - stull_wetbulb(t_mean, rh_mean),
#|         needs=("era5_wet_bulb_temperature", "prism_tmean_mean", "prism_tdmean_mean"))
#|     add("era5_dewpoint_depression", e_t - e_td,
#|         needs=("era5_2m_temperature", "era5_2m_dew_point_temperature"))
#|     add("era5_wetbulb_depression", e_t - e_wb,
#|         needs=("era5_2m_temperature", "era5_wet_bulb_temperature"))
#|     # Column water vapour per unit surface pressure: a moisture-loading index.
#|     add("column_moisture_loading", _safe_ratio(e_tcwv, e_sp),
#|         needs=("era5_total_column_water_vapour", "era5_surface_pressure"))
#|     add("ventilation_index", e_ws * np.clip(e_t - e_td, 0.1, None),
#|         needs=("era5_10m_wind_speed", "era5_2m_temperature",
#|                "era5_2m_dew_point_temperature"))
#|
#|     # ---- 3. CAMS PHOTOCHEMISTRY -------------------------------------------
#|     # The formaldehyde-to-nitrogen-dioxide ratio is the standard satellite
#|     # diagnostic of the ozone-production regime: low values indicate formation
#|     # limited by volatile organic compounds, high values indicate formation
#|     # limited by nitrogen oxides. It is a genuinely different quantity from
#|     # either species alone, which is exactly what a fusion model needs.
#|     hcho = _first(df, "cams_formaldehyde_x1e9", "cams_total_column_formaldehyde_x1e9")
#|     no2 = _first(df, "cams_nitrogen_dioxide_x1e9", "cams_total_column_nitrogen_dioxide_x1e9")
#|     o3 = _first(df, "cams_ozone_x1e9", "cams_total_column_ozone")
#|     co = _first(df, "cams_carbon_monoxide_x1e9", "cams_total_column_carbon_monoxide")
#|     so2 = _first(df, "cams_sulphur_dioxide_x1e9", "cams_total_column_sulphur_dioxide_x1e9")
#|     oh = _first(df, "cams_hydroxyl_radical_x1e9", "cams_total_column_hydroxyl_radical_x1e9")
#|     isop = _first(df, "cams_isoprene_x1e9", "cams_total_column_isoprene_x1e9")
#|     pan = _first(df, "cams_peroxyacetyl_nitrate_x1e9",
#|                  "cams_total_column_peroxyacetyl_nitrate_x1e9")
#|     h2o2 = _first(df, "cams_hydrogen_peroxide_x1e9",
#|                   "cams_total_column_hydrogen_peroxide_x1e9")
#|     bc_aod = _c(df, "cams_black_carbon_aod_at_550_nm")
#|     om_aod = _c(df, "cams_organic_matter_aod_at_550_nm")
#|     su_aod = _c(df, "cams_sulphate_aod_at_550_nm")
#|     du_aod = _c(df, "cams_dust_aod_at_550_nm")
#|     ss_aod = _c(df, "cams_sea_salt_aod_at_550_nm")
#|     tot_aod = _c(df, "cams_total_aod_at_550_nm")
#|     pm25c = _first(df, "cams_pm2_5_x1e9", "pm25_mean")
#|     pm10c = _c(df, "cams_pm10_x1e9")
#|
#|     if "cams_formaldehyde_x1e9" in df.columns or \
#|        "cams_total_column_formaldehyde_x1e9" in df.columns:
#|         add("hcho_no2_ratio", _safe_ratio(hcho, no2))
#|         add("oxidant_load", pd.Series(o3) + pd.Series(no2))
#|         add("photochem_activity", _safe_ratio(pd.Series(hcho) * pd.Series(oh), 1.0))
#|         add("biogenic_fraction", _safe_ratio(isop, pd.Series(isop) + pd.Series(hcho)))
#|         add("nox_reservoir_ratio", _safe_ratio(pan, no2))
#|         add("radical_termination", _safe_ratio(h2o2, pd.Series(h2o2) + pd.Series(no2)))
#|         add("combustion_tracer_ratio", _safe_ratio(co, no2))
#|         add("acid_precursor_load", pd.Series(so2) + pd.Series(no2))
#|
#|     # Aerosol composition: what the particulate burden is made of, not just
#|     # how much there is. Anthropogenic fraction is the health-relevant part.
#|     anthro = pd.Series(bc_aod).fillna(0) + pd.Series(om_aod).fillna(0) + \
#|              pd.Series(su_aod).fillna(0)
#|     natural = pd.Series(du_aod).fillna(0) + pd.Series(ss_aod).fillna(0)
#|     if "cams_total_aod_at_550_nm" in df.columns:
#|         add("anthropogenic_aod_fraction", _safe_ratio(anthro, pd.Series(tot_aod)))
#|         add("natural_aod_fraction", _safe_ratio(natural, pd.Series(tot_aod)))
#|         add("anthro_natural_aod_ratio", _safe_ratio(anthro, natural))
#|     if "cams_black_carbon_aod_at_550_nm" in df.columns:
#|         add("black_carbon_share", _safe_ratio(bc_aod, anthro))
#|     if "cams_pm10_x1e9" in df.columns:
#|         add("fine_coarse_pm_ratio", _safe_ratio(pm25c, pm10c))
#|
#|     # Chronic exposure interaction: fine particulate load carried in humid air.
#|     # Hygroscopic growth in humid conditions increases deposited dose.
#|     add("humid_particulate_load", pd.Series(pm25c) * (rh_mean / 100.0),
#|         needs=("prism_tmean_mean", "prism_tdmean_mean"))
#|
#|     # ---- 4. SURFACE THERMAL BEHAVIOUR -------------------------------------
#|     add("day_night_lst_range", lst_day - lst_night,
#|         needs=("lst_day_1km_mean", "lst_night_1km_mean"))
#|     add("night_cooling_efficiency", _safe_ratio(lst_day - lst_night, lst_day),
#|         needs=("lst_day_1km_mean", "lst_night_1km_mean"))
#|     add("surface_air_night_offset", lst_night - 273.15 - t_min,
#|         needs=("lst_night_1km_mean", "prism_tmin_mean"))
#|
#|     ndvi = _first(df, "ndvi_ndvi_mean_div10k", "ndvi_evi_mean_div10k",
#|                   "landsat_ndvi_mean", "ls57_ndvi_mean")
#|     add("vegetation_heat_buffer", pd.Series(ndvi) * (lst_day - lst_night),
#|         needs=("lst_day_1km_mean", "lst_night_1km_mean"))
#|     ghm = _first(df, "ghm_mean", "ghm_p50")
#|     add("impervious_heat_index", pd.Series(ghm) * (lst_day - 273.15),
#|         needs=("lst_day_1km_mean",))
#|     add("urban_moisture_trap", pd.Series(ghm) * pd.Series(td_min),
#|         needs=("prism_tdmean_min",))
#|
#|     # ---- 5. WATER BALANCE --------------------------------------------------
#|     pet = _first(df, "terra_pet_mean", "terra_pet_p50")
#|     aet = _first(df, "terra_aet_mean", "terra_aet_p50")
#|     add("aridity_index", _safe_ratio(ppt, pet))
#|     add("evaporative_fraction", _safe_ratio(aet, pet))
#|
#|     # ---- assemble and report ----------------------------------------------
#|     rows = []
#|     year = _c(df, "year")
#|     for name, series in new.items():
#|         s = pd.to_numeric(pd.Series(np.asarray(series, dtype="float64"),
#|                                     index=df.index), errors="coerce")
#|         s = s.replace([np.inf, -np.inf], np.nan)
#|         n = int(s.notna().sum())
#|         if n == 0:
#|             continue
#|         out[name] = s
#|         yy = year[s.notna()]
#|         rows.append(dict(feature=name, n_observed=n,
#|                          pct_coverage=round(100 * n / len(df), 1),
#|                          year_min=int(yy.min()) if len(yy) else None,
#|                          year_max=int(yy.max()) if len(yy) else None,
#|                          n_unique=int(s.nunique())))
#|
#|     report = pd.DataFrame(rows).sort_values("pct_coverage", ascending=False)
#|     if verbose and len(report):
#|         full = (report.pct_coverage >= 99).sum()
#|         print(f"[engineered] added {len(report)} features "
#|               f"({full} at full coverage, {len(report)-full} partial)")
#|         partial = report[report.pct_coverage < 99]
#|         for _, r in partial.iterrows():
#|             print(f"    partial: {r.feature:<44} {r.pct_coverage:5.1f}%  "
#|                   f"{r.year_min}-{r.year_max}")
#|     return out, report
#|
#|
#| if __name__ == "__main__":
#|     import sys
#|     src = sys.argv[1] if len(sys.argv) > 1 else "Final_ML_Ready_Dataset.csv"
#|     d = pd.read_csv(src, low_memory=False)
#|     d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
#|     aug, rep = add_engineered_features(d)
#|     rep.to_csv("engineered_v2_coverage.csv", index=False)
#|     print(f"\naugmented: {aug.shape[0]:,} x {aug.shape[1]:,}")
#|
#@@ END pipeline/engineered_features_v2.py
#@@ FILE pipeline/paths.py
#| import os
#|
#| # Define your project paths here (update these strings to match your actual folders!)
#| BASE_DIR = os.path.abspath(".")
#| MAP_DIR = os.path.join(BASE_DIR, "maps")
#| DATA_DIR = os.path.join(BASE_DIR, "data")
#| OUTPUT_DIR = os.path.join(BASE_DIR, "output")
#|
#| # Automatically create these folders if they don't exist yet
#| for directory in [MAP_DIR, DATA_DIR, OUTPUT_DIR]:
#|     os.makedirs(directory, exist_ok=True)
#|
#@@ END pipeline/paths.py
#@@ FILE pipeline/pipeline_A_moisture_v2.py
#| #!/usr/bin/env python3
#| # =============================================================================
#| #  pipeline_A_moisture_v2.py
#| #  STUDY A  ->  Papers 1 (flagship) and 2 (nocturnal discovery)
#| #
#| #  Environmental only. No socioeconomic, no census, no health-outcome columns.
#| #  Window 2003-2019, set by the first year of ERA5 and CAMS availability.
#| #
#| #  Stages
#| #    1  load superset, add engineered layer, save reference frame
#| #    2  temporal-coverage gate at 95 percent, environment-only scope
#| #    3  unsupervised redundancy filter with protected physiological features
#| #    4  learner tournament under identical county-grouped protocol
#| #    5  refit winner, bootstrap interval, out-of-fold predictions
#| #    6  interventional TreeSHAP on the winner, out of fold
#| #    7  discovery analyses: group hierarchy, thermal-index audit, dew-point
#| #       profile, threshold with bootstrap, orthogonality, era stability,
#| #       modality leave-one-out, error structure, residual spatial statistic
#| #    8  optional forward prediction into 2020-2023
#| #
#| #  Every stage writes artefacts; nothing downstream needs to refit a model.
#| #  Run:  python pipeline_A_moisture_v2.py --csv Final_ML_Ready_Dataset.csv --gpu
#| # =============================================================================
#|
#| from __future__ import annotations
#| import argparse, json, time
#| import numpy as np
#| import pandas as pd
#|
#| import pipeline_core_v2 as core
#| from pipeline_core_v2 import (TARGET, GROUP_COL, YEAR_COL, Artifacts,
#|                               modality_of, is_protected)
#|
#| WINDOW = (2003, 2019)
#| FORWARD = (2020, 2023)
#| TAG = "studyA"
#|
#| # leading candidates for the nocturnal-moisture analyses
#| DP_MIN = "prism_tdmean_min"
#| T_MIN = "prism_tmin_min"
#|
#|
#| # ---------------------------------------------------------------------------
#| #  physical grouping used for the attribution hierarchy
#| # ---------------------------------------------------------------------------
#| def physical_group(f: str) -> str:
#|     import re
#|     g = [
#|         (r"night_cool|nightcool|lst_night|wetbulb_night|rel_humidity_night|"
#|          r"nocturnal|tmin_min|days_tmin_over|urban_moisture_trap|"
#|          r"surface_air_night_offset", "Nocturnal thermal"),
#|         (r"tdmean|dew_?point|wetbulb|wet_bulb|humidex|rel_humidity|"
#|          r"specific_humidity|absolute_humidity|vapour_pressure|"
#|          r"total_column_water|column_moisture", "Atmospheric moisture"),
#|         (r"vpd|aridity|evaporative_fraction|terra_pet|terra_aet|terra_def|"
#|          r"ventilation_index", "Evaporative demand"),
#|         (r"cams_|hcho|no2|ozone|oxidant|aerosol|aod|pm25|pm10|pm1|"
#|          r"photochem|biogenic|combustion|acid_precursor|black_carbon|"
#|          r"radical|nox_reservoir|particulate", "Air quality and composition"),
#|         (r"tmax|tmean|prism_tmin|heat_index|wbgt|discomfort|degree_days|"
#|          r"days_tmax_over|diurnal|lst_day|summer_t|winter_t|era5_2m_temperature|"
#|          r"era5_temperature|era5_fot_temperature|day_night_lst", "Heat and temperature"),
#|         (r"ghm_|impervious|ntl_|urban|econ_", "Built environment"),
#|         (r"ndvi|evi|savi|ndmi|meta_canopy|leaf_area|vegetation_heat", "Vegetation"),
#|         (r"soil_|dem_|geopotential|topo|elevation", "Soil and terrain"),
#|         (r"jrc_|water|ndwi|snow|terra_ro|terra_swe|runoff", "Water and hydrology"),
#|         (r"wind|era5_10m_wind", "Wind"),
#|         (r"pdsi|drought", "Drought"),
#|         (r"ppt|precip", "Precipitation"),
#|         (r"usda_|live_|crop", "Agriculture and livestock"),
#|     ]
#|     for pat, label in g:
#|         if re.search(pat, f, re.I):
#|             return label
#|     return "Other"
#|
#|
#| def smooth_xy(x, y, frac=0.3):
#|     try:
#|         from statsmodels.nonparametric.smoothers_lowess import lowess
#|         s = lowess(y, x, frac=frac, return_sorted=True)
#|         return s[:, 0], s[:, 1]
#|     except Exception:
#|         o = np.argsort(x)
#|         xs, ys = np.asarray(x)[o], np.asarray(y)[o]
#|         k = max(15, len(xs) // 25)
#|         return xs, pd.Series(ys).rolling(k, center=True, min_periods=1).mean().values
#|
#|
#| def zero_crossing(x, y, n_bins=40):
#|     """Value at which the binned mean response passes through zero.
#|
#|     Binning rather than smoothing makes each bootstrap replicate O(n) instead
#|     of O(n^2), which matters at 50,000+ rows. The estimate is restricted to the
#|     data-dense interior (5th to 95th percentile) so that a wiggle in a sparse
#|     tail cannot be mistaken for a threshold. That restriction is deliberate: an
#|     unconstrained derivative on this response returns a value far outside the
#|     observed range, which is an artefact rather than a finding.
#|     """
#|     x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
#|     ok = np.isfinite(x) & np.isfinite(y)
#|     x, y = x[ok], y[ok]
#|     if len(x) < 50:
#|         return np.nan
#|     lo, hi = np.percentile(x, [5, 95])
#|     m = (x >= lo) & (x <= hi)
#|     x, y = x[m], y[m]
#|     if len(x) < 50:
#|         return np.nan
#|     edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
#|     edges = np.unique(edges)
#|     if len(edges) < 5:
#|         return np.nan
#|     idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
#|     cx = np.bincount(idx, weights=x, minlength=len(edges) - 1)
#|     cy = np.bincount(idx, weights=y, minlength=len(edges) - 1)
#|     cn = np.bincount(idx, minlength=len(edges) - 1).astype(float)
#|     good = cn >= 5
#|     if good.sum() < 5:
#|         return np.nan
#|     bx, by = cx[good] / cn[good], cy[good] / cn[good]
#|     sign = np.sign(by)
#|     cross = np.where(np.diff(sign) != 0)[0]
#|     if not len(cross):
#|         return np.nan
#|     j = cross[len(cross) // 2]
#|     denom = by[j + 1] - by[j]
#|     if abs(denom) < 1e-12:
#|         return float(bx[j])
#|     return float(bx[j] - by[j] * (bx[j + 1] - bx[j]) / denom)
#|
#|
#| # ---------------------------------------------------------------------------
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--csv", default="Final_ML_Ready_Dataset.csv")
#|     ap.add_argument("--outdir", default="results_studyA")
#|     ap.add_argument("--folds", type=int, default=8)
#|     ap.add_argument("--gpu", action="store_true")
#|     ap.add_argument("--fast", action="store_true", help="fewer trees, for a smoke test")
#|     ap.add_argument("--coverage", type=float, default=0.95)
#|     ap.add_argument("--n-boot", type=int, default=500,
#|                     help="bootstrap replicates for the threshold interval")
#|     ap.add_argument("--skip-era", action="store_true",
#|                     help="skip the era sub-model stability step (expensive)")
#|     ap.add_argument("--skip-loo", action="store_true",
#|                     help="skip modality leave-one-out (expensive)")
#|     ap.add_argument("--forward", action="store_true",
#|                     help="also predict 2020-2023 where inputs exist (no ground truth)")
#|     args = ap.parse_args()
#|
#|     t_start = time.time()
#|     A = Artifacts(args.outdir, TAG)
#|     print("=" * 74)
#|     print(f"STUDY A  environment only  window {WINDOW[0]}-{WINDOW[1]}")
#|     print("=" * 74)
#|
#|     # -- 1  load and engineer ------------------------------------------------
#|     df_all = core.load_master(args.csv, add_engineered=True)
#|     df = df_all[(df_all[YEAR_COL] >= WINDOW[0]) &
#|                 (df_all[YEAR_COL] <= WINDOW[1])].reset_index(drop=True)
#|     print(f"[window] {len(df):,} rows, {df[GROUP_COL].nunique():,} counties, "
#|           f"{df[YEAR_COL].nunique()} years")
#|     A.frame("reference_frame", df)
#|
#|     # -- 2  coverage gate ----------------------------------------------------
#|     feats, gate = core.coverage_gate(df, WINDOW, min_coverage=args.coverage,
#|                                      allow_sdoh=False, drop_raw_ntl=True)
#|     A.table("feature_gate", gate)
#|
#|     # -- 3  redundancy filter -----------------------------------------------
#|     feats = core.redundancy_filter(df[feats])
#|     A.json("feature_set", dict(n_features=len(feats), features=feats))
#|
#|     # -- 4  learner tournament ----------------------------------------------
#|     print("\n[4] learner tournament")
#|     table, results, best = core.run_tournament(
#|         df, feats, n_folds=args.folds, use_gpu=args.gpu, fast=args.fast)
#|     A.table("model_tournament", table)
#|
#|     zoo = core.build_models(use_gpu=args.gpu, fast=args.fast)
#|     best_ctor, best_shap_ok = zoo[best]
#|     res = results[best]
#|
#|     # -- 5  headline metrics -------------------------------------------------
#|     y = res["data"][TARGET].values
#|     ci = core.bootstrap_r2_ci(y, res["oof"])
#|     m = dict(res["metrics"]); m.update(model=best, r2_ci_lo=ci[0], r2_ci_hi=ci[1],
#|                                        window=list(WINDOW))
#|     print(f"\n[5] best model {best}: R2={m['r2']:.4f} "
#|           f"(95% CI {ci[0]:.4f}-{ci[1]:.4f}) MAE={m['mae']:.4f} RMSE={m['rmse']:.4f}")
#|     print(f"    within 1 yr {m['within_1yr']:.1f}% | within 2 yr {m['within_2yr']:.1f}%")
#|
#|     oof = res["data"][[GROUP_COL, YEAR_COL, TARGET]].copy()
#|     oof["pred"] = res["oof"]
#|     oof["resid"] = oof[TARGET] - oof["pred"]
#|     A.table("oof_predictions", oof)
#|     A.table("cv_folds", res["folds"])
#|
#|     # -- 6  attribution ------------------------------------------------------
#|     if best_shap_ok:
#|         print("\n[6] interventional TreeSHAP on the winning model")
#|         ranking, per_obs, sdata = core.interventional_shap(
#|             df, feats, best_ctor, n_folds=args.folds)
#|         ranking["group"] = ranking.feature.map(physical_group)
#|         A.table("shap_ranking", ranking)
#|         keep_cols = [c for c in [DP_MIN, T_MIN, "eng2_wetbulb_night", "eng2_wetbulb_hot",
#|                                  "eng2_night_cooling_efficiency", "lst_night_1km_mean",
#|                                  "eng2_hcho_no2_ratio", "pm25_mean"] if c in feats]
#|         dep = sdata.loc[per_obs.index, keep_cols].copy()
#|         for c in keep_cols:
#|             dep["shap_" + c] = per_obs[c].values
#|         A.table("dependence_data", dep)
#|         print("    top 15:")
#|         for _, r in ranking.head(15).iterrows():
#|             print(f"      #{int(r['rank']):<4d} {r.feature[:46]:<46} "
#|                   f"|phi|={r.mean_abs_shap:.4f}  [{r.group}]")
#|     else:
#|         ranking, per_obs = None, None
#|         print("\n[6] winning model does not support TreeSHAP; attribution skipped")
#|
#|     # -- 7  discovery analyses ----------------------------------------------
#|     print("\n[7] discovery analyses")
#|
#|     if ranking is not None:
#|         hier = (ranking.groupby("group")
#|                 .agg(sum_shap=("mean_abs_shap", "sum"),
#|                      per_feature=("mean_abs_shap", "mean"),
#|                      n_features=("feature", "size"),
#|                      best_rank=("rank", "min"),
#|                      top_feature=("feature", "first"))
#|                 .sort_values("sum_shap", ascending=False))
#|         hier["pct_of_total"] = 100 * hier.sum_shap / hier.sum_shap.sum()
#|         rng = np.random.default_rng(0)
#|         lo, hi = [], []
#|         for g in hier.index:
#|             v = ranking.loc[ranking.group == g, "mean_abs_shap"].values
#|             b = [rng.choice(v, len(v), replace=True).sum() for _ in range(2000)]
#|             lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
#|         hier["ci_lo"], hier["ci_hi"] = lo, hi
#|         A.table("group_hierarchy", hier.reset_index())
#|         print("  group hierarchy (per-feature order):")
#|         for g, r in hier.sort_values("per_feature", ascending=False).head(8).iterrows():
#|             print(f"    {g[:34]:<34} n={int(r.n_features):>4} "
#|                   f"sum={r.sum_shap:.3f} perfeat={r.per_feature:.4f}")
#|
#|         FAM = {
#|             "Pure moisture (dew point)": r"tdmean",
#|             "Composite wet-bulb": r"wetbulb|wet_bulb",
#|             "Composite WBGT / heat index / humidex": r"wbgt|heat_index|humidex|discomfort",
#|             "Pure dry heat (air temperature)": r"prism_tm(ax|in|ean)|era5_2m_temperature",
#|             "Evaporative demand (VPD)": r"vpd",
#|             "Air composition (CAMS)": r"^cams_|hcho_no2",
#|             "Particulates": r"pm25|pm10|pm1_",
#|         }
#|         rows = []
#|         for fam, pat in FAM.items():
#|             s = ranking[ranking.feature.str.contains(pat, regex=True, case=False)]
#|             if not len(s):
#|                 continue
#|             rows.append(dict(family=fam, n=len(s), best_rank=int(s["rank"].min()),
#|                              best_feature=s.iloc[0].feature,
#|                              best_shap=float(s.mean_abs_shap.max()),
#|                              sum_shap=float(s.mean_abs_shap.sum()),
#|                              per_feature=float(s.mean_abs_shap.mean())))
#|         fam = pd.DataFrame(rows).sort_values("best_shap", ascending=False)
#|         A.table("thermal_index_family", fam)
#|         print("  thermal-index family audit:")
#|         for _, r in fam.iterrows():
#|             print(f"    {r.family[:38]:<38} n={int(r.n):>3} best=#{int(r.best_rank):<4d} "
#|                   f"{r.best_shap:.4f}")
#|
#|         prof = ranking[ranking.feature.str.startswith("prism_tdmean")].copy()
#|         if len(prof):
#|             A.table("dewpoint_profile", prof)
#|
#|         s = ranking.mean_abs_shap.sort_values(ascending=False).values
#|         cum = np.cumsum(s) / s.sum()
#|         A.table("concentration", pd.DataFrame(
#|             {"k": np.arange(1, len(s) + 1), "cum_share": cum}))
#|         lor = np.cumsum(np.sort(s)) / s.sum()
#|         trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
#|         m["attribution_gini"] = float(1 - 2 * trap(lor, dx=1 / len(s)))
#|         m["n_feat_50pct"] = int(np.searchsorted(cum, .5) + 1)
#|         m["n_feat_80pct"] = int(np.searchsorted(cum, .8) + 1)
#|
#|     # threshold and orthogonality
#|     if per_obs is not None and DP_MIN in feats:
#|         x = sdata.loc[per_obs.index, DP_MIN].values
#|         phi = per_obs[DP_MIN].values
#|         ok = ~np.isnan(x) & ~np.isnan(phi)
#|         x, phi = x[ok], phi[ok]
#|         pt = zero_crossing(x, phi)
#|         rng = np.random.default_rng(0)
#|         bs = []
#|         for _ in range(args.n_boot):
#|             s_ = rng.integers(0, len(x), len(x))
#|             c = zero_crossing(x[s_], phi[s_])
#|             if not np.isnan(c):
#|                 bs.append(c)
#|         thr = dict(zero_cross=pt,
#|                    ci_lo=float(np.percentile(bs, 2.5)) if bs else np.nan,
#|                    ci_hi=float(np.percentile(bs, 97.5)) if bs else np.nan,
#|                    pct_above=float((df[DP_MIN] > pt).mean() * 100) if not np.isnan(pt) else np.nan,
#|                    n_boot=len(bs))
#|         es = 6.112 * np.exp(17.67 * pt / (pt + 243.5)) if not np.isnan(pt) else np.nan
#|         thr["abs_humidity_g_per_kg"] = float(0.622 * es / (1013.25 - es) * 1000) \
#|             if not np.isnan(pt) else np.nan
#|         A.table("dpmin_threshold", pd.DataFrame([thr]))
#|         print(f"  dew-point threshold: {pt:.3f} C "
#|               f"[{thr['ci_lo']:.3f}, {thr['ci_hi']:.3f}], "
#|               f"{thr['pct_above']:.1f}% of county-years above")
#|
#|         if T_MIN in feats:
#|             raw_r = float(df[[DP_MIN, T_MIN]].corr().iloc[0, 1])
#|             shap_r = float(np.corrcoef(per_obs[DP_MIN], per_obs[T_MIN])[0, 1])
#|             orth = dict(raw_corr=raw_r, shap_corr=shap_r,
#|                         dp_shap_range=float(per_obs[DP_MIN].max() - per_obs[DP_MIN].min()),
#|                         t_shap_range=float(per_obs[T_MIN].max() - per_obs[T_MIN].min()))
#|             orth["range_ratio"] = orth["dp_shap_range"] / max(orth["t_shap_range"], 1e-9)
#|             A.table("orthogonality", pd.DataFrame([orth]))
#|             print(f"  orthogonality: raw r={raw_r:+.3f}  SHAP r={shap_r:+.3f}  "
#|                   f"range ratio {orth['range_ratio']:.1f}x")
#|
#|             # dose-response inside minimum-temperature quartiles
#|             tq = pd.qcut(sdata.loc[per_obs.index, T_MIN], 4,
#|                          labels=["Q1 coldest", "Q2", "Q3", "Q4 warmest"])
#|             srows = []
#|             for lev in tq.cat.categories:
#|                 mm = (tq == lev).values
#|                 if mm.sum() < 50:
#|                     continue
#|                 xs = sdata.loc[per_obs.index, DP_MIN].values[mm]
#|                 ps = per_obs[DP_MIN].values[mm]
#|                 good = ~np.isnan(xs) & ~np.isnan(ps)
#|                 srows.append(dict(stratum=str(lev), n=int(good.sum()),
#|                                   zero_cross=zero_crossing(xs[good], ps[good]),
#|                                   slope=float(np.polyfit(xs[good], ps[good], 1)[0])))
#|             A.table("temp_strata", pd.DataFrame(srows))
#|
#|     # era stability
#|     eras = [] if args.skip_era else [(WINDOW[0], WINDOW[0] + 4),
#|                                      (WINDOW[1] - 4, WINDOW[1])]
#|     erows = []
#|     for a, b in eras:
#|         sub = df[(df[YEAR_COL] >= a) & (df[YEAR_COL] <= b)]
#|         if sub[GROUP_COL].nunique() < 100:
#|             continue
#|         rk, _, _ = core.interventional_shap(sub, feats, best_ctor, n_folds=5,
#|                                             sample_size=1500, verbose=False)
#|         row = dict(era=f"{a}-{b}", n_obs=len(sub))
#|         for f in [DP_MIN, "eng2_night_cooling_efficiency"]:
#|             r = rk[rk.feature == f]
#|             if len(r):
#|                 row[f + "_rank"] = int(r["rank"].iloc[0])
#|                 row[f + "_shap"] = float(r.mean_abs_shap.iloc[0])
#|         erows.append(row)
#|     if erows:
#|         A.table("era_stability", pd.DataFrame(erows))
#|         print(f"  era stability: {erows}")
#|
#|     # modality leave-one-out
#|     loo = []
#|     if not args.skip_loo:
#|         print("  modality leave-one-out")
#|     mods = [] if args.skip_loo else sorted({modality_of(f) for f in feats})
#|     for mk in mods:
#|         keep = [f for f in feats if modality_of(f) != mk]
#|         if len(keep) < 20:
#|             continue
#|         r = core.cv_evaluate(df, keep, best_ctor, n_folds=5)["metrics"]["r2"]
#|         loo.append(dict(modality=mk, n_removed=len(feats) - len(keep),
#|                         r2_without=r, delta=r - m["r2"]))
#|         print(f"    {mk[:38]:<38} R2={r:.4f}  delta={r - m['r2']:+.4f}")
#|     if loo:
#|         A.table("modality_loo", pd.DataFrame(loo).sort_values("delta"))
#|
#|     # error structure and residual spatial statistic
#|     oof["bracket"] = pd.cut(oof[TARGET], [60, 70, 74, 76.5, 79, 95],
#|                             labels=["60-70", "70-74", "74-76.5", "76.5-79", "79-95"])
#|     br = (oof.groupby("bracket", observed=True)
#|           .agg(n=("resid", "size"), mae=("resid", lambda s: s.abs().mean()),
#|                mean_resid=("resid", "mean")).reset_index())
#|     A.table("bracket_error", br)
#|     per_year = (oof.groupby(YEAR_COL)
#|                 .apply(lambda g: pd.Series({
#|                     "r2": 1 - ((g.resid ** 2).sum() /
#|                                ((g[TARGET] - g[TARGET].mean()) ** 2).sum()),
#|                     "mae": g.resid.abs().mean(), "n": len(g)}))
#|                 .reset_index())
#|     A.table("per_year", per_year)
#|
#|     cty = oof.groupby(GROUP_COL).resid.mean().reset_index()
#|     cty["st"] = cty[GROUP_COL].astype(str).str.zfill(5).str[:2]
#|     gm = cty.resid.mean(); dev = cty.resid - gm
#|     num = W = 0.0
#|     for _, sub in cty.groupby("st"):
#|         v = (sub.resid - gm).values; k = len(v)
#|         if k > 1:
#|             num += v.sum() ** 2 - (v ** 2).sum(); W += k * (k - 1)
#|     den = (dev ** 2).sum()
#|     m["morans_I_resid"] = float((len(cty) / W) * (num / den)) if W and den else None
#|     print(f"  Moran's I of residuals: {m['morans_I_resid']}")
#|
#|     # -- 8  final model and optional forward prediction ---------------------
#|     print("\n[8] refit on all rows and save model")
#|     full_model, Xf, fdata, med = core.fit_full_model(df, feats, best_ctor)
#|     A.model("final_model", dict(model=full_model, features=feats,
#|                                 median_fill=med.to_dict(), window=WINDOW,
#|                                 model_name=best))
#|
#|     if args.forward:
#|         fw = df_all[(df_all[YEAR_COL] >= FORWARD[0]) &
#|                     (df_all[YEAR_COL] <= FORWARD[1])].copy()
#|         if len(fw):
#|             avail = fw[feats].notna().mean()
#|             usable = avail[avail >= 0.80].index.tolist()
#|             print(f"  forward window {FORWARD[0]}-{FORWARD[1]}: {len(fw):,} rows, "
#|                   f"{len(usable)}/{len(feats)} features at >=80% coverage")
#|             if len(usable) >= 0.7 * len(feats):
#|                 pred = full_model.predict(fw[feats].fillna(med))
#|                 out = fw[[GROUP_COL, YEAR_COL]].copy()
#|                 out["predicted_le"] = pred
#|                 out["note"] = "extrapolation beyond target availability; no ground truth"
#|                 A.table("forward_predictions", out)
#|             else:
#|                 print("  too few features available; forward prediction skipped")
#|
#|     A.json("metrics", m)
#|     A.close()
#|     print(f"\nSTUDY A complete in {(time.time()-t_start)/60:.1f} min "
#|           f"-> {args.outdir}/")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END pipeline/pipeline_A_moisture_v2.py
#@@ FILE pipeline/pipeline_B_vanishing_v2.py
#| #!/usr/bin/env python3
#| # =============================================================================
#| #  pipeline_B_vanishing_v2.py
#| #  STUDY B  ->  Paper 3 (the vanishing environmental increment)
#| #
#| #  Environmental superset fused with the socioeconomic panel. Window 2009-2019,
#| #  set by the first year of socioeconomic availability.
#| #
#| #  Stages
#| #    1  load superset, add engineered layer, save reference frame
#| #    2  temporal-coverage gate at 80 percent, socioeconomic scope allowed
#| #    3  unsupervised redundancy filter with protected physiological features
#| #    4  learner tournament, fused feature set
#| #    5  headline metrics, bootstrap interval, out-of-fold predictions
#| #    6  information partition, saturation-richness curve
#| #    7  perturbation stress tests, conformal calibration
#| #    8  interventional TreeSHAP, composition, debiased partial screen
#| #    9  spurious-proxy audit, residual diagnostics
#| #   10  environment-only backcast into the pre-socioeconomic era
#| #
#| #  Run:  python pipeline_B_vanishing_v2.py --csv Final_ML_Ready_Dataset.csv --gpu
#| # =============================================================================
#|
#| from __future__ import annotations
#| import argparse, time
#| import numpy as np
#| import pandas as pd
#|
#| import pipeline_core_v2 as core
#| from pipeline_core_v2 import (TARGET, GROUP_COL, YEAR_COL, Artifacts,
#|                               modality_of, is_sdoh)
#|
#| WINDOW = (2009, 2019)
#| BACKCAST = (2003, 2008)     # env-only era; CAMS and ERA5 start in 2003
#| TAG = "studyB"
#|
#|
#| def split_sources(features):
#|     sdoh = [f for f in features if is_sdoh(f)]
#|     env = [f for f in features if not is_sdoh(f)]
#|     return sdoh, env
#|
#|
#| # ---------------------------------------------------------------------------
#| def main():
#|     ap = argparse.ArgumentParser()
#|     ap.add_argument("--csv", default="Final_ML_Ready_Dataset.csv")
#|     ap.add_argument("--outdir", default="results_studyB")
#|     ap.add_argument("--folds", type=int, default=8)
#|     ap.add_argument("--gpu", action="store_true")
#|     ap.add_argument("--fast", action="store_true")
#|     ap.add_argument("--coverage", type=float, default=0.80)
#|     args = ap.parse_args()
#|
#|     t0 = time.time()
#|     B = Artifacts(args.outdir, TAG)
#|     print("=" * 74)
#|     print(f"STUDY B  satellite + socioeconomic fusion  window {WINDOW[0]}-{WINDOW[1]}")
#|     print("=" * 74)
#|
#|     # -- 1  load -------------------------------------------------------------
#|     df_all = core.load_master(args.csv, add_engineered=True)
#|     df = df_all[(df_all[YEAR_COL] >= WINDOW[0]) &
#|                 (df_all[YEAR_COL] <= WINDOW[1])].reset_index(drop=True)
#|     print(f"[window] {len(df):,} rows, {df[GROUP_COL].nunique():,} counties")
#|     B.frame("reference_frame", df)
#|
#|     # -- 2  gate -------------------------------------------------------------
#|     feats, gate = core.coverage_gate(df, WINDOW, min_coverage=args.coverage,
#|                                      allow_sdoh=True, drop_raw_ntl=True)
#|     B.table("feature_gate", gate)
#|     n_leak = int((gate.reason == "LEAKAGE-outcome-derived").sum())
#|     print(f"[gate] {n_leak} outcome-derived columns blocked before modelling")
#|
#|     # -- 3  filter -----------------------------------------------------------
#|     feats = core.redundancy_filter(df[feats])
#|     sdoh, env = split_sources(feats)
#|     print(f"[scope] {len(feats)} features: {len(sdoh)} socioeconomic, "
#|           f"{len(env)} environmental")
#|     B.json("feature_set", dict(n_total=len(feats), n_sdoh=len(sdoh),
#|                                n_env=len(env), sdoh=sdoh, env=env))
#|
#|     # -- 4  tournament -------------------------------------------------------
#|     print("\n[4] learner tournament, fused feature set")
#|     table, results, best = core.run_tournament(
#|         df, feats, n_folds=args.folds, use_gpu=args.gpu, fast=args.fast)
#|     B.table("model_tournament", table)
#|     zoo = core.build_models(use_gpu=args.gpu, fast=args.fast)
#|     best_ctor, best_shap_ok = zoo[best]
#|     res = results[best]
#|
#|     # -- 5  headline ---------------------------------------------------------
#|     y = res["data"][TARGET].values
#|     ci = core.bootstrap_r2_ci(y, res["oof"])
#|     M = dict(res["metrics"]); M.update(model=best, r2_ci_lo=ci[0], r2_ci_hi=ci[1],
#|                                        window=list(WINDOW), n_sdoh=len(sdoh),
#|                                        n_env=len(env), n_leakage_blocked=n_leak)
#|     print(f"\n[5] fused {best}: R2={M['r2']:.4f} (95% CI {ci[0]:.4f}-{ci[1]:.4f}) "
#|           f"RMSE={M['rmse']:.4f} MAE={M['mae']:.4f}")
#|     oof = res["data"][[GROUP_COL, YEAR_COL, TARGET]].copy()
#|     oof["pred"] = res["oof"]; oof["resid"] = oof[TARGET] - oof["pred"]
#|     B.table("oof_predictions", oof)
#|     B.table("cv_folds", res["folds"])
#|
#|     # -- 6  information partition -------------------------------------------
#|     print("\n[6] information partition")
#|     R_SE = M["r2"]
#|     R_S = core.cv_evaluate(df, sdoh, best_ctor, n_folds=args.folds)["metrics"]["r2"]
#|     R_E = core.cv_evaluate(df, env, best_ctor, n_folds=args.folds)["metrics"]["r2"]
#|     part = dict(R_S=R_S, R_E=R_E, R_SE=R_SE,
#|                 U_E=R_SE - R_S, U_S=R_SE - R_E, Shared=R_S + R_E - R_SE)
#|     part["U_E_pct"] = 100 * part["U_E"] / R_SE
#|     part["U_S_pct"] = 100 * part["U_S"] / R_SE
#|     part["Shared_pct"] = 100 * part["Shared"] / R_SE
#|     B.table("info_partition", pd.DataFrame([part]))
#|     print(f"    socioeconomic only R2={R_S:.4f} | environment only R2={R_E:.4f} "
#|           f"| fused R2={R_SE:.4f}")
#|     print(f"    unique to environment {part['U_E']:.4f} ({part['U_E_pct']:.2f}%) | "
#|           f"unique to socioeconomic {part['U_S']:.4f} | shared {part['Shared']:.4f}")
#|     if part["Shared"] < 0:
#|         # R_S + R_E < R_SE means the two streams are SYNERGISTIC rather than
#|         # redundant: together they explain more than the sum of their parts.
#|         # That is a different finding from the vanishing increment and must be
#|         # reported as such rather than described as redundancy.
#|         print("    NOTE: shared component is negative, so the two sources are "
#|               "synergistic\n          rather than redundant in this window. "
#|               "Report this explicitly; it is\n          NOT the vanishing-increment "
#|               "result and must not be worded as one.")
#|         part["interpretation"] = "synergistic (negative shared component)"
#|     else:
#|         part["interpretation"] = "redundant (positive shared component)" 
#|
#|     # saturation-richness curve: rank socioeconomic features first
#|     print("\n    saturation-richness curve")
#|     rank_s = core.interventional_shap(df, sdoh, best_ctor, n_folds=4,
#|                                       sample_size=1500, verbose=False)[0] \
#|         if best_shap_ok else None
#|     order = list(rank_s.feature) if rank_s is not None else sdoh
#|     sizes = [n for n in (1, 5, 20, 50, 100, 200, len(order)) if n <= len(order)]
#|     srows = []
#|     for n in sorted(set(sizes)):
#|         base = order[:n]
#|         rs = core.cv_evaluate(df, base, best_ctor, n_folds=5)["metrics"]["r2"]
#|         rf = core.cv_evaluate(df, base + env, best_ctor, n_folds=5)["metrics"]["r2"]
#|         srows.append(dict(n_sdoh=n, R_S=rs, R_SE=rf, increment=rf - rs))
#|         print(f"      n={n:>4}  base R2={rs:.4f}  fused R2={rf:.4f}  "
#|               f"increment={rf-rs:+.4f}")
#|     sat = pd.DataFrame(srows)
#|     pos = sat[sat.increment > 0]
#|     if len(pos) >= 3:
#|         b, loga = np.polyfit(np.log(pos.n_sdoh), np.log(pos.increment), 1)
#|         sat["power_fit"] = np.exp(loga) * sat.n_sdoh ** b
#|         M["sat_exponent"] = float(b); M["sat_coef"] = float(np.exp(loga))
#|     B.table("saturation_curve", sat)
#|
#|     # -- 7  perturbation and calibration ------------------------------------
#|     print("\n[7] perturbation stress tests")
#|     from sklearn.model_selection import GroupKFold
#|     from sklearn.metrics import r2_score
#|     rng = np.random.default_rng(0)
#|     data = res["data"]
#|     X, yv, g = data[feats].copy(), data[TARGET].values, data[GROUP_COL].values
#|
#|     def oof_r2(Xin):
#|         pred = np.full(len(data), np.nan)
#|         for tr, te in GroupKFold(n_splits=5).split(Xin, yv, g):
#|             med = Xin.iloc[tr].median(numeric_only=True)
#|             mm = best_ctor(); mm.fit(Xin.iloc[tr].fillna(med), yv[tr])
#|             pred[te] = mm.predict(Xin.iloc[te].fillna(med))
#|         return float(r2_score(yv, pred))
#|
#|     pert = {"baseline": oof_r2(X)}
#|     for label, cols in [("SDOH_shuffled", sdoh), ("EO_shuffled", env)]:
#|         Z = X.copy(); Z[cols] = Z[cols].values[rng.permutation(len(Z))]
#|         pert[label] = oof_r2(Z)
#|     for frac in (0.1, 0.3, 0.6):
#|         Z = X.copy()
#|         Z[env] = Z[env].mask(rng.random(Z[env].shape) < frac)
#|         pert[f"EO_missing_{int(frac*100)}pct"] = oof_r2(Z)
#|     for sd in (0.1, 0.5, 1.0):
#|         Z = X.copy()
#|         Z.loc[:, env] = Z[env].values + rng.normal(0, 1, Z[env].shape) * \
#|             Z[env].std(axis=0).values * sd
#|         pert[f"EO_noise_{int(sd*100)}pct"] = oof_r2(Z)
#|     B.table("perturbation", pd.DataFrame([pert]))
#|     drop_s = pert["baseline"] - pert["SDOH_shuffled"]
#|     drop_e = pert["baseline"] - pert["EO_shuffled"]
#|     print(f"    shuffling socioeconomic costs {drop_s:.4f}; "
#|           f"shuffling environment costs {drop_e:.4f} "
#|           f"(asymmetry {drop_s/max(drop_e,1e-9):.1f}x)")
#|     M["pert_asymmetry"] = float(drop_s / max(drop_e, 1e-9))
#|
#|     # conformal calibration
#|     idx = rng.permutation(len(data))
#|     ntr, ncal = int(.6 * len(data)), int(.2 * len(data))
#|     tr, cal, te = idx[:ntr], idx[ntr:ntr + ncal], idx[ntr + ncal:]
#|     med = X.iloc[tr].median(numeric_only=True)
#|     cm = best_ctor(); cm.fit(X.iloc[tr].fillna(med), yv[tr])
#|     r_cal = np.abs(yv[cal] - cm.predict(X.iloc[cal].fillna(med)))
#|     q = float(np.quantile(r_cal, 0.8))
#|     naive = float(np.quantile(np.abs(yv[tr] - cm.predict(X.iloc[tr].fillna(med))), 0.8))
#|     pr = cm.predict(X.iloc[te].fillna(med))
#|     conf = dict(nominal=0.8,
#|                 coverage_before=float(np.mean(np.abs(yv[te] - pr) <= naive)),
#|                 coverage_after=float(np.mean(np.abs(yv[te] - pr) <= q)),
#|                 width_before=2 * naive, width_after=2 * q)
#|     B.table("conformal", pd.DataFrame([conf]))
#|     print(f"    conformal coverage {conf['coverage_before']:.3f} -> "
#|           f"{conf['coverage_after']:.3f} (nominal 0.80)")
#|
#|     # -- 8  attribution and debiased screen ---------------------------------
#|     if best_shap_ok:
#|         print("\n[8] interventional TreeSHAP on the fused model")
#|         ranking, per_obs, sdata = core.interventional_shap(
#|             df, feats, best_ctor, n_folds=args.folds)
#|         ranking["source"] = np.where(ranking.feature.map(is_sdoh),
#|                                      "Socioeconomic", "Environmental")
#|         B.table("shap_ranking", ranking)
#|         comp = (ranking.groupby("source")
#|                 .agg(n=("feature", "size"), sum_shap=("mean_abs_shap", "sum"),
#|                      best_rank=("rank", "min")))
#|         comp["pct_attribution"] = 100 * comp.sum_shap / comp.sum_shap.sum()
#|         B.table("source_composition", comp.reset_index())
#|         print("    top 15:")
#|         for _, r in ranking.head(15).iterrows():
#|             print(f"      #{int(r['rank']):<4d} {r.feature[:44]:<44} "
#|                   f"|phi|={r.mean_abs_shap:.4f} [{r.source}]")
#|
#|         # debiased partial screen on the leading features
#|         print("\n    debiased partial screen")
#|         from sklearn.model_selection import KFold
#|         top = list(ranking.feature.head(8))
#|         drows = []
#|         Xall = data[feats]
#|         for f in top:
#|             others = [c for c in feats if c != f]
#|             Xo = Xall[others].fillna(Xall[others].median())
#|             d = Xall[f].fillna(Xall[f].median()).values
#|             yr_, dr_ = np.zeros(len(d)), np.zeros(len(d))
#|             for tr2, te2 in KFold(5, shuffle=True, random_state=0).split(Xo):
#|                 my = best_ctor(); my.fit(Xo.iloc[tr2], yv[tr2])
#|                 md = best_ctor(); md.fit(Xo.iloc[tr2], d[tr2])
#|                 yr_[te2] = yv[te2] - my.predict(Xo.iloc[te2])
#|                 dr_[te2] = d[te2] - md.predict(Xo.iloc[te2])
#|             beta = np.polyfit(dr_, yr_, 1)[0]
#|             drows.append(dict(feature=f, partial_r2=max(0.0, r2_score(yr_, beta * dr_))))
#|             print(f"      {f[:46]:<46} partial R2={drows[-1]['partial_r2']:.5f}")
#|         B.table("dml_partial_r2", pd.DataFrame(drows).sort_values(
#|             "partial_r2", ascending=False))
#|
#|         # -- 9  spurious-proxy audit ----------------------------------------
#|         print("\n[9] spurious-proxy audit")
#|         susp = ranking[(ranking["rank"] <= 60) &
#|                        (ranking.feature.str.contains(
#|                            r"^live_|^usda_|_class_|buffalo|corn|soy|wheat",
#|                            regex=True, case=False))]
#|         arows = []
#|         for f in susp.feature:
#|             if f not in df.columns:
#|                 continue
#|             gg = df.groupby(GROUP_COL)[f]
#|             within, between = gg.std().mean(), gg.mean().std()
#|             st = (df.assign(_st=df[GROUP_COL].astype(str).str.zfill(5).str[:2])
#|                   .groupby("_st")[f].mean())
#|             arows.append(dict(
#|                 feature=f, rank=int(ranking.loc[ranking.feature == f, "rank"].iloc[0]),
#|                 between_within=float(between / max(within, 1e-9)),
#|                 state_concentration=float(st.std() / max(abs(st.mean()), 1e-9)),
#|                 nonzero_share=float((df[f].fillna(0) != 0).mean()),
#|                 observed_share=float(df[f].notna().mean())))
#|         if arows:
#|             aud = pd.DataFrame(arows).sort_values("between_within", ascending=False)
#|             B.table("proxy_audit", aud)
#|             print(aud.round(3).to_string(index=False))
#|             # sensitivity: does removing them change anything?
#|             keep2 = [f for f in feats if f not in set(aud.feature)]
#|             r2_wo = core.cv_evaluate(df, keep2, best_ctor, n_folds=5)["metrics"]["r2"]
#|             s2, e2 = split_sources(keep2)
#|             rs2 = core.cv_evaluate(df, s2, best_ctor, n_folds=5)["metrics"]["r2"]
#|             M["proxy_sensitivity_r2_with"] = R_SE
#|             M["proxy_sensitivity_r2_without"] = r2_wo
#|             M["proxy_sensitivity_UE_without"] = r2_wo - rs2
#|             print(f"    sensitivity: fused R2 {R_SE:.4f} -> {r2_wo:.4f}; "
#|                   f"unique-environment {part['U_E']:.4f} -> {r2_wo - rs2:.4f}")
#|         else:
#|             print("    no candidates flagged in the top 60")
#|     else:
#|         ranking = None
#|
#|     # residual outliers
#|     worst = (oof.groupby(GROUP_COL).resid.mean()
#|              .sort_values().head(15).reset_index())
#|     B.table("residual_outliers", worst)
#|     print("\n    largest shortfalls:")
#|     print(worst.head(6).round(2).to_string(index=False))
#|
#|     # -- 10  backcast --------------------------------------------------------
#|     print("\n[10] environment-only backcast")
#|     bc = df_all[(df_all[YEAR_COL] >= BACKCAST[0]) &
#|                 (df_all[YEAR_COL] <= BACKCAST[1])].dropna(subset=[TARGET])
#|     if len(bc):
#|         med_e = df[env].median()
#|         bm = best_ctor(); bm.fit(df[env].fillna(med_e), df[TARGET].values)
#|         pred = bm.predict(bc[env].fillna(med_e))
#|         from sklearn.metrics import r2_score as _r2
#|         M["backcast_r2"] = float(_r2(bc[TARGET].values, pred))
#|         M["backcast_n"] = int(len(bc))
#|         out = bc[[GROUP_COL, YEAR_COL, TARGET]].copy(); out["pred"] = pred
#|         B.table("backcast_predictions", out)
#|         print(f"    {BACKCAST[0]}-{BACKCAST[1]} env-only R2={M['backcast_r2']:.4f} "
#|               f"(n={len(bc):,})")
#|
#|     # -- final model ---------------------------------------------------------
#|     full_model, Xf, fdata, medf = core.fit_full_model(df, feats, best_ctor)
#|     B.model("final_model", dict(model=full_model, features=feats,
#|                                 median_fill=medf.to_dict(), window=WINDOW,
#|                                 model_name=best))
#|     M.update(part)
#|     B.json("metrics", M)
#|     B.close()
#|     print(f"\nSTUDY B complete in {(time.time()-t0)/60:.1f} min -> {args.outdir}/")
#|
#|
#| if __name__ == "__main__":
#|     main()
#|
#@@ END pipeline/pipeline_B_vanishing_v2.py
#@@ FILE pipeline/pipeline_core_v2.py
#| #!/usr/bin/env python3
#| # =============================================================================
#| #  pipeline_core_v2.py
#| #  Shared machinery for BOTH studies on the 1,318-column superset dataset
#| #  (now including ERA5 reanalysis and CAMS atmospheric composition).
#| #
#| #  WHAT CHANGED FROM v1, AND WHY
#| #  -----------------------------------------------------------------------
#| #  1. NEW MODALITIES. ERA5 (23 features, 2003-2019) and CAMS (97 features,
#| #     2003-2019, including formaldehyde, NO2, ozone, aerosol speciation).
#| #     Because both start in 2003, Study A now runs 2003-2019.
#| #  2. PER-FEATURE GATING IS NOW MANDATORY, not a nicety. In this dataset the
#| #     "Sentinel-5P & Particulates" label mixes pm25_* (2000-2019, keep) with
#| #     air_* (2019-2023, drop). Gating by modality would be wrong either way.
#| #  3. LEAKAGE BLOCKLIST EXPANDED. The superset contains seven columns that are
#| #     functions of mortality, including sdoh_chr_avg_life_expec, which IS the
#| #     target under another name. All are blocked before any modelling.
#| #  4. DEGENERATE-FEATURE REMOVAL. 22 columns carry <=2 distinct values across
#| #     74,617 rows (20 of them meta_canopy_cover_*). They are dropped.
#| #  5. MODEL TOURNAMENT. Instead of assuming LightGBM, several learners compete
#| #     under the identical protocol and SHAP runs on the winner.
#| #  6. FULL ARTEFACT CAPTURE. Every run writes the fitted model, out-of-fold
#| #     predictions, per-fold metrics, the SHAP ranking, the gate report, the
#| #     coverage report, and a machine-readable metrics JSON, so downstream
#| #     plotting and analysis never needs to re-run the model.
#| #
#| #  INVARIANTS (unchanged, and enforced)
#| #    * County-grouped cross-validation; all years of a county in one fold.
#| #    * Imputation fitted on the TRAIN fold only. No global imputation anywhere.
#| #    * Variance and collinearity filters never see the target.
#| #    * A protected list shields physiologically motivated features from pruning.
#| # =============================================================================
#|
#| from __future__ import annotations
#| import json, re, time, warnings
#| from pathlib import Path
#|
#| import numpy as np
#| import pandas as pd
#|
#| warnings.filterwarnings("ignore")
#|
#| # ---------------------------------------------------------------------------
#| #  COLUMN ROLES
#| # ---------------------------------------------------------------------------
#| TARGET = "meanlifeexpectency"
#| GROUP_COL = "fips"
#| YEAR_COL = "year"
#| ID_COLS = ["fips", "year", "statefp", "countyfp", "name", "geoid", "is_post_2015"]
#|
#| # ---------------------------------------------------------------------------
#| #  LEAKAGE BLOCKLIST
#| #  Columns that are functions of mortality. Including any of them lets the
#| #  model predict life expectancy from life expectancy. Non-negotiable.
#| # ---------------------------------------------------------------------------
#| LEAKAGE_EXACT = {
#|     "sdoh_chr_avg_life_expec",          # this IS the target, renamed
#|     "sdoh_chr_premat_death_rate",
#|     "sdoh_chr_infant_death_rate",
#|     "sdoh_chr_child_death_rate",
#|     "sdoh_chr_firearm_death_rate",
#|     "sdoh_chr_pct_alcohol_driv_death",
#|     "health_lung_cancer_mortality_rate",
#| }
#| LEAKAGE_REGEX = re.compile(
#|     r"(life_expec|lifeexpec|meanlife|_death_|death_rate|mortality|"
#|     r"yrs_life_lost|years_of_life|premature_death|infant_death|suicide_rate|"
#|     r"overdose|drug_death)", re.I)
#|
#|
#| def is_leakage(col: str) -> bool:
#|     return col in LEAKAGE_EXACT or bool(LEAKAGE_REGEX.search(col))
#|
#|
#| # ---------------------------------------------------------------------------
#| #  MODALITY REGISTRY  (prefix -> human label).  Longest prefix wins.
#| # ---------------------------------------------------------------------------
#| MODALITY_PREFIX = {
#|     "cams_": "CAMS atmospheric composition",
#|     "era5_": "ERA5 reanalysis",
#|     "prism_": "PRISM climate",
#|     "terra_": "TerraClimate",
#|     "lst_": "MODIS land-surface temperature",
#|     "ndvi_": "MODIS vegetation",
#|     "landsat_": "Landsat optical",
#|     "ls57_": "Landsat optical",
#|     "dem_": "Elevation and terrain",
#|     "ghm_": "Human modification",
#|     "jrc_": "Surface water",
#|     "meta_": "Canopy structure",
#|     "ntl_harmonized_": "Night-time lights",
#|     "ntl_": "Night-time lights (raw)",
#|     "pm25_": "Fine particulate matter",
#|     "pm1": "Particulate matter (other)",
#|     "air_": "Sentinel-5P air quality",
#|     "soil_": "Soil profile",
#|     "usda_": "Cropland and agriculture",
#|     "live_": "Livestock",
#|     "s1_": "Sentinel-1 radar",
#|     "s2_": "Sentinel-2 optical",
#|     "econ_": "Macro economics",
#|     "health_": "Health outcomes",
#|     "sdoh_": "Socioeconomic (SDOH)",
#|     "eng_": "Engineered (legacy)",
#|     "eng2_": "Engineered (v2)",
#| }
#| _PREFIXES_SORTED = sorted(MODALITY_PREFIX, key=len, reverse=True)
#|
#|
#| def modality_of(col: str) -> str:
#|     for p in _PREFIXES_SORTED:
#|         if col.startswith(p):
#|             return MODALITY_PREFIX[p]
#|     return "Unclassified"
#|
#|
#| # Modalities that are socioeconomic rather than environmental. Study A excludes
#| # these by design (its claim is "no census input"); Study B includes them.
#| SDOH_MODALITIES = {"Socioeconomic (SDOH)", "Macro economics", "Health outcomes"}
#|
#|
#| def is_sdoh(col: str) -> bool:
#|     return modality_of(col) in SDOH_MODALITIES
#|
#|
#| # ---------------------------------------------------------------------------
#| #  PROTECTED FEATURES
#| #  Physiologically motivated variables that must survive variance and
#| #  collinearity pruning, so that the question "does wet-bulb matter?" can
#| #  actually be asked rather than silently filtered away.
#| # ---------------------------------------------------------------------------
#| PROTECT_PATTERNS = [
#|     r"wet_?bulb", r"wbgt", r"heat_?index", r"humidex", r"discomfort",
#|     r"apparent_?temp", r"wind_?chill", r"tdmean", r"dew_?point", r"dewpoint",
#|     r"relative_humidity", r"specific_humidity", r"absolute_humidity",
#|     r"prism_tmin", r"tmin_min", r"night_cool", r"nightcool", r"vpd",
#|     r"diurnal", r"hot_?night", r"days_tmin_over", r"days_tmax_over",
#|     r"formaldehyde", r"nitrogen_dioxide", r"ozone", r"pm25_", r"pm2_5",
#|     r"total_column_water", r"lst_night",
#| ]
#| _PROTECT_RE = re.compile("|".join(PROTECT_PATTERNS), re.I)
#|
#|
#| def is_protected(col: str) -> bool:
#|     return bool(_PROTECT_RE.search(col))
#|
#|
#| # ---------------------------------------------------------------------------
#| #  LOADING AND CLEANING
#| # ---------------------------------------------------------------------------
#| def load_master(csv_path: str, add_engineered: bool = True, verbose: bool = True):
#|     """Load the superset CSV, normalise column names, drop degenerate and
#|     duplicated columns, and (optionally) add the v2 engineered layer."""
#|     t0 = time.time()
#|     df = pd.read_csv(csv_path, low_memory=False)
#|     df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
#|     if verbose:
#|         print(f"[load] {csv_path}: {df.shape[0]:,} rows x {df.shape[1]:,} cols "
#|               f"({time.time()-t0:.1f}s)")
#|
#|     # -- duplicate column names: keep the copy with the most observed values --
#|     dups = df.columns[df.columns.duplicated()].unique().tolist()
#|     if dups:
#|         keep_idx = []
#|         for name in df.columns.unique():
#|             block = df.loc[:, df.columns == name]
#|             if block.shape[1] == 1:
#|                 keep_idx.append(block)
#|             else:
#|                 best = block.iloc[:, int(np.argmax(block.notna().sum().values))]
#|                 keep_idx.append(best.to_frame(name))
#|         df = pd.concat(keep_idx, axis=1)
#|         if verbose:
#|             print(f"[load] resolved {len(dups)} duplicated column names "
#|                   f"(kept the copy with the most observed values)")
#|
#|     # -- contiguous US only, if the identifier allows it --
#|     if "statefp" in df.columns:
#|         before = len(df)
#|         df = df[~df["statefp"].isin([2, 15, 60, 66, 69, 72, 78])]
#|         if verbose and before != len(df):
#|             print(f"[load] dropped {before-len(df):,} non-contiguous rows")
#|
#|     if add_engineered:
#|         from engineered_features_v2 import add_engineered_features
#|         df, cov = add_engineered_features(df, verbose=verbose)
#|     return df
#|
#|
#| def degenerate_columns(df: pd.DataFrame, min_unique: int = 3) -> list[str]:
#|     """Columns carrying fewer than `min_unique` distinct observed values.
#|     In this dataset that catches the 20 broken meta_canopy_cover_* columns."""
#|     out = []
#|     for c in df.columns:
#|         if c in ID_COLS or c == TARGET:
#|             continue
#|         if not pd.api.types.is_numeric_dtype(df[c]):
#|             continue
#|         if df[c].nunique(dropna=True) < min_unique:
#|             out.append(c)
#|     return out
#|
#|
#| # ---------------------------------------------------------------------------
#| #  TEMPORAL-COVERAGE GATE  (computed from the data, not from metadata)
#| # ---------------------------------------------------------------------------
#| def coverage_gate(df, window, min_coverage=0.95, allow_sdoh=False,
#|                   drop_raw_ntl=True, extra_exclude=(), verbose=True):
#|     """Return (kept_features, report). A feature is admitted only if it is
#|     actually observed for at least `min_coverage` of the county-years inside
#|     the study window. Every decision is recorded with its reason."""
#|     y0, y1 = window
#|     win = df[(df[YEAR_COL] >= y0) & (df[YEAR_COL] <= y1)]
#|     if len(win) == 0:
#|         raise ValueError(f"no rows in window {y0}-{y1}")
#|
#|     degen = set(degenerate_columns(win))
#|     rows, kept = [], []
#|     for c in df.columns:
#|         if c in ID_COLS or c == TARGET:
#|             continue
#|         if not pd.api.types.is_numeric_dtype(df[c]):
#|             reason, ok = "non-numeric", False
#|         elif is_leakage(c):
#|             reason, ok = "LEAKAGE-outcome-derived", False
#|         elif c in extra_exclude:
#|             reason, ok = "explicitly-excluded", False
#|         elif c in degen:
#|             reason, ok = "degenerate-<3-unique-values", False
#|         elif (not allow_sdoh) and is_sdoh(c):
#|             reason, ok = "socioeconomic-not-in-scope", False
#|         elif drop_raw_ntl and c.startswith("ntl_") and not c.startswith("ntl_harmonized_"):
#|             reason, ok = "raw-NTL-superseded-by-harmonized", False
#|         else:
#|             cov = float(win[c].notna().mean())
#|             if cov >= min_coverage:
#|                 reason, ok = f"admitted ({cov:.0%})", True
#|             else:
#|                 reason, ok = f"coverage-below-threshold ({cov:.0%})", False
#|         cov = float(win[c].notna().mean()) if pd.api.types.is_numeric_dtype(df[c]) else 0.0
#|         rows.append(dict(feature=c, modality=modality_of(c), window_coverage=round(cov, 4),
#|                          admitted=ok, reason=reason, protected=is_protected(c)))
#|         if ok:
#|             kept.append(c)
#|
#|     report = pd.DataFrame(rows).sort_values(
#|         ["admitted", "modality", "window_coverage"], ascending=[False, True, False])
#|     if verbose:
#|         adm = report[report.admitted]
#|         print(f"[gate] window {y0}-{y1}: admitted {len(adm)} of {len(report)} features "
#|               f"across {adm.modality.nunique()} modalities")
#|         for r, n in report[~report.admitted].reason.str.replace(
#|                 r"\s*\(.*\)", "", regex=True).value_counts().items():
#|             print(f"        {n:>5} excluded: {r}")
#|         print(f"        {int(adm.protected.sum())} protected features retained")
#|     return kept, report
#|
#|
#| # ---------------------------------------------------------------------------
#| #  UNSUPERVISED REDUNDANCY FILTER  (never sees the target)
#| # ---------------------------------------------------------------------------
#| def redundancy_filter(X: pd.DataFrame, corr_threshold=0.98, var_threshold=1e-8,
#|                       verbose=True):
#|     protected = {c for c in X.columns if is_protected(c)}
#|
#|     var = X.var(numeric_only=True, skipna=True)
#|     keep = [c for c in X.columns if c in protected or var.get(c, 0) > var_threshold]
#|     n_var = len(X.columns) - len(keep)
#|
#|     sub = X[keep]
#|     corr = sub.corr().abs()
#|     cols = list(sub.columns)
#|     dropped = set()
#|     for i, a in enumerate(cols):
#|         if a in dropped:
#|             continue
#|         col_a = corr.loc[a]
#|         for b in cols[i + 1:]:
#|             if b in dropped:
#|                 continue
#|             if col_a.get(b, 0) >= corr_threshold:
#|                 if a in protected and b in protected:
#|                     continue
#|                 victim = b if b not in protected else a
#|                 if victim not in protected:
#|                     dropped.add(victim)
#|     keep = [c for c in cols if c not in dropped]
#|
#|     lost = [c for c in protected if c not in keep]
#|     assert not lost, f"PROTECTED FEATURE DROPPED: {lost}"
#|     if verbose:
#|         print(f"[filter] variance-dropped {n_var}, collinearity-dropped {len(dropped)}, "
#|               f"kept {len(keep)} ({len(protected & set(keep))} protected)")
#|     return keep
#|
#|
#| # ---------------------------------------------------------------------------
#| #  MODEL ZOO
#| # ---------------------------------------------------------------------------
#| def build_models(use_gpu=False, seed=0, fast=False):
#|     """Return {name: (constructor, supports_treeshap)}. Constructors are
#|     callables so each fold gets a fresh, unfitted estimator."""
#|     n_est = 400 if fast else 1200
#|     zoo = {}
#|
#|     try:
#|         import lightgbm as lgb
#|         def _lgb():
#|             kw = dict(n_estimators=n_est, learning_rate=0.03, num_leaves=127,
#|                       min_child_samples=40, subsample=0.8, subsample_freq=1,
#|                       colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1,
#|                       random_state=seed, verbosity=-1)
#|             if use_gpu:
#|                 kw.update(device_type="gpu")
#|             return lgb.LGBMRegressor(**kw)
#|         zoo["LightGBM"] = (_lgb, True)
#|     except ImportError:
#|         pass
#|
#|     try:
#|         import xgboost as xgb
#|         def _xgb():
#|             kw = dict(n_estimators=n_est, learning_rate=0.03, max_depth=8,
#|                       min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
#|                       reg_lambda=1.0, n_jobs=-1, random_state=seed,
#|                       tree_method="hist", enable_categorical=False)
#|             if use_gpu:
#|                 kw.update(device="cuda")
#|             return xgb.XGBRegressor(**kw)
#|         zoo["XGBoost"] = (_xgb, True)
#|     except ImportError:
#|         pass
#|
#|     try:
#|         from catboost import CatBoostRegressor
#|         def _cat():
#|             kw = dict(iterations=n_est, learning_rate=0.03, depth=8,
#|                       l2_leaf_reg=3.0, random_seed=seed, verbose=0,
#|                       allow_writing_files=False)
#|             if use_gpu:
#|                 kw.update(task_type="GPU")
#|             return CatBoostRegressor(**kw)
#|         zoo["CatBoost"] = (_cat, True)
#|     except ImportError:
#|         pass
#|
#|     if not zoo:
#|         from sklearn.ensemble import HistGradientBoostingRegressor
#|         zoo["HistGB"] = (lambda: HistGradientBoostingRegressor(
#|             max_iter=n_est, learning_rate=0.03, random_state=seed), False)
#|     return zoo
#|
#|
#| # ---------------------------------------------------------------------------
#| #  CROSS-VALIDATION  (county-grouped, in-fold imputation only)
#| # ---------------------------------------------------------------------------
#| def as_float(X: pd.DataFrame) -> pd.DataFrame:
#|     """Coerce every column to float64. Without this, integer-typed columns can
#|     trigger categorical splits in some boosters, which TreeSHAP cannot read."""
#|     return X.astype("float64")
#|
#|
#| def cv_evaluate(df, features, model_ctor, n_folds=8, seed=0, return_models=False):
#|     from sklearn.model_selection import GroupKFold
#|     from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#|
#|     data = df.dropna(subset=[TARGET]).reset_index(drop=True)
#|     X = as_float(data[features]); y = data[TARGET].values; g = data[GROUP_COL].values
#|     oof = np.full(len(data), np.nan)
#|     fold_rows, models = [], []
#|
#|     for k, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, g), 1):
#|         med = X.iloc[tr].median(numeric_only=True)      # TRAIN FOLD ONLY
#|         Xtr, Xte = X.iloc[tr].fillna(med), X.iloc[te].fillna(med)
#|         m = model_ctor()
#|         m.fit(Xtr, y[tr])
#|         pred = m.predict(Xte)
#|         oof[te] = pred
#|         fold_rows.append(dict(fold=k, n_train=len(tr), n_test=len(te),
#|                               r2=r2_score(y[te], pred),
#|                               mae=mean_absolute_error(y[te], pred),
#|                               rmse=float(mean_squared_error(y[te], pred) ** 0.5)))
#|         if return_models:
#|             models.append((m, med))
#|
#|     metrics = dict(r2=float(r2_score(y, oof)),
#|                    mae=float(mean_absolute_error(y, oof)),
#|                    rmse=float(mean_squared_error(y, oof) ** 0.5),
#|                    pearson_r=float(np.corrcoef(y, oof)[0, 1]),
#|                    within_1yr=float(np.mean(np.abs(y - oof) <= 1) * 100),
#|                    within_2yr=float(np.mean(np.abs(y - oof) <= 2) * 100),
#|                    n_obs=int(len(data)), n_features=len(features))
#|     folds = pd.DataFrame(fold_rows)
#|     metrics["r2_fold_mean"] = float(folds.r2.mean())
#|     metrics["r2_fold_std"] = float(folds.r2.std())
#|     out = dict(data=data, oof=oof, metrics=metrics, folds=folds, features=features)
#|     if return_models:
#|         out["models"] = models
#|     return out
#|
#|
#| def bootstrap_r2_ci(y, pred, n_boot=1000, seed=0):
#|     from sklearn.metrics import r2_score
#|     rng = np.random.default_rng(seed)
#|     idx = np.arange(len(y))
#|     vals = [r2_score(y[b], pred[b])
#|             for b in (rng.choice(idx, len(idx), replace=True) for _ in range(n_boot))]
#|     return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
#|
#|
#| def run_tournament(df, features, n_folds=8, use_gpu=False, seed=0, fast=False,
#|                    verbose=True):
#|     """Fit every available learner under the identical protocol and return the
#|     per-learner table plus the name of the winner by out-of-fold R^2."""
#|     zoo = build_models(use_gpu=use_gpu, seed=seed, fast=fast)
#|     rows, results = [], {}
#|     
#|     for name, (ctor, shap_ok) in zoo.items():
#|         t0 = time.time()
#|         if verbose:
#|             print(f"  [tournament] evaluating {name:<10} ...", end="\r", flush=True)
#|             
#|         try:
#|             res = cv_evaluate(df, features, ctor, n_folds=n_folds, seed=seed)
#|             res["shap_ok"] = shap_ok
#|             results[name] = res
#|             m = res["metrics"]
#|             rows.append(dict(model=name, r2=m["r2"], mae=m["mae"], rmse=m["rmse"],
#|                              r2_fold_mean=m["r2_fold_mean"], r2_fold_std=m["r2_fold_std"],
#|                              seconds=round(time.time() - t0, 1), treeshap=shap_ok))
#|             if verbose:
#|                 # Overwrite the evaluating line with the success metrics
#|                 print(f"  [tournament] {name:<10} R2={m['r2']:.4f} "
#|                       f"MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
#|                       f"({time.time()-t0:.0f}s)")
#|                 
#|         except Exception as e:
#|             if verbose:
#|                 # Extract just the first line of the error so it doesn't flood the logs
#|                 err_msg = str(e).splitlines()[0] if str(e) else "Unknown Error"
#|                 print(f"  [tournament] {name:<10} FAILED: {err_msg} (Skipping)")
#|             continue # Gracefully skip to the next model
#|             
#|     if not rows:
#|         raise RuntimeError("All models in the tournament failed!")
#|         
#|     table = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
#|     best = table.iloc[0]["model"]
#|     if verbose:
#|         print(f"  [tournament] winner: {best} (R2={table.iloc[0]['r2']:.4f})")
#|     return table, results, best
#|
#|
#| # ---------------------------------------------------------------------------
#| #  ATTRIBUTION
#| # ---------------------------------------------------------------------------
#| def fit_full_model(df, features, model_ctor):
#|     """Refit on all rows with a global median fill, for downstream attribution
#|     and dependence analysis. Never used to report skill."""
#|     data = df.dropna(subset=[TARGET]).reset_index(drop=True)
#|     X = as_float(data[features])
#|     med = X.median(numeric_only=True)
#|     Xf = X.fillna(med)
#|     m = model_ctor()
#|     m.fit(Xf, data[TARGET].values)
#|     return m, Xf, data, med
#|
#|
#| def interventional_shap(df, features, model_ctor, n_folds=8, bg_size=1000,
#|                         sample_size=4000, seed=0, verbose=True):
#|     """Out-of-fold interventional TreeSHAP. Background and explainer are built
#|     from the TRAINING fold only, so attribution inherits the leakage control."""
#|     import shap
#|     from sklearn.model_selection import GroupKFold
#|
#|     data = df.dropna(subset=[TARGET]).reset_index(drop=True)
#|     X = as_float(data[features]); y = data[TARGET].values; g = data[GROUP_COL].values
#|     acc = np.zeros(len(features))
#|     n_used = 0
#|     per_obs_idx, per_obs_val = [], []
#|
#|     for k, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, g), 1):
#|         med = X.iloc[tr].median(numeric_only=True)
#|         Xtr, Xte = X.iloc[tr].fillna(med), X.iloc[te].fillna(med)
#|         m = model_ctor()
#|         m.fit(Xtr, y[tr])
#|         bg = Xtr.sample(min(bg_size, len(Xtr)), random_state=seed)
#|         ex = shap.TreeExplainer(m, data=bg, feature_perturbation="interventional")
#|         samp = Xte.sample(min(sample_size, len(Xte)), random_state=seed)
#|         sv = ex.shap_values(samp, check_additivity=False)
#|         acc += np.abs(sv).mean(axis=0)
#|         n_used += 1
#|         per_obs_idx.append(samp.index.values)
#|         per_obs_val.append(sv)
#|         if verbose:
#|             print(f"  [shap] fold {k}/{n_folds}")
#|
#|     ranking = (pd.DataFrame({"feature": features, "mean_abs_shap": acc / max(n_used, 1)})
#|                .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
#|     ranking["rank"] = ranking.index + 1
#|     ranking["modality"] = ranking.feature.map(modality_of)
#|     ranking["protected"] = ranking.feature.map(is_protected)
#|
#|     idx = np.concatenate(per_obs_idx)
#|     vals = np.vstack(per_obs_val)
#|     per_obs = pd.DataFrame(vals, columns=features, index=idx)
#|     return ranking, per_obs, data
#|
#|
#| # ---------------------------------------------------------------------------
#| #  ARTEFACTS
#| # ---------------------------------------------------------------------------
#| class Artifacts:
#|     """Every run writes the same set of files, so downstream plotting and
#|     analysis code never has to re-fit anything."""
#|
#|     def __init__(self, outdir: str, tag: str):
#|         self.dir = Path(outdir); self.dir.mkdir(parents=True, exist_ok=True)
#|         self.tag = tag
#|         self.manifest = {}
#|
#|     def _p(self, name): return self.dir / f"{self.tag}_{name}"
#|
#|     def table(self, name, df, index=False):
#|         p = self._p(name if name.endswith(".csv") else name + ".csv")
#|         df.to_csv(p, index=index)
#|         self.manifest[name] = str(p)
#|         print(f"    [saved] {p.name}  ({len(df):,} rows)")
#|         return p
#|
#|     def frame(self, name, df):
#|         p = self._p(name if name.endswith(".parquet") else name + ".parquet")
#|         df.to_parquet(p, index=False)
#|         self.manifest[name] = str(p)
#|         print(f"    [saved] {p.name}  ({df.shape[0]:,} x {df.shape[1]:,})")
#|         return p
#|
#|     def model(self, name, obj):
#|         import joblib
#|         p = self._p(name if name.endswith(".joblib") else name + ".joblib")
#|         joblib.dump(obj, p, compress=3)
#|         self.manifest[name] = str(p)
#|         print(f"    [saved] {p.name}")
#|         return p
#|
#|     def json(self, name, obj):
#|         p = self._p(name if name.endswith(".json") else name + ".json")
#|         with open(p, "w") as fh:
#|             json.dump(obj, fh, indent=2, default=str)
#|         self.manifest[name] = str(p)
#|         print(f"    [saved] {p.name}")
#|         return p
#|
#|     def close(self):
#|         self.json("manifest", self.manifest)
#|
#@@ END pipeline/pipeline_core_v2.py
#@@ FILE pipeline/run_ml_satallite_and_SDOH.py
#| #!/usr/bin/env python3
#| """
#| ================================================================================
#|  BIGBANG v3 — CHECKPOINTED, LEAKAGE-FREE ML PIPELINE FOR LIFE-EXPECTANCY PAPER
#| ================================================================================
#|  Input   : Master_Cleaned_NoArtifacts_With_LIVE_Prefix.csv (or any county-year CSV)
#|  Target  : MeanLifeExpectency
#|  Output  : RESULTS_<stamp>/  — every metric, prediction, residual, model, SHAP
#|            value, interval, OOD score, stability table, and statistical test a
#|            reviewer could ask for, plus manuscript-compatible CSVs for 03_plots.py.
#|
#|  WHAT v3 ADDS OVER v2 (the "never re-run from scratch again" release)
#|  ─────────────────────────────────────────────────────────────────────────────
#|  1. STAGE CHECKPOINTING + RESUME. Every expensive stage persists its result to
#|     checkpoints/<stage>.pkl guarded by a config-hash marker. Re-running the
#|     script skips completed stages. Crash at Step 14 → restart resumes at 14.
#|        --rerun stageA,stageB   force specific stages (downstream auto-invalidated)
#|        --fresh                 ignore all checkpoints, start clean
#|     Changing any scientific config (year range, leakage list, folds, seed,
#|     thresholds, data file) changes the hash and auto-invalidates stale caches.
#|  2. EXHAUSTIVE ARTIFACT SAVING at every stage: per-fold predictions WITH
#|     residuals + train metrics + timing, fold-assignment table (GEOID→fold),
#|     split indices (.npz), holdout GEOID list, feature lists at every pruning
#|     stage, correlation-pruning audit, config manifest with dataset SHA256.
#|  3. STATISTICAL TESTING SUITE: bootstrap CI on winner OOF R², permutation test
#|     (reuses preprocessed fold matrices — valid because preprocessing never
#|     touches y — so it is ~10× faster), and paired Wilcoxon on per-row |error|
#|     between the top models.
#|  4. Everything from v2 kept: honest 20% GEOID holdout, anchor-protected
#|     collinearity pruning, precomputed fold splits, CONCURRENT model tuning,
#|     quantile intervals + coverage check, OOD (PCA+Mahalanobis) for 2020-2023,
#|     SHAP fold-stability (Jaccard/Spearman/CV), top-3 inverse-RMSE ensemble,
#|     forward-chaining temporal CV, static aggregation of short-coverage
#|     modalities, diagnostic tables (bracket / temporal / county / residual).
#|  5. OUR locked scientific decisions preserved (see 01_RESULTS_LEDGER.md):
#|     - HARD leakage blocklist: exact LE-proxy names + regex net (v2's regexes
#|       alone would MISS SDOH_CHR_YRS_LIFE_LOST — ours catch it).
#|     - Default window 2009-2019 (SDOH availability); configurable via CLI.
#|     - Headline ablation SDOH vs Env vs Combined + smoothed-SDOH IHME baseline
#|       + incremental modality leaderboard.
#|     - FAST mode uses the previously-tuned gold-mine params (skips search);
#|       FULL mode re-tunes with the concurrent tournament.
#|
#|  USAGE
#|  ─────────────────────────────────────────────────────────────────────────────
#|    python 02_pipeline.py --data master.csv                       # FAST, resumable
#|    python 02_pipeline.py --data master.csv --mode full           # full re-tune
#|    python 02_pipeline.py --data master.csv --outdir RESULTS_X    # resume that run
#|    python 02_pipeline.py --data master.csv --outdir RESULTS_X --rerun shap,tables
#|    python 02_pipeline.py --data master.csv --test-mode           # 3-5 min smoke test
#|    BIGBANG_N_JOBS=32 python 02_pipeline.py --data master.csv     # thread override
#|
#|  HPC:  sbatch --cpus-per-task=64 --mem=256G bigbang.sh   (python -u 02_pipeline.py ...)
#| ================================================================================
#| """
#| from __future__ import annotations
#|
#| # =============================================================================
#| # 0. SINGLE-TIER THREAD CONTROL  (must precede numpy/sklearn/booster imports)
#| # =============================================================================
#| import os
#| _DEFAULT_N_JOBS = max(1, (os.cpu_count() or 2) - 1)
#| N_JOBS = max(1, int(os.getenv("BIGBANG_N_JOBS", os.getenv("N_JOBS", str(_DEFAULT_N_JOBS)))))
#| _BLAS_THREADS = max(1, N_JOBS // 2)
#| for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
#|            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
#|     os.environ.setdefault(_v, str(_BLAS_THREADS))
#|
#| import sys, gc, json, pickle, re, time, warnings, traceback, logging, signal
#| import platform, hashlib, argparse, random, subprocess
#| from pathlib import Path
#| from datetime import datetime
#| from collections import Counter
#| from dataclasses import dataclass, field, asdict
#|
#| import numpy as np
#| import pandas as pd
#| from scipy import stats as sp_stats
#|
#| from sklearn.ensemble import RandomForestRegressor
#| from sklearn.model_selection import GroupKFold
#| from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#| from sklearn.decomposition import PCA
#| from sklearn.covariance import MinCovDet, EmpiricalCovariance
#| from joblib import Parallel, delayed
#|
#| try:
#|     import xgboost as xgb;      XGB_OK = True
#| except ImportError:             xgb = None; XGB_OK = False
#| try:
#|     import lightgbm as lgb;     LGB_OK = True
#| except ImportError:             lgb = None; LGB_OK = False
#| try:
#|     import catboost as cbst;    CB_OK = True
#| except ImportError:             cbst = None; CB_OK = False
#| try:
#|     import shap;                SHAP_OK = True
#| except ImportError:             shap = None; SHAP_OK = False
#|
#| warnings.filterwarnings("ignore")
#|
#| MODEL_AVAILABILITY = {"RF": True, "XGB": XGB_OK, "LGB": LGB_OK, "CB": CB_OK}
#| MODEL_LABELS = {"RF": "RandomForest", "XGB": "XGBoost",
#|                 "LGB": "LightGBM", "CB": "CatBoost"}
#|
#| # =============================================================================
#| # 1. CONFIG  (CLI-driven dataclass; its hash guards every checkpoint)
#| # =============================================================================
#| TARGET_COL = "MeanLifeExpectency"
#| ID_COLS    = ["GEOID", "year", "STATEFP", "COUNTYFP", "NAME", "location_name", "fips"]
#| EXCLUDE_STATEFP = {"02", "15", "60", "66", "69", "72", "78"}   # CONUS only
#|
#| # --- HARD leakage blocklist: exact names (ledger §4.1) + regex safety net.
#| #     NOTE: v2's regexes (^yll etc.) would MISS SDOH_CHR_YRS_LIFE_LOST because
#| #     of the prefix — the union below cannot.
#| LEAKAGE_EXACT = {
#|     "SDOH_CHR_AVG_LIFE_EXPEC",
#|     "SDOH_CHR_YRS_LIFE_LOST", "SDOH_CHR_YRS_LIFE_LOST_AIAN",
#|     "SDOH_CHR_YRS_LIFE_LOST_ASIAN", "SDOH_CHR_YRS_LIFE_LOST_BLACK",
#|     "SDOH_CHR_YRS_LIFE_LOST_HISPANIC", "SDOH_CHR_YRS_LIFE_LOST_WHITE",
#|     "SDOH_CHR_PREMAT_DEATH_RATE", "SDOH_CHR_INFANT_DEATH_RATE",
#|     "SDOH_CHR_CHILD_DEATH_RATE",
#| }
#| LEAKAGE_REGEXES = [
#|     r"LIFE_EXPEC", r"YRS_LIFE_LOST", r"DEATH_RATE", r"MORTALITY", r"PREMAT",
#|     r"life.*expect", r"expect.*life", r"(^|_)yll", r"(^|_)yld", r"(^|_)daly",
#|     r"all_cause_death",
#| ]
#| _LEAK_RE = [re.compile(p, re.I) for p in LEAKAGE_REGEXES]
#|
#| def is_leakage(col: str) -> bool:
#|     if col == TARGET_COL:
#|         return False
#|     if col in LEAKAGE_EXACT:
#|         return True
#|     return any(rx.search(col) for rx in _LEAK_RE)
#|
#| # --- residual livestock aggregates that survive dataset rebuilds
#| EXPLICIT_DROP_COLS = {"total_area_km2", "total_weighted_mean",
#|                       "avg_weighted_mean", "total_head_count"}
#|
#| # --- gold-mine hyperparameters from prior Optuna runs (ledger §3) — FAST mode
#| BEST_PARAMS_GOLD = {
#|     "XGB": dict(n_estimators=1500, max_depth=8, learning_rate=0.013832442448846112,
#|                 subsample=0.7311989040672405, colsample_bytree=0.7116167224336398,
#|                 min_child_weight=9, reg_alpha=0.2537815508265665,
#|                 reg_lambda=0.679657809075816),
#|     "CB":  dict(iterations=2000, depth=8, learning_rate=0.051178145982369534,
#|                 l2_leaf_reg=2.7970640394252375, random_strength=0.11400863701127326),
#|     "RF":  dict(n_estimators=1000, max_depth=25, min_samples_leaf=2, max_features=0.3),
#|     "LGB": dict(n_estimators=1500, num_leaves=127, learning_rate=0.02,
#|                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
#| }
#|
#| @dataclass
#| class Config:
#|     data_path: str
#|     outdir: str
#|     mode: str = "fast"                    # fast (gold params) | full (re-tune)
#|     test_mode: bool = False
#|     seed: int = 42
#|     n_folds: int = 5
#|     n_iter_search: int = 40
#|     year_start: int = 2009               # OUR locked default (SDOH availability)
#|     year_end: int = 2019
#|     forecast_start: int = 2020
#|     forecast_end: int = 2023
#|     holdout_frac: float = 0.20
#|     max_missing_frac: float = 0.50
#|     nzv_thresh: float = 1e-8
#|     corr_thresh: float = 0.97
#|     importance_prune_thr: float = 0.001
#|     static_modality_thresh: float = 0.70
#|     shap_sample: int = 5000
#|     shap_bg: int = 500
#|     shap_topk: int = 30
#|     ood_pca: int = 20
#|     ood_quantile: float = 0.99
#|     bootstrap_n: int = 1000
#|     perm_n: int = 30
#|     loso_max_states: int = 10            # capped in fast; full = all
#|     exclude_prefixes: tuple = ("LIVE_",) # S1/S2/AIR handled by static-agg, not dropped
#|     rerun: list = field(default_factory=list)
#|     fresh: bool = False
#|     skip: list = field(default_factory=list)
#|
#|     def apply_test_mode(self):
#|         if self.test_mode:
#|             self.n_folds = 2; self.n_iter_search = 3
#|             self.shap_sample = 200; self.shap_bg = 50
#|             self.bootstrap_n = 100; self.perm_n = 3
#|             self.loso_max_states = 3
#|
#|     def science_hash(self) -> str:
#|         """Hash of everything that changes the SCIENCE (not runtime knobs).
#|         A checkpoint is valid only if this hash matches."""
#|         keys = ["data_path", "mode", "test_mode", "seed", "n_folds",
#|                 "year_start", "year_end", "forecast_start", "forecast_end",
#|                 "holdout_frac", "max_missing_frac", "nzv_thresh", "corr_thresh",
#|                 "importance_prune_thr", "static_modality_thresh",
#|                 "exclude_prefixes"]
#|         d = {k: getattr(self, k) for k in keys}
#|         d["leakage_exact"] = sorted(LEAKAGE_EXACT)
#|         d["leakage_regex"] = LEAKAGE_REGEXES
#|         d["target"] = TARGET_COL
#|         try:
#|             st = os.stat(self.data_path)
#|             d["data_sig"] = f"{st.st_size}_{int(st.st_mtime)}"
#|         except OSError:
#|             d["data_sig"] = "missing"
#|         return hashlib.sha256(json.dumps(d, sort_keys=True, default=str)
#|                               .encode()).hexdigest()[:16]
#|
#|
#| def parse_args() -> Config:
#|     ap = argparse.ArgumentParser(description="BIGBANG v3 checkpointed LE pipeline")
#|     ap.add_argument("--data", required=True)
#|     ap.add_argument("--outdir", default=None,
#|                     help="reuse an existing RESULTS dir to RESUME it")
#|     ap.add_argument("--mode", choices=["fast", "full"], default="fast")
#|     ap.add_argument("--test-mode", action="store_true")
#|     ap.add_argument("--seed", type=int, default=42)
#|     ap.add_argument("--year-start", type=int, default=2009)
#|     ap.add_argument("--year-end", type=int, default=2019)
#|     ap.add_argument("--rerun", default="", help="comma list of stages to force")
#|     ap.add_argument("--skip", default="", help="comma list of stages to skip")
#|     ap.add_argument("--fresh", action="store_true", help="ignore all checkpoints")
#|     ap.add_argument("--list-stages", action="store_true")
#|     a = ap.parse_args()
#|     outdir = a.outdir or f"RESULTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#|     cfg = Config(data_path=a.data, outdir=outdir, mode=a.mode,
#|                  test_mode=a.test_mode, seed=a.seed,
#|                  year_start=a.year_start, year_end=a.year_end,
#|                  rerun=[s for s in a.rerun.split(",") if s],
#|                  skip=[s for s in a.skip.split(",") if s],
#|                  fresh=a.fresh)
#|     cfg.apply_test_mode()
#|     cfg._list_stages = a.list_stages
#|     return cfg
#|
#| # =============================================================================
#| # 2. LOGGING
#| # =============================================================================
#| logger = logging.getLogger("bigbang")
#|
#| def setup_logging(outdir: Path):
#|     logger.setLevel(logging.INFO); logger.handlers.clear()
#|     fmt = logging.Formatter("[%(asctime)s] [%(levelname)-5s] %(message)s",
#|                             datefmt="%Y-%m-%d %H:%M:%S")
#|     fh = logging.FileHandler(outdir / "bigbang_pipeline.log", mode="a")
#|     sh = logging.StreamHandler(sys.stdout)
#|     fh.setFormatter(fmt); sh.setFormatter(fmt)
#|     logger.addHandler(fh); logger.addHandler(sh)
#|
#| def log(msg, level="INFO"): getattr(logger, level.lower(), logger.info)(msg)
#|
#| def log_sep(title=""):
#|     log("=" * 78)
#|     if title: log(f"  {title.upper()}"); log("=" * 78)
#|
#| def log_exc(prefix="Unhandled exception"):
#|     log(f"{prefix}:\n{traceback.format_exc()}", "ERROR")
#|
#|
#| # =============================================================================
#| # 3. CHECKPOINT MANAGER — the "never start over" machinery
#| # =============================================================================
#| # Design:
#| #   • Each stage is a named function returning a picklable dict.
#| #   • mgr.run("name", fn) → if checkpoints/name.pkl exists AND its marker's
#| #     science_hash matches the current config AND the stage isn't forced,
#| #     the pickle is loaded and the stage skipped.
#| #   • Stages form a LINEAR dependency chain: once any stage actually
#| #     recomputes, every later stage is invalidated ("dirty cascade") — this
#| #     prevents a forced early stage from silently feeding stale caches
#| #     downstream.
#| #   • Artifacts (CSVs, models) are written inside stage functions; on resume
#| #     they already exist on disk, so nothing is lost.
#| class CheckpointManager:
#|     def __init__(self, outdir: Path, cfg: Config):
#|         self.dir = outdir / "checkpoints"
#|         self.dir.mkdir(parents=True, exist_ok=True)
#|         self.hash = cfg.science_hash()
#|         self.force = set(cfg.rerun)
#|         self.skip = set(cfg.skip)
#|         self.fresh = cfg.fresh
#|         self.dirty = False          # once True, all later stages recompute
#|         self.timings = []
#|
#|     def _paths(self, name):
#|         return self.dir / f"{name}.pkl", self.dir / f"{name}.done.json"
#|
#|     def _valid(self, name) -> bool:
#|         pkl, marker = self._paths(name)
#|         if self.fresh or self.dirty or (name in self.force):
#|             return False
#|         if not (pkl.exists() and marker.exists()):
#|             return False
#|         try:
#|             meta = json.loads(marker.read_text())
#|             return meta.get("science_hash") == self.hash
#|         except Exception:
#|             return False
#|
#|     def run(self, name: str, fn, *args, **kwargs):
#|         if name in self.skip:
#|             log(f"⏭  Stage [{name}] SKIPPED by --skip")
#|             return None
#|         pkl, marker = self._paths(name)
#|         if self._valid(name):
#|             t0 = time.time()
#|             with open(pkl, "rb") as f:
#|                 out = pickle.load(f)
#|             log(f"⏩ Stage [{name}] restored from checkpoint "
#|                 f"({time.time()-t0:.1f}s load) — skipping compute")
#|             self.timings.append({"stage": name, "seconds": 0.0, "cached": True})
#|             return out
#|         # compute
#|         reason = ("forced" if name in self.force else
#|                   "downstream-of-recomputed" if self.dirty else
#|                   "fresh" if self.fresh else "no valid checkpoint")
#|         log_sep(f"STAGE [{name}]  ({reason})")
#|         t0 = time.time()
#|         out = fn(*args, **kwargs)
#|         dt = time.time() - t0
#|         try:
#|             with open(pkl, "wb") as f:
#|                 pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
#|             marker.write_text(json.dumps({
#|                 "science_hash": self.hash, "completed": datetime.now().isoformat(),
#|                 "seconds": round(dt, 1)}))
#|         except Exception:
#|             log_exc(f"checkpoint save failed for [{name}] (stage output still returned)")
#|         self.dirty = True   # everything after a recomputed stage must recompute
#|         self.timings.append({"stage": name, "seconds": round(dt, 1), "cached": False})
#|         log(f"✔ Stage [{name}] complete in {dt:.1f}s (checkpointed)")
#|         return out
#|
#|     def save_timing_report(self, outdir: Path):
#|         pd.DataFrame(self.timings).to_csv(outdir / "stage_timings.csv", index=False)
#|
#|
#| # =============================================================================
#| # 4. DEVICE DETECTION  (CUDA probe with timeout; env overrides)
#| # =============================================================================
#| def _probe_with_timeout(fn, timeout_s=15, fallback="cpu"):
#|     if not hasattr(signal, "SIGALRM"):
#|         try: return fn()
#|         except Exception: return fallback
#|     def _h(s, f): raise TimeoutError()
#|     old = signal.signal(signal.SIGALRM, _h); signal.alarm(timeout_s)
#|     try: return fn()
#|     except Exception: return fallback
#|     finally: signal.alarm(0); signal.signal(signal.SIGALRM, old)
#|
#| def _has_nvidia() -> bool:
#|     import shutil
#|     if shutil.which("nvidia-smi") is None: return False
#|     try:
#|         r = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
#|                            text=True, timeout=5)
#|         return r.returncode == 0 and "GPU" in (r.stdout or "")
#|     except Exception:
#|         return False
#|
#| def detect_devices(seed: int):
#|     xgb_dev, cb_task = "cpu", "CPU"
#|     user = os.getenv("BIGBANG_DEVICE", "").lower()
#|     if XGB_OK:
#|         u = os.getenv("BIGBANG_XGB_DEVICE", user).lower()
#|         if u in ("cpu", "cuda"): xgb_dev = u
#|         elif _has_nvidia():
#|             def _p():
#|                 m = xgb.XGBRegressor(tree_method="hist", device="cuda",
#|                                      n_estimators=2, verbosity=0,
#|                                      random_state=seed, n_jobs=1)
#|                 m.fit(np.random.rand(20, 3), np.random.rand(20))
#|                 m.predict(np.random.rand(5, 3)); return "cuda"
#|             xgb_dev = _probe_with_timeout(_p, 15)
#|     if CB_OK:
#|         u = os.getenv("BIGBANG_CB_DEVICE", user).upper()
#|         if u == "CUDA": u = "GPU"
#|         if u in ("CPU", "GPU"): cb_task = u
#|         elif _has_nvidia():
#|             def _p():
#|                 m = cbst.CatBoostRegressor(iterations=2, task_type="GPU",
#|                                            verbose=False, allow_writing_files=False)
#|                 m.fit(np.random.rand(40, 3), np.random.rand(40)); return "GPU"
#|             cb_task = _probe_with_timeout(_p, 20, fallback="CPU")
#|     return xgb_dev, cb_task
#|
#| XGB_DEVICE, CB_TASK_TYPE = "cpu", "CPU"     # set properly in main()
#|
#| # =============================================================================
#| # 5. MODALITY DEFINITIONS  (order: specific prefixes before general)
#| # =============================================================================
#| MODALITY_PREFIX_MAP = {
#|     "USDA_Cropland": ["USDA_"], "JRC_Water": ["JRC_"], "DEM_Topography": ["DEM_"],
#|     "Landsat_OLI": ["Landsat_"], "Sentinel2": ["S2_"], "Sentinel1_SAR": ["S1_"],
#|     "Landsat_LS57": ["LS57_"], "MODIS_LST": ["LST_"], "MODIS_NDVI": ["NDVI_EVI_", "NDVI_"],
#|     "Engineered": ["ENG_"], "AirQuality": ["AIR_"], "Economic": ["ECON_"],
#|     "Human_Modification": ["GHM_"], "Meta_Canopy": ["META_canopy_", "META_"],
#|     "PRISM_Climate": ["PRISM_"], "Soil_Texture": ["SOIL_"], "Soil_USDA": ["Soil_"],
#|     "TerraClimate": ["TERRA_"], "NightLights": ["NTL_"], "PM25": ["PM25_"],
#|     "SDOH": ["SDOH_"],
#| }
#| MODALITY_YEAR_RANGES = {
#|     "USDA_Cropland": (2000, 2019), "JRC_Water": (2000, 2019), "DEM_Topography": (2000, 2019),
#|     "Landsat_OLI": (2013, 2019), "Sentinel2": (2015, 2019), "Sentinel1_SAR": (2015, 2019),
#|     "Landsat_LS57": (2000, 2019), "MODIS_LST": (2000, 2019), "MODIS_NDVI": (2000, 2019),
#|     "Engineered": (2000, 2019), "AirQuality": (2019, 2023), "Economic": (2014, 2023),
#|     "Human_Modification": (2000, 2023), "Meta_Canopy": (2000, 2023),
#|     "PRISM_Climate": (2000, 2023), "Soil_Texture": (2000, 2023), "Soil_USDA": (2000, 2019),
#|     "TerraClimate": (2000, 2023), "NightLights": (2012, 2023), "PM25": (2000, 2019),
#|     "SDOH": (2009, 2019),
#| }
#| # High-level groups for the headline SDOH-vs-Env story
#| ENV_MODALITIES = {m for m in MODALITY_PREFIX_MAP if m != "SDOH"}
#|
#| def assign_modality(col: str) -> str:
#|     for mod, prefixes in MODALITY_PREFIX_MAP.items():
#|         for p in prefixes:
#|             if col.startswith(p): return mod
#|     return "Other"
#|
#| def overlap_fraction(modality, target_range) -> float:
#|     if modality not in MODALITY_YEAR_RANGES: return 1.0
#|     m_lo, m_hi = MODALITY_YEAR_RANGES[modality]; t_lo, t_hi = target_range
#|     ov = max(0, min(m_hi, t_hi) - max(m_lo, t_lo) + 1); span = t_hi - t_lo + 1
#|     return ov / span if span > 0 else 0.0
#|
#| # =============================================================================
#| # 6. PROTECTED ANCHORS  (canonical, well-cited features immune to pruning)
#| # =============================================================================
#| PROTECTED_PATTERNS = [
#|     r"^PRISM_tmean_mean$", r"^PRISM_tmax_mean$", r"^PRISM_tmin_mean$",
#|     r"^PRISM_ppt_sum_mean$", r"^PRISM_summer_tmean_mean$", r"^PRISM_vpdmax_mean$",
#|     r"^PRISM_days_tmax_over_35C_mean$", r"^PRISM_days_tmin_under_0C_mean$",
#|     r"^NDVI_mean$", r"^NDVI_EVI_mean$", r"^Landsat_NDVI_mean$", r"^LS57_NDVI_mean$",
#|     r"^LST_Day_1km_mean$", r"^LST_Night_1km_mean$", r"^DEM_mean$",
#|     r"^SOIL_org_carbon_0cm_mean$", r"^SOIL_ph_0cm_mean$", r"^Soil_mean$",
#|     r"^TERRA_aet_mean$", r"^TERRA_pdsi_worst_month_mean$", r"^TERRA_def_mean$",
#|     r"^AIR_no2_mean_mean$", r"^AIR_aerosol_index_mean$", r"^PM25_mean$",
#|     r"^GHM_mean$", r"^ECON_mean$", r"^NTL_mean$",
#|     r"^META_canopy_height_mean$", r"^S1_VV_mean$", r"^S1_VH_mean$",
#|     r"^ENG_Diurnal_Temp_Range$", r"^ENG_NDVI_LST_Divergence$", r"^ENG_Wet_Bulb_Proxy$",
#|     # a few canonical SDOH anchors so the SDOH-only model keeps its backbone
#|     r"^SDOH_SAIPE_MEDIAN_HH_INCOME$", r"^SDOH_ACS_PCT_LT_HS$",
#|     r"^SDOH_CHR_PCT_SMOKING$", r"^SDOH_CHR_PCT_ADULT_OBESITY$",
#| ]
#| _PROT_RE = [re.compile(p) for p in PROTECTED_PATTERNS]
#| def is_protected(name: str) -> bool:
#|     return any(rx.match(name) for rx in _PROT_RE)
#|
#| # =============================================================================
#| # 7. METRICS
#| # =============================================================================
#| def compute_all_metrics(y_true, y_pred, label="") -> dict:
#|     y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
#|     m = ~(np.isnan(y_true) | np.isnan(y_pred))
#|     y_true, y_pred = y_true[m], y_pred[m]
#|     n = len(y_true)
#|     if n < 3:
#|         return {"label": label, "n": n, "R2": np.nan, "MAE": np.nan,
#|                 "MedAE": np.nan, "RMSE": np.nan, "MSE": np.nan,
#|                 "MAPE_pct": np.nan, "Pearson_r": np.nan}
#|     try:
#|         mse = mean_squared_error(y_true, y_pred)
#|         with np.errstate(divide="ignore", invalid="ignore"):
#|             denom = np.where(y_true == 0, np.nan, y_true)
#|             mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
#|         pr, _ = sp_stats.pearsonr(y_true, y_pred)
#|         return {"label": label, "n": n,
#|                 "R2": round(float(r2_score(y_true, y_pred)), 6),
#|                 "MAE": round(float(mean_absolute_error(y_true, y_pred)), 6),
#|                 "MedAE": round(float(np.median(np.abs(y_true - y_pred))), 6),
#|                 "RMSE": round(float(np.sqrt(mse)), 6),
#|                 "MSE": round(float(mse), 6),
#|                 "MAPE_pct": round(float(mape), 4),
#|                 "Pearson_r": round(float(pr), 6)}
#|     except Exception:
#|         log_exc("metric computation failed")
#|         return {"label": label, "n": n, "R2": np.nan, "MAE": np.nan,
#|                 "MedAE": np.nan, "RMSE": np.nan, "MSE": np.nan,
#|                 "MAPE_pct": np.nan, "Pearson_r": np.nan}
#|
#| def summarise_folds(fold_metrics: list) -> dict:
#|     if not fold_metrics: return {}
#|     keys = [k for k in fold_metrics[0] if k not in ("label", "n")]
#|     out = {"n_folds": len(fold_metrics)}
#|     for k in keys:
#|         vals = [fm[k] for fm in fold_metrics if not pd.isna(fm[k])]
#|         out[f"{k}_mean"] = round(float(np.mean(vals)), 6) if vals else np.nan
#|         out[f"{k}_std"]  = round(float(np.std(vals)), 6) if vals else np.nan
#|     return out
#|
#| # =============================================================================
#| # 8. PREPROCESSING  (fit on train fold only — leakage-free)
#| # =============================================================================
#| def preprocess_fit(X_train: pd.DataFrame):
#|     Xf = X_train.replace([np.inf, -np.inf], np.nan)
#|     return (Xf.median(numeric_only=True),
#|             Xf.quantile(0.001), Xf.quantile(0.999))
#|
#| def preprocess_transform(X, medians, lo, hi):
#|     X = X.replace([np.inf, -np.inf], np.nan).fillna(medians)
#|     X = X.clip(lower=lo, upper=hi, axis=1)
#|     return X.fillna(0.0)
#|
#| # =============================================================================
#| # 9. STATIC AGGREGATION  (kills flatline-imputation anachronism)
#| # =============================================================================
#| def apply_static_aggregation(df, feature_cols, target_range, threshold):
#|     static_cols, actions = [], []
#|     for mod, yrng in MODALITY_YEAR_RANGES.items():
#|         cov = overlap_fraction(mod, target_range)
#|         if cov >= threshold:
#|             actions.append({"modality": mod, "coverage_frac": round(cov, 3),
#|                             "action": "kept_temporal", "n_features": 0}); continue
#|         mod_cols = [c for c in feature_cols if assign_modality(c) == mod]
#|         if not mod_cols: continue
#|         m_lo, m_hi = yrng
#|         mask = (df["year"] >= m_lo) & (df["year"] <= m_hi)
#|         cty = df.loc[mask, ["GEOID"] + mod_cols].groupby("GEOID").mean()
#|         for c in mod_cols:
#|             df[c] = df["GEOID"].map(cty[c])
#|         static_cols += mod_cols
#|         actions.append({"modality": mod, "coverage_frac": round(cov, 3),
#|                         "action": "static_per_county_mean", "n_features": len(mod_cols),
#|                         "observed_years": f"{m_lo}-{m_hi}"})
#|     return df, static_cols, pd.DataFrame(actions)
#|
#| # =============================================================================
#| # 10. COLLINEARITY PRUNER  (numpy-accelerated, anchor-protected)
#| # =============================================================================
#| def collinearity_pruning_report(X, cfg: Config):
#|     audit = []
#|     X_ = X.replace([np.inf, -np.inf], np.nan)
#|     miss = X_.isna().mean()
#|     too_missing = miss[miss > cfg.max_missing_frac].index.tolist()
#|     for c in too_missing:
#|         audit.append({"feature": c, "stage": "missingness",
#|                       "score": float(miss[c]), "kept_partner": ""})
#|     keep = [c for c in X_.columns if c not in too_missing]
#|     Xv = X_[keep].fillna(X_[keep].median(numeric_only=True))
#|     var = Xv.var(numeric_only=True)
#|     nzv = var[var < cfg.nzv_thresh].index.tolist()
#|     for c in nzv:
#|         audit.append({"feature": c, "stage": "near_zero_variance",
#|                       "score": float(var.get(c, 0.0)), "kept_partner": ""})
#|     keep = [c for c in keep if c not in nzv]
#|     if len(keep) > 1:
#|         sub = Xv[keep]
#|         if len(sub) > 20000:
#|             sub = sub.sample(n=20000, random_state=cfg.seed)
#|         with np.errstate(invalid="ignore", divide="ignore"):
#|             cm = np.abs(np.corrcoef(sub.values, rowvar=False))
#|         np.fill_diagonal(cm, 0.0); cm = np.where(np.isnan(cm), 0.0, cm)
#|         cols = sub.columns.tolist(); n = len(cols)
#|         keep_mask = np.ones(n, dtype=bool)
#|         def _score(name):
#|             s = 0.0
#|             if name.endswith("_mean"): s -= 5
#|             if "weighted_mean" in name: s += 3
#|             if "_stdDev" in name or "_std" in name: s += 4
#|             if "growth_rate_" in name: s += 6
#|             for tag in ("_p10","_p20","_p25","_p30","_p40","_p60","_p70","_p75","_p80","_p90","_p99"):
#|                 if tag in name: s += 2; break
#|             if name.endswith("_max") or name.endswith("_min"): s += 1
#|             return s + len(name) * 0.01
#|         order = np.argsort(-cm.mean(axis=0))
#|         for ic in order:
#|             if not keep_mask[ic]: continue
#|             for ip in np.where((cm[ic] > cfg.corr_thresh) & keep_mask)[0]:
#|                 if ip == ic or not keep_mask[ip]: continue
#|                 c, p = cols[ic], cols[ip]
#|                 ca, pa = is_protected(c), is_protected(p)
#|                 if ca and not pa: drop, kept = ip, c
#|                 elif pa and not ca: drop, kept = ic, p
#|                 else:
#|                     drop = ip if _score(c) <= _score(p) else ic
#|                     kept = cols[ic if drop == ip else ip]
#|                 keep_mask[drop] = False
#|                 audit.append({"feature": cols[drop], "stage": f"corr>{cfg.corr_thresh:.2f}",
#|                               "score": float(cm[ic, ip]), "kept_partner": kept})
#|                 if drop == ic: break
#|         keep = [cols[i] for i in range(n) if keep_mask[i]]
#|     return keep, pd.DataFrame(audit)
#|
#| # =============================================================================
#| # 11. MODEL FACTORY + SEARCH SPACES
#| # =============================================================================
#| def make_estimator(name, params=None, threads=N_JOBS, seed=42):
#|     p = dict(params or {})
#|     if name == "RF":
#|         return RandomForestRegressor(random_state=seed, n_jobs=threads, **p)
#|     if name == "XGB" and XGB_OK:
#|         base = dict(verbosity=0, random_state=seed, tree_method="hist",
#|                     device=XGB_DEVICE, n_jobs=threads)
#|         return xgb.XGBRegressor(**{**base, **p})
#|     if name == "LGB" and LGB_OK:
#|         base = dict(verbosity=-1, random_state=seed, n_jobs=threads, force_col_wise=True)
#|         obj = p.pop("objective", None); alpha = p.pop("alpha", None)
#|         if obj: base["objective"] = obj
#|         if alpha is not None: base["alpha"] = alpha
#|         return lgb.LGBMRegressor(**{**base, **p})
#|     if name == "CB" and CB_OK:
#|         base = dict(verbose=False, random_seed=seed, thread_count=threads,
#|                     allow_writing_files=False, task_type=CB_TASK_TYPE)
#|         return cbst.CatBoostRegressor(**{**base, **p})
#|     raise ValueError(f"model {name} unavailable")
#|
#| def search_space(name, test_mode):
#|     if test_mode:
#|         return {"RF": {"n_estimators":[50,100],"max_depth":[8,12],"min_samples_leaf":[1,2],"max_features":["sqrt"]},
#|                 "XGB":{"n_estimators":[80,120],"max_depth":[4,6],"learning_rate":[0.05,0.1],"subsample":[0.8,1.0]},
#|                 "LGB":{"n_estimators":[80,120],"max_depth":[6,10],"learning_rate":[0.05,0.1],"num_leaves":[15,31]},
#|                 "CB": {"iterations":[80,120],"depth":[4,6],"learning_rate":[0.05,0.1],"l2_leaf_reg":[1,3]}}[name]
#|     return {"RF": {"n_estimators":[800,1200,1600,2000],"max_depth":[None,25,35,50],
#|                    "min_samples_split":[2,4,8],"min_samples_leaf":[1,2,4],
#|                    "max_features":["sqrt",0.3,0.5],"max_samples":[0.7,0.85,0.95]},
#|             "XGB":{"n_estimators":[1000,1500,2000,2500],"max_depth":[6,8,10,12],
#|                    "learning_rate":[0.01,0.02,0.05],"subsample":[0.7,0.8,0.9],
#|                    "colsample_bytree":[0.4,0.5,0.6,0.8],"reg_alpha":[0.0,0.1,1.0],
#|                    "reg_lambda":[1.0,5.0,10.0],"min_child_weight":[1,3,5],"gamma":[0.0,0.1,1.0]},
#|             "LGB":{"n_estimators":[1000,1500,2000,2500],"max_depth":[-1,8,12,16],
#|                    "learning_rate":[0.01,0.02,0.05],"num_leaves":[31,63,95,127],
#|                    "subsample":[0.6,0.8,1.0],"colsample_bytree":[0.4,0.6,0.8],
#|                    "reg_alpha":[0.0,0.1,1.0],"reg_lambda":[0.0,1.0,5.0,10.0],
#|                    "min_child_samples":[10,20,40]},
#|             "CB": {"iterations":[800,1200,1600,2000],"depth":[6,8,10],
#|                    "learning_rate":[0.03,0.05,0.1],"l2_leaf_reg":[1,3,5,7],
#|                    "subsample":[0.7,0.85,1.0],"colsample_bylevel":[0.6,0.8,1.0],
#|                    "min_data_in_leaf":[1,5,20]}}[name]
#|
#| DEFAULT_PARAMS_FOR_ABLATION = {
#|     "RF": dict(n_estimators=500, max_depth=30, min_samples_leaf=2, max_features="sqrt"),
#|     "XGB": dict(n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.8,
#|                 colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0),
#|     "LGB": dict(n_estimators=800, max_depth=-1, num_leaves=63, learning_rate=0.05,
#|                 subsample=0.8, colsample_bytree=0.8, min_child_samples=20),
#|     "CB": dict(iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=3),
#| }
#|
#| # =============================================================================
#| # 12. PRECOMPUTED FOLD SPLITS  (saved to disk so folds are auditable/reusable)
#| # =============================================================================
#| def precompute_fold_splits(X, y, groups, cfg: Config, outdir: Path = None,
#|                            tag: str = "main"):
#|     gkf = GroupKFold(n_splits=cfg.n_folds)
#|     splits = []
#|     fold_assign = np.full(len(X), -1, dtype=int)
#|     for k, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
#|         m, lo, hi = preprocess_fit(X.iloc[tr])
#|         splits.append({"fold": k, "tr_idx": tr, "te_idx": te,
#|                        "X_train": preprocess_transform(X.iloc[tr], m, lo, hi),
#|                        "y_train": y.iloc[tr],
#|                        "X_test": preprocess_transform(X.iloc[te], m, lo, hi),
#|                        "y_test": y.iloc[te], "medians": m, "lo": lo, "hi": hi})
#|         fold_assign[te] = k
#|     if outdir is not None:
#|         # persist fold assignment (GEOID→fold) + raw indices so any later
#|         # re-analysis uses IDENTICAL folds without recomputing
#|         pd.DataFrame({"row": np.arange(len(X)), "GEOID": groups.values,
#|                       "fold": fold_assign}).to_csv(
#|                           outdir / f"fold_assignment_{tag}.csv", index=False)
#|         np.savez_compressed(outdir / f"fold_indices_{tag}.npz",
#|                             **{f"train_{s['fold']}": s["tr_idx"] for s in splits},
#|                             **{f"test_{s['fold']}": s["te_idx"] for s in splits})
#|     return splits
#|
#| # =============================================================================
#| # 13. CV RUNNER  (saves residuals + train metrics + timing for every fold)
#| # =============================================================================
#| def run_cv_on_splits(splits, name, params, cfg, label="", years=None,
#|                      groups=None, return_models=False):
#|     fold_metrics, all_preds, fold_models, fold_rows = [], [], [], []
#|     for spl in splits:
#|         t0 = time.time()
#|         mdl = make_estimator(name, params, seed=cfg.seed)
#|         mdl.fit(spl["X_train"], spl["y_train"])
#|         pred_te = mdl.predict(spl["X_test"])
#|         pred_tr = mdl.predict(spl["X_train"])
#|         dt = time.time() - t0
#|         fm = compute_all_metrics(spl["y_test"], pred_te, f"{label}_fold{spl['fold']}")
#|         tr_r2 = r2_score(spl["y_train"], pred_tr)
#|         fold_metrics.append(fm)
#|         fold_rows.append({"fold": spl["fold"], "label": label,
#|                           "test_R2": fm["R2"], "test_RMSE": fm["RMSE"],
#|                           "test_MAE": fm["MAE"], "train_R2": round(float(tr_r2), 6),
#|                           "overfit_gap": round(float(tr_r2 - (fm["R2"] or 0)), 6),
#|                           "n_test": fm["n"], "seconds": round(dt, 1)})
#|         log(f"  [{label}] fold {spl['fold']}/{len(splits)}: "
#|             f"test R²={fm['R2']:.4f} RMSE={fm['RMSE']:.4f} "
#|             f"(train R²={tr_r2:.4f}, gap={tr_r2-(fm['R2'] or 0):.3f}, {dt:.1f}s)")
#|         te = spl["te_idx"]
#|         all_preds.append(pd.DataFrame({
#|             "GEOID": groups.iloc[te].values if groups is not None else te,
#|             "year": years.iloc[te].values if years is not None else np.nan,
#|             "actual": spl["y_test"].values, "predicted": pred_te,
#|             "residual": spl["y_test"].values - pred_te,
#|             "abs_error": np.abs(spl["y_test"].values - pred_te), "fold": spl["fold"]}))
#|         if return_models: fold_models.append(mdl)
#|         else: del mdl; gc.collect()
#|     return (fold_metrics, summarise_folds(fold_metrics),
#|             pd.concat(all_preds, ignore_index=True),
#|             pd.DataFrame(fold_rows), fold_models)
#|
#| # =============================================================================
#| # 14. TUNING  (reuses precomputed splits; falls back to gold params in FAST)
#| # =============================================================================
#| def tune_model_on_splits(splits, name, cfg, threads=N_JOBS):
#|     space = search_space(name, cfg.test_mode)
#|     keys, values = list(space.keys()), list(space.values())
#|     space_size = int(np.prod([len(v) for v in values]))
#|     n_eff = min(cfg.n_iter_search, max(1, space_size))
#|     rng = random.Random(cfg.seed)
#|     seen, trials, attempts = set(), [], 0
#|     while len(trials) < n_eff and attempts < max(50, n_eff * 20):
#|         attempts += 1
#|         combo = tuple(rng.choice(v) for v in values)
#|         if combo in seen: continue
#|         seen.add(combo); trials.append(dict(zip(keys, combo)))
#|     best_rmse, best_params, trial_log = np.inf, None, []
#|     for i, params in enumerate(trials, 1):
#|         try:
#|             rmses = []
#|             for spl in splits:
#|                 mdl = make_estimator(name, params, threads=threads, seed=cfg.seed)
#|                 mdl.fit(spl["X_train"], spl["y_train"])
#|                 rmses.append(np.sqrt(mean_squared_error(
#|                     spl["y_test"], mdl.predict(spl["X_test"]))))
#|                 del mdl
#|             mean_rmse = float(np.mean(rmses))
#|         except Exception:
#|             log_exc(f"trial {i} failed for {name}"); continue
#|         trial_log.append({"trial": i, "mean_rmse": mean_rmse, **params})
#|         if mean_rmse < best_rmse:
#|             best_rmse, best_params = mean_rmse, params
#|             log(f"    [{name}] trial {i}: RMSE={mean_rmse:.4f} ★")
#|     return best_params, best_rmse, pd.DataFrame(trial_log)
#|
#| # =============================================================================
#| # 15. FEATURE IMPORTANCE
#| # =============================================================================
#| def get_feature_importances(model, names):
#|     if hasattr(model, "feature_importances_"):
#|         imp = np.asarray(model.feature_importances_, float)
#|         if imp.sum() > 0: imp = imp / imp.sum()
#|     else:
#|         imp = np.zeros(len(names))
#|     df = pd.DataFrame({"Feature": names, "Importance": imp}) \
#|            .sort_values("Importance", ascending=False).reset_index(drop=True)
#|     df["Rank"] = df.index + 1
#|     return df
#|
#| # =============================================================================
#| # 16. SHAP + cross-fold stability
#| # =============================================================================
#| def compute_shap_for_model(model, X_full, cfg):
#|     if not SHAP_OK: return None, None, None
#|     Xs = X_full.sample(n=min(cfg.shap_sample, len(X_full)), random_state=cfg.seed)
#|     Xbg = X_full.sample(n=min(cfg.shap_bg, len(X_full)), random_state=cfg.seed + 1)
#|     try:
#|         ex = shap.TreeExplainer(model, data=Xbg, feature_perturbation="interventional")
#|     except Exception:
#|         ex = shap.TreeExplainer(model)
#|     try:
#|         sv = ex.shap_values(Xs, check_additivity=False)
#|     except TypeError:
#|         sv = ex.shap_values(Xs)
#|     except Exception:
#|         log_exc("SHAP failed"); return None, None, None
#|     imp = pd.DataFrame({"Feature": Xs.columns.tolist(),
#|                         "Mean_AbsSHAP": np.abs(sv).mean(0)}) \
#|             .sort_values("Mean_AbsSHAP", ascending=False).reset_index(drop=True)
#|     imp["SHAP_Rank"] = imp.index + 1
#|     return sv, Xs, imp
#|
#| def compute_shap_stability(fold_models, fold_X_test, cfg):
#|     if not SHAP_OK or not fold_models: return pd.DataFrame(), pd.DataFrame()
#|     topk = cfg.shap_topk; dfs = []
#|     for fi, (mdl, Xte) in enumerate(zip(fold_models, fold_X_test), 1):
#|         Xs = Xte.sample(n=min(2000 if not cfg.test_mode else 100, len(Xte)),
#|                         random_state=cfg.seed + fi)
#|         Xbg = Xte.sample(n=min(200 if not cfg.test_mode else 50, len(Xte)),
#|                          random_state=cfg.seed + 100 + fi)
#|         try: ex = shap.TreeExplainer(mdl, data=Xbg, feature_perturbation="interventional")
#|         except Exception: ex = shap.TreeExplainer(mdl)
#|         try: sv = ex.shap_values(Xs, check_additivity=False)
#|         except TypeError: sv = ex.shap_values(Xs)
#|         except Exception: log_exc(f"SHAP fold {fi} failed"); continue
#|         d = pd.DataFrame({"Feature": Xs.columns.tolist(),
#|                           "Mean_AbsSHAP": np.abs(sv).mean(0)})
#|         d["fold"] = fi
#|         d = d.sort_values("Mean_AbsSHAP", ascending=False).reset_index(drop=True)
#|         d["Rank"] = d.index + 1; dfs.append(d)
#|     if not dfs: return pd.DataFrame(), pd.DataFrame()
#|     long = pd.concat(dfs, ignore_index=True)
#|     rw = long.pivot_table(index="Feature", columns="fold", values="Rank", aggfunc="first")
#|     vw = long.pivot_table(index="Feature", columns="fold", values="Mean_AbsSHAP", aggfunc="first")
#|     stab = pd.DataFrame({"Feature": rw.index, "rank_mean": rw.mean(1).values,
#|                          "rank_std": rw.std(1).values, "AbsSHAP_mean": vw.mean(1).values,
#|                          "AbsSHAP_CV": (vw.std(1) / vw.mean(1).replace(0, np.nan)).values,
#|                          "n_folds_topK": (rw <= topk).sum(1).values,
#|                          "n_folds_present": rw.notna().sum(1).values}) \
#|              .sort_values("rank_mean").reset_index(drop=True)
#|     rows = []; fp = sorted(rw.columns.tolist())
#|     for i in range(len(fp)):
#|         for j in range(i + 1, len(fp)):
#|             fi, fj = fp[i], fp[j]
#|             ti = set(rw[rw[fi] <= topk].index); tj = set(rw[rw[fj] <= topk].index)
#|             jac = len(ti & tj) / len(ti | tj) if (ti | tj) else np.nan
#|             common = rw.dropna(subset=[fi, fj])
#|             rho = sp_stats.spearmanr(common[fi], common[fj])[0] if len(common) > 5 else np.nan
#|             rows.append({"fold_i": fi, "fold_j": fj,
#|                          f"Jaccard_top{topk}": round(float(jac), 4),
#|                          "Spearman_rho": round(float(rho), 4)})
#|     return stab, pd.DataFrame(rows)
#|
#| # =============================================================================
#| # 17. QUANTILE INTERVALS
#| # =============================================================================
#| def fit_quantile_models(Xtr, ytr, params, cfg, alphas=(0.1, 0.5, 0.9)):
#|     if not LGB_OK: return {}
#|     out = {}
#|     for a in alphas:
#|         m = make_estimator("LGB", {**params, "objective": "quantile", "alpha": a}, seed=cfg.seed)
#|         m.fit(Xtr, ytr); out[a] = m
#|     return out
#|
#| def predict_with_intervals(qmodels, X):
#|     if not qmodels: return pd.DataFrame(index=X.index)
#|     out = {f"q{int(a*100):02d}": m.predict(X) for a, m in qmodels.items()}
#|     df = pd.DataFrame(out, index=X.index)
#|     if "q10" in df and "q90" in df: df["interval_width_80"] = df["q90"] - df["q10"]
#|     return df
#|
#| # =============================================================================
#| # 18. OOD DETECTION
#| # =============================================================================
#| def fit_ood_detector(Xtr, cfg):
#|     nc = min(cfg.ood_pca, Xtr.shape[1], Xtr.shape[0] - 1)
#|     if nc < 2: return {"available": False}
#|     pca = PCA(n_components=nc, random_state=cfg.seed); Z = pca.fit_transform(Xtr.values)
#|     try:
#|         cov = MinCovDet(random_state=cfg.seed).fit(Z) if len(Z) <= 30000 else EmpiricalCovariance().fit(Z)
#|     except Exception:
#|         cov = EmpiricalCovariance().fit(Z)
#|     d = cov.mahalanobis(Z)
#|     return {"available": True, "pca": pca, "cov": cov,
#|             "threshold": float(np.quantile(d, cfg.ood_quantile)), "n_components": nc}
#|
#| def score_ood(det, X):
#|     if not det.get("available"): return np.full(len(X), np.nan), np.zeros(len(X), bool)
#|     Z = det["pca"].transform(X.values); d = det["cov"].mahalanobis(Z)
#|     return d, d > det["threshold"]
#|
#| # =============================================================================
#| # 19. ENSEMBLE (top-3 inverse-RMSE)
#| # =============================================================================
#| def fit_top_k_ensemble(splits, top_models, cfg, years, groups):
#|     fold_preds, weights = [], {}
#|     for name, params in top_models:
#|         _, summ, preds, _, _ = run_cv_on_splits(splits, name, params, cfg,
#|                                                  label=f"ENS_{name}", years=years, groups=groups)
#|         fold_preds.append((name, preds))
#|         weights[name] = 1.0 / max(summ.get("RMSE_mean", np.inf), 1e-6)
#|     z = sum(weights.values()); weights = {k: v / z for k, v in weights.items()}
#|     base = fold_preds[0][1].rename(columns={"predicted": f"pred_{fold_preds[0][0]}"}) \
#|                            .drop(columns=["residual", "abs_error"])
#|     for name, preds in fold_preds[1:]:
#|         base = base.merge(preds[["GEOID", "year", "fold", "predicted"]]
#|                           .rename(columns={"predicted": f"pred_{name}"}),
#|                           on=["GEOID", "year", "fold"], how="inner")
#|     base["predicted"] = sum(weights[n] * base[f"pred_{n}"] for n, _ in top_models)
#|     base["residual"] = base["actual"] - base["predicted"]
#|     base["abs_error"] = base["residual"].abs()
#|     fm = [compute_all_metrics(base[base.fold == f]["actual"], base[base.fold == f]["predicted"],
#|                               f"ens_fold{f}") for f in sorted(base["fold"].unique())]
#|     return base, summarise_folds(fm), weights
#|
#| # =============================================================================
#| # 20. FORWARD-CHAINING TEMPORAL CV
#| # =============================================================================
#| def forward_chaining_temporal_cv(df, feats, name, params, cfg):
#|     lo, hi = cfg.year_start, cfg.year_end; rows = []; block = 3 if (hi - lo) < 12 else 4
#|     train_end = lo + max(5, (hi - lo) // 2)
#|     while train_end + 1 <= hi:
#|         test_end = min(train_end + block, hi)
#|         tr = df[(df.year >= lo) & (df.year <= train_end)]
#|         te = df[(df.year > train_end) & (df.year <= test_end)]
#|         if len(tr) < 500 or len(te) < 100: break
#|         m, l, h = preprocess_fit(tr[feats])
#|         mdl = make_estimator(name, params, seed=cfg.seed)
#|         mdl.fit(preprocess_transform(tr[feats], m, l, h), tr[TARGET_COL].astype(float))
#|         pred = mdl.predict(preprocess_transform(te[feats], m, l, h))
#|         fm = compute_all_metrics(te[TARGET_COL].astype(float), pred,
#|                                  f"fwd_{lo}-{train_end}_test_{train_end+1}-{test_end}")
#|         fm.update({"train_years": f"{lo}-{train_end}", "test_years": f"{train_end+1}-{test_end}",
#|                    "n_train": len(tr), "n_test": len(te)})
#|         rows.append(fm)
#|         log(f"  fwd-chain {fm['label']}: R²={fm['R2']:.4f} RMSE={fm['RMSE']:.4f}")
#|         train_end += block
#|     return pd.DataFrame(rows)
#|
#| # =============================================================================
#| # 21. STATISTICAL TESTS  (bootstrap CI, permutation, paired Wilcoxon)
#| # =============================================================================
#| def bootstrap_r2_ci(y_true, y_pred, cfg):
#|     rng = np.random.default_rng(cfg.seed)
#|     yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
#|     m = ~(np.isnan(yt) | np.isnan(yp)); yt, yp = yt[m], yp[m]
#|     idx = np.arange(len(yt)); r2s = []
#|     for _ in range(cfg.bootstrap_n):
#|         b = rng.choice(idx, len(idx), replace=True)
#|         r2s.append(r2_score(yt[b], yp[b]))
#|     lo, hi = np.percentile(r2s, [2.5, 97.5])
#|     return {"r2": float(r2_score(yt, yp)), "ci_low": float(lo), "ci_high": float(hi),
#|             "n_boot": cfg.bootstrap_n}
#|
#| def permutation_test(splits, name, params, cfg, real_oof_r2):
#|     """Shuffle y, re-run CV on the SAME preprocessed fold matrices (valid
#|     because preprocessing never used y), collect null R² distribution."""
#|     rng = np.random.default_rng(cfg.seed); null = []
#|     for _ in range(cfg.perm_n):
#|         preds_all, ytrue_all = [], []
#|         for spl in splits:
#|             yptr = pd.Series(rng.permutation(spl["y_train"].values), index=spl["y_train"].index)
#|             mdl = make_estimator(name, params, seed=cfg.seed)
#|             mdl.fit(spl["X_train"], yptr)
#|             preds_all.append(mdl.predict(spl["X_test"])); ytrue_all.append(spl["y_test"].values)
#|             del mdl
#|         null.append(r2_score(np.concatenate(ytrue_all), np.concatenate(preds_all)))
#|     null = np.array(null)
#|     p = (np.sum(null >= real_oof_r2) + 1) / (cfg.perm_n + 1)
#|     return {"real_r2": float(real_oof_r2), "null_mean": float(null.mean()),
#|             "null_std": float(null.std()), "p_value": float(p), "n_perm": cfg.perm_n}, null
#|
#| def paired_wilcoxon(preds_a, preds_b, label_a, label_b):
#|     """Paired test on per-row absolute error between two models (aligned by
#|     GEOID+year+fold)."""
#|     key = ["GEOID", "year", "fold"]
#|     m = preds_a[key + ["abs_error"]].merge(
#|         preds_b[key + ["abs_error"]], on=key, suffixes=("_a", "_b"))
#|     if len(m) < 10:
#|         return {"model_a": label_a, "model_b": label_b, "n": len(m),
#|                 "wilcoxon_stat": np.nan, "p_value": np.nan, "median_ae_a": np.nan,
#|                 "median_ae_b": np.nan}
#|     try:
#|         stat, p = sp_stats.wilcoxon(m["abs_error_a"], m["abs_error_b"])
#|     except Exception:
#|         stat, p = np.nan, np.nan
#|     return {"model_a": label_a, "model_b": label_b, "n": len(m),
#|             "wilcoxon_stat": float(stat), "p_value": float(p),
#|             "median_ae_a": float(m["abs_error_a"].median()),
#|             "median_ae_b": float(m["abs_error_b"].median())}
#|
#| # =============================================================================
#| # 22. DIAGNOSTIC TABLES
#| # =============================================================================
#| def make_bracket_table(preds):
#|     bins = [60, 70, 74, 76.5, 79, 95]; labels = ["60-70","70-74","74-76.5","76.5-79","79-95"]
#|     p = preds.copy(); p["LE_Bracket"] = pd.cut(p["actual"], bins=bins, labels=labels)
#|     def _rmse(x): return float(np.sqrt((np.asarray(x, float) ** 2).mean()))
#|     return (p.groupby("LE_Bracket", observed=True)
#|               .agg(Unique_Counties=("GEOID","nunique"), Count=("actual","count"),
#|                    Mean_LE=("actual","mean"), MAE=("abs_error","mean"),
#|                    RMSE=("residual", _rmse), Mean_Residual=("residual","mean")).reset_index())
#|
#| def make_temporal_table(preds):
#|     if "year" not in preds.columns: return pd.DataFrame()
#|     rows = []
#|     for yr, g in preds.groupby("year", observed=True):
#|         if pd.isna(yr): continue
#|         rows.append({"year": yr, **compute_all_metrics(g["actual"], g["predicted"], str(yr)),
#|                      "n_counties": g["GEOID"].nunique()})
#|     return pd.DataFrame(rows).sort_values("year") if rows else pd.DataFrame()
#|
#| def make_county_error_table(preds):
#|     t = (preds.groupby("GEOID").agg(Mean_LE=("actual","mean"), MAE=("abs_error","mean"),
#|          Mean_Resid=("residual","mean"), N_Years=("actual","count")).reset_index().sort_values("MAE"))
#|     return t.head(20), t.tail(20).sort_values("MAE", ascending=False)
#|
#| def make_residual_stats(preds):
#|     r = preds["residual"].dropna()
#|     if len(r) == 0: return pd.DataFrame(columns=["Statistic", "Value"])
#|     sw = sp_stats.shapiro(r.sample(min(5000, len(r)), random_state=42))
#|     return pd.DataFrame([("Mean_Residual", float(r.mean())), ("Std_Residual", float(r.std())),
#|         ("Skewness", float(sp_stats.skew(r))), ("Kurtosis", float(sp_stats.kurtosis(r))),
#|         ("Shapiro_W", float(sw[0])), ("Shapiro_p", float(sw[1])),
#|         ("Min", float(r.min())), ("Max", float(r.max())),
#|         ("CI95_lower", float(np.percentile(r, 2.5))),
#|         ("CI95_upper", float(np.percentile(r, 97.5)))], columns=["Statistic", "Value"])
#|
#| # =============================================================================
#| # 23. HEADLINE-SPECIFIC BLOCKS (SDOH-vs-Env, smoothed-SDOH baseline, ablation)
#| # =============================================================================
#| def sdoh_env_split(feats):
#|     sdoh = [c for c in feats if assign_modality(c) == "SDOH"]
#|     env  = [c for c in feats if assign_modality(c) in ENV_MODALITIES]
#|     return sdoh, env
#|
#| def baseline_smoothed_sdoh(df, sdoh_cols, name, params, cfg, groups, years):
#|     """IHME apples-to-apples: SDOH + fold-safe state-mean-LE spatial feature."""
#|     y = df[TARGET_COL].astype(float).values
#|     state = (pd.to_numeric(df["GEOID"]) // 1000).values
#|     oof = np.full(len(df), np.nan)
#|     for tr, te in GroupKFold(cfg.n_folds).split(df, y, groups):
#|         sm = pd.Series(y[tr]).groupby(state[tr]).mean(); gm = y[tr].mean()
#|         smt = np.array([sm.get(s, gm) for s in state[tr]])
#|         sme = np.array([sm.get(s, gm) for s in state[te]])
#|         Xtr = np.column_stack([df.iloc[tr][sdoh_cols].values, smt])
#|         Xte = np.column_stack([df.iloc[te][sdoh_cols].values, sme])
#|         m, lo, hi = preprocess_fit(pd.DataFrame(Xtr))
#|         mdl = make_estimator(name, params, seed=cfg.seed)
#|         mdl.fit(preprocess_transform(pd.DataFrame(Xtr), m, lo, hi), y[tr])
#|         oof[te] = mdl.predict(preprocess_transform(pd.DataFrame(Xte), m, lo, hi))
#|     return compute_all_metrics(y, oof, "smoothed_sdoh_IHME_comparable"), oof
#|
#| # =============================================================================
#| # 24. MAIN — fully checkpointed stage graph
#| # =============================================================================
#| STAGE_LIST = ["load", "features", "holdout", "prune", "splits", "tournament",
#|               "cv_all", "final_fit", "importance", "shap", "quantile",
#|               "ensemble", "ablation", "headline", "temporal", "forecast",
#|               "stats", "tables", "export"]
#|
#| def main():
#|     global XGB_DEVICE, CB_TASK_TYPE
#|     cfg = parse_args()
#|     if getattr(cfg, "_list_stages", False):
#|         print("Stages:", ", ".join(STAGE_LIST)); return
#|     outdir = Path(cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)
#|     setup_logging(outdir)
#|     XGB_DEVICE, CB_TASK_TYPE = detect_devices(cfg.seed)
#|     mgr = CheckpointManager(outdir, cfg)
#|     t_start = time.time()
#|
#|     log_sep("BIGBANG v3 — CHECKPOINTED LEAKAGE-FREE PIPELINE")
#|     log(f"OUTPUT      : {outdir}")
#|     log(f"DATA        : {cfg.data_path}")
#|     log(f"MODE        : {cfg.mode}  | TEST_MODE={cfg.test_mode}")
#|     log(f"WINDOW      : {cfg.year_start}-{cfg.year_end}  forecast {cfg.forecast_start}-{cfg.forecast_end}")
#|     log(f"SCIENCE HASH: {mgr.hash}  (checkpoints validated against this)")
#|     log(f"N_JOBS={N_JOBS} BLAS={_BLAS_THREADS} | XGB={XGB_DEVICE} CB={CB_TASK_TYPE}")
#|     log(f"Models: {MODEL_AVAILABILITY} | SHAP={SHAP_OK}")
#|
#|     # ---- STAGE load ----
#|     def _load():
#|         df = pd.read_csv(cfg.data_path, low_memory=False)
#|         if "GEOID" not in df.columns and "fips" in df.columns:
#|             df["GEOID"] = df["fips"]
#|         df["GEOID"] = df["GEOID"].astype(str).str.replace(r"\.0+$", "", regex=True).str.zfill(5)
#|         if "STATEFP" not in df.columns:
#|             df["STATEFP"] = df["GEOID"].str[:2]
#|         df["STATEFP"] = df["STATEFP"].astype(str).str.replace(r"\.0+$", "", regex=True).str.zfill(2)
#|         n0 = len(df)
#|         df = df[~df["STATEFP"].isin(EXCLUDE_STATEFP)].reset_index(drop=True)
#|         log(f"CONUS filter: {n0-len(df):,} dropped -> {len(df):,} rows")
#|         if TARGET_COL not in df.columns:
#|             log(f"FATAL target missing", "ERROR"); sys.exit(1)
#|         sha = hashlib.sha256(open(cfg.data_path, "rb").read()).hexdigest()
#|         # leakage audit
#|         leak = sorted([c for c in df.columns if is_leakage(c)])
#|         log(f"⚠ leakage cols dropped ({len(leak)}): {leak}")
#|         (outdir / "leakage_dropped.json").write_text(json.dumps(leak, indent=2))
#|         # missingness
#|         miss = df.isnull().sum(); miss = miss[miss > 0].sort_values(ascending=False)
#|         md = miss.reset_index(); md.columns = ["col", "n_missing"]
#|         md["pct"] = md["n_missing"] / len(df) * 100
#|         md.to_csv(outdir / "data_quality_missing_report.csv", index=False)
#|         return {"df": df, "sha256": sha, "leakage": leak}
#|     S = mgr.run("load", _load)
#|     df_raw, sha256, leakage_cols = S["df"], S["sha256"], S["leakage"]
#|
#|     # ---- STAGE features ----
#|     def _features():
#|         yr = pd.to_numeric(df_raw["year"], errors="coerce")
#|         dft = df_raw[(yr >= cfg.year_start) & (yr <= cfg.year_end)].dropna(
#|             subset=[TARGET_COL]).reset_index(drop=True)
#|         if cfg.test_mode:
#|             keep_g = dft["GEOID"].unique()[:50]
#|             dft = dft[dft["GEOID"].isin(keep_g)].reset_index(drop=True)
#|         forbidden = set(ID_COLS) | {TARGET_COL} | set(leakage_cols) | EXPLICIT_DROP_COLS
#|         cand = [c for c in dft.columns if c not in forbidden
#|                 and not any(c.startswith(p) for p in cfg.exclude_prefixes)]
#|         num = [c for c in cand if pd.api.types.is_numeric_dtype(dft[c])]
#|         # static aggregation
#|         dft, static_cols, agg = apply_static_aggregation(
#|             dft, num, (cfg.year_start, cfg.year_end), cfg.static_modality_thresh)
#|         agg.to_csv(outdir / "static_aggregation_actions.csv", index=False)
#|         # modality coverage report
#|         cov_rows = []
#|         for mod, cnt in Counter(assign_modality(c) for c in num).items():
#|             yl, yh = MODALITY_YEAR_RANGES.get(mod, ("?", "?"))
#|             cov_rows.append({"modality": mod, "n_features": cnt, "year_lo": yl,
#|                              "year_hi": yh, "coverage": round(overlap_fraction(mod, (cfg.year_start, cfg.year_end)), 3)})
#|         pd.DataFrame(cov_rows).to_csv(outdir / "modality_coverage_report.csv", index=False)
#|         pd.DataFrame({"feature": num}).to_csv(outdir / "feature_list_initial.csv", index=False)
#|         log(f"feature matrix: {len(dft):,} rows × {len(num):,} feats | static-agg {len(static_cols)}")
#|         return {"df": dft, "feats": num, "static_cols": static_cols}
#|     S = mgr.run("features", _features)
#|     df, feats_initial, static_cols = S["df"], S["feats"], S["static_cols"]
#|     y = df[TARGET_COL].astype(float); groups = df["GEOID"]; years = df["year"]
#|
#|     # ---- STAGE holdout ----
#|     def _holdout():
#|         rng = np.random.default_rng(cfg.seed)
#|         ug = np.array(sorted(df["GEOID"].unique())); rng.shuffle(ug)
#|         n_hold = max(5 if cfg.test_mode else 50, int(len(ug) * cfg.holdout_frac))
#|         hold = set(ug[:n_hold])
#|         is_h = df["GEOID"].isin(hold).values
#|         pd.DataFrame({"GEOID": sorted(hold)}).to_csv(outdir / "holdout_geoids.csv", index=False)
#|         log(f"holdout {len(hold)} counties / {is_h.sum():,} rows; dev {(~is_h).sum():,} rows")
#|         return {"is_holdout": is_h, "holdout_geoids": sorted(hold)}
#|     S = mgr.run("holdout", _holdout)
#|     is_h = np.asarray(S["is_holdout"]); holdout_geoids = S["holdout_geoids"]
#|     X_full = df[feats_initial]
#|     X_dev, y_dev = X_full.loc[~is_h].reset_index(drop=True), y.loc[~is_h].reset_index(drop=True)
#|     g_dev, yr_dev = groups.loc[~is_h].reset_index(drop=True), years.loc[~is_h].reset_index(drop=True)
#|     X_hold, y_hold = X_full.loc[is_h].reset_index(drop=True), y.loc[is_h].reset_index(drop=True)
#|     g_hold, yr_hold = groups.loc[is_h].reset_index(drop=True), years.loc[is_h].reset_index(drop=True)
#|
#|     # ---- STAGE prune ----
#|     def _prune():
#|         kept, audit = collinearity_pruning_report(X_dev, cfg)
#|         audit.to_csv(outdir / "feature_pruning_audit.csv", index=False)
#|         pd.DataFrame({"feature": kept}).to_csv(outdir / "feature_list_kept.csv", index=False)
#|         log(f"pruned to {len(kept)} feats (anchors kept: {sum(is_protected(c) for c in kept)})")
#|         return {"kept": kept}
#|     kept_cols = mgr.run("prune", _prune)["kept"]
#|     X_dev, X_hold = X_dev[kept_cols], X_hold[kept_cols]
#|
#|     # ---- STAGE splits ----
#|     def _splits():
#|         sp = precompute_fold_splits(X_dev, y_dev, g_dev, cfg, outdir=outdir, tag="main")
#|         return {"splits": sp}
#|     splits = mgr.run("splits", _splits)["splits"]
#|
#|     # ---- STAGE tournament ----
#|     def _tournament():
#|         candidates = [m for m, ok in MODEL_AVAILABILITY.items() if ok]
#|         if cfg.mode == "fast":
#|             log("FAST mode: using gold-mine params (no search)")
#|             rows, params_all, trials = [], {}, {}
#|             for m in candidates:
#|                 p = BEST_PARAMS_GOLD.get(m, DEFAULT_PARAMS_FOR_ABLATION[m])
#|                 _, summ, _, _, _ = run_cv_on_splits(splits, m, p, cfg, label=f"gold_{m}",
#|                                                     years=yr_dev, groups=g_dev)
#|                 rows.append({"model": m, "model_name": MODEL_LABELS[m],
#|                              "CV_RMSE": summ.get("RMSE_mean", np.nan),
#|                              "CV_R2": summ.get("R2_mean", np.nan), "best_params": p})
#|                 params_all[m] = p
#|             tour = pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)
#|         else:
#|             log("FULL mode: concurrent tuning tournament")
#|             cpu_models = [m for m in candidates if m in ("RF", "LGB") or (m == "CB" and CB_TASK_TYPE == "CPU")]
#|             tpm = max(1, N_JOBS // max(1, len(cpu_models)))
#|             def _tune(m):
#|                 thr = N_JOBS if (m == "XGB" and XGB_DEVICE == "cuda") else tpm
#|                 try:
#|                     p, rmse, tl = tune_model_on_splits(splits, m, cfg, threads=thr)
#|                     tl.to_csv(outdir / f"tuning_trials_{m}.csv", index=False)
#|                     return {"model": m, "model_name": MODEL_LABELS[m], "CV_RMSE": rmse,
#|                             "best_params": p} if p else None
#|                 except Exception:
#|                     log_exc(f"tuning failed {m}"); return None
#|             res = Parallel(n_jobs=len(candidates), backend="threading")(
#|                 delayed(_tune)(m) for m in candidates)
#|             rows = [r for r in res if r]
#|             params_all = {r["model"]: r["best_params"] for r in rows}
#|             tour = pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)
#|         tour.drop(columns=["best_params"]).to_csv(outdir / "model_tournament.csv", index=False)
#|         (outdir / "best_params_all.json").write_text(json.dumps(params_all, indent=2, default=str))
#|         log(f"🏆 winner: {tour.iloc[0]['model']}  CV-RMSE={tour.iloc[0]['CV_RMSE']:.4f}")
#|         return {"tournament": tour, "params_all": params_all,
#|                 "winner": tour.iloc[0]["model"]}
#|     S = mgr.run("tournament", _tournament)
#|     tournament_df, best_params_all, winner = S["tournament"], S["params_all"], S["winner"]
#|     winner_params = best_params_all[winner]
#|
#|     # ---- STAGE cv_all ----
#|     def _cv_all():
#|         summaries, preds_by, foldrows_by = [], {}, []
#|         fmodels_winner, fXtest_winner = [], []
#|         for m, p in best_params_all.items():
#|             ret = (m == winner) and SHAP_OK
#|             _, summ, preds, foldrows, fmods = run_cv_on_splits(
#|                 splits, m, p, cfg, label=MODEL_LABELS[m], years=yr_dev,
#|                 groups=g_dev, return_models=ret)
#|             summ["model"] = m; summaries.append(summ)
#|             preds_by[m] = preds
#|             preds.to_csv(outdir / f"cv_predictions_{m}.csv", index=False)
#|             foldrows.to_csv(outdir / f"cv_fold_details_{m}.csv", index=False)
#|             foldrows_by.append(foldrows)
#|             if ret:
#|                 fmodels_winner = fmods
#|                 fXtest_winner = [s["X_test"] for s in splits]
#|         pd.DataFrame(summaries).to_csv(outdir / "cv_summary_all_models.csv", index=False)
#|         pd.concat(foldrows_by, ignore_index=True).to_csv(
#|             outdir / "cv_fold_details_ALL.csv", index=False)
#|         return {"summaries": summaries, "preds_by": preds_by,
#|                 "fmodels_winner": fmodels_winner, "fXtest_winner": fXtest_winner}
#|     S = mgr.run("cv_all", _cv_all)
#|     all_summaries, preds_by_model = S["summaries"], S["preds_by"]
#|     fold_models_winner, fold_X_test_winner = S["fmodels_winner"], S["fXtest_winner"]
#|
#|     # ---- STAGE final_fit ----
#|     def _final_fit():
#|         m, lo, hi = preprocess_fit(X_dev)
#|         Xd, Xh = preprocess_transform(X_dev, m, lo, hi), preprocess_transform(X_hold, m, lo, hi)
#|         mdl = make_estimator(winner, winner_params, seed=cfg.seed)
#|         mdl.fit(Xd, y_dev)
#|         hp = mdl.predict(Xh)
#|         hm = compute_all_metrics(y_hold, hp, "holdout")
#|         log(f"HOLDOUT: R²={hm['R2']:.4f} RMSE={hm['RMSE']:.4f} MAE={hm['MAE']:.4f}")
#|         hdf = pd.DataFrame({"GEOID": g_hold.values, "year": yr_hold.values,
#|                             "actual": y_hold.values, "predicted": hp,
#|                             "residual": y_hold.values - hp, "abs_error": np.abs(y_hold.values - hp)})
#|         hdf.to_csv(outdir / "holdout_predictions.csv", index=False)
#|         with open(outdir / "final_model.pkl", "wb") as f:
#|             pickle.dump({"model": mdl, "medians": m, "lo": lo, "hi": hi,
#|                          "features": kept_cols}, f)
#|         return {"model": mdl, "medians": m, "lo": lo, "hi": hi,
#|                 "holdout_metrics": hm, "holdout_preds": hdf, "Xd": Xd, "Xh": Xh}
#|     S = mgr.run("final_fit", _final_fit)
#|     final_model = S["model"]; medians_f, lo_f, hi_f = S["medians"], S["lo"], S["hi"]
#|     holdout_metrics, holdout_preds_df = S["holdout_metrics"], S["holdout_preds"]
#|     X_dev_imp, X_hold_imp = S["Xd"], S["Xh"]
#|
#|     # ---- STAGE importance (+ post-fit pruned model) ----
#|     def _importance():
#|         imp = get_feature_importances(final_model, kept_cols)
#|         imp.to_csv(outdir / "feature_importance_full.csv", index=False)
#|         keep2 = imp.loc[(imp.Importance >= cfg.importance_prune_thr)
#|                         | imp.Feature.apply(is_protected), "Feature"].tolist()
#|         pd.DataFrame({"feature": keep2}).to_csv(outdir / "feature_list_pruned.csv", index=False)
#|         mp = make_estimator(winner, winner_params, seed=cfg.seed)
#|         mp.fit(X_dev_imp[keep2], y_dev)
#|         phm = compute_all_metrics(y_hold, mp.predict(X_hold_imp[keep2]), "holdout_pruned")
#|         ps = precompute_fold_splits(X_dev[keep2], y_dev, g_dev, cfg, outdir=outdir, tag="pruned")
#|         _, sp, pp, fr, _ = run_cv_on_splits(ps, winner, winner_params, cfg,
#|                                             label=f"{winner}_pruned", years=yr_dev, groups=g_dev)
#|         pp.to_csv(outdir / "cv_predictions_winner_pruned.csv", index=False)
#|         log(f"importance-pruned to {len(keep2)} | pruned holdout R²={phm['R2']:.4f}")
#|         return {"imp": imp, "keep2": keep2, "model_pruned": mp,
#|                 "pruned_holdout": phm, "pruned_oof": pp, "pruned_summary": sp}
#|     S = mgr.run("importance", _importance)
#|     imp_df, keep_after_imp = S["imp"], S["keep2"]
#|     final_model_pruned, pruned_holdout_metrics = S["model_pruned"], S["pruned_holdout"]
#|     preds_pruned = S["pruned_oof"]
#|     X_pruned, X_pruned_h = X_dev_imp[keep_after_imp], X_hold_imp[keep_after_imp]
#|
#|     # ---- STAGE shap ----
#|     def _shap():
#|         sv, Xs, imp = compute_shap_for_model(final_model_pruned, X_pruned, cfg)
#|         if sv is not None:
#|             np.save(outdir / "shap_values.npy", sv)
#|             Xs.to_csv(outdir / "shap_sample_X.csv", index=False)
#|             imp.to_csv(outdir / "shap_importance.csv", index=False)
#|         stab, pair = compute_shap_stability(fold_models_winner, fold_X_test_winner, cfg)
#|         if not stab.empty:
#|             stab.to_csv(outdir / "shap_stability_per_feature.csv", index=False)
#|             pair.to_csv(outdir / "shap_stability_pairwise.csv", index=False)
#|         return {"shap_imp": imp, "stability": stab, "pairwise": pair}
#|     S = mgr.run("shap", _shap)
#|     shap_imp_df, stability_df = S.get("shap_imp"), S.get("stability")
#|
#|     # ---- STAGE quantile ----
#|     def _quantile():
#|         if not LGB_OK: return {"ok": False}
#|         qp = best_params_all.get("LGB", DEFAULT_PARAMS_FOR_ABLATION["LGB"])
#|         qp = {k: v for k, v in qp.items() if k != "objective"}
#|         qm = fit_quantile_models(X_pruned, y_dev, qp, cfg)
#|         qd = predict_with_intervals(qm, X_pruned); qh = predict_with_intervals(qm, X_pruned_h)
#|         cov = np.nan
#|         if {"q10", "q90"}.issubset(qh.columns):
#|             cov = float(((y_hold.values >= qh["q10"].values) &
#|                          (y_hold.values <= qh["q90"].values)).mean())
#|             log(f"holdout 80% interval coverage: {cov*100:.1f}%")
#|         qd.to_csv(outdir / "quantile_predictions_dev.csv", index=False)
#|         qh.to_csv(outdir / "quantile_predictions_holdout.csv", index=False)
#|         return {"ok": True, "qmodels": qm, "coverage80": cov}
#|     S = mgr.run("quantile", _quantile)
#|     qmodels = S.get("qmodels", {}) if S else {}
#|
#|     # ---- STAGE ensemble ----
#|     def _ensemble():
#|         top = [(r["model"], best_params_all[r["model"]]) for _, r in tournament_df.head(3).iterrows()]
#|         if len(top) < 2: return {"ok": False}
#|         ep, es, ew = fit_top_k_ensemble(splits, top, cfg, years=yr_dev, groups=g_dev)
#|         ep.to_csv(outdir / "ensemble_cv_predictions.csv", index=False)
#|         ehp = np.zeros(len(X_hold_imp))
#|         for name, params in top:
#|             mm = make_estimator(name, params, seed=cfg.seed); mm.fit(X_dev_imp, y_dev)
#|             ehp += ew[name] * mm.predict(X_hold_imp); del mm; gc.collect()
#|         ehm = compute_all_metrics(y_hold, ehp, "ensemble_holdout")
#|         log(f"ensemble CV R²={es.get('R2_mean', np.nan):.4f} | holdout R²={ehm['R2']:.4f}")
#|         return {"ok": True, "summary": es, "weights": ew, "holdout_metrics": ehm}
#|     ens = mgr.run("ensemble", _ensemble) or {"ok": False}
#|
#|     # ---- STAGE ablation (modality single + LOO with sensible defaults) ----
#|     def _ablation():
#|         abl_params = DEFAULT_PARAMS_FOR_ABLATION[winner]
#|         rows = []
#|         full = next((s for s in all_summaries if s["model"] == winner), None)
#|         rows.append({"ablation_type": "FULL_MODEL", "modality": "ALL",
#|                      "n_features": len(kept_cols),
#|                      "R2_mean": full.get("R2_mean") if full else np.nan,
#|                      "RMSE_mean": full.get("RMSE_mean") if full else np.nan})
#|         f2m = {c: assign_modality(c) for c in kept_cols}
#|         for mk in sorted(set(f2m.values())):
#|             mc = [c for c in kept_cols if f2m[c] == mk]
#|             if not mc: continue
#|             try:
#|                 sm = run_cv_on_splits(precompute_fold_splits(X_dev[mc], y_dev, g_dev, cfg),
#|                                       winner, abl_params, cfg, label=f"SINGLE_{mk}",
#|                                       years=yr_dev, groups=g_dev)[1]
#|                 rows.append({"ablation_type": "SINGLE_MODALITY", "modality": mk,
#|                              "n_features": len(mc), "R2_mean": sm.get("R2_mean"),
#|                              "RMSE_mean": sm.get("RMSE_mean")})
#|             except Exception: log_exc(f"single {mk} failed")
#|             loo = [c for c in kept_cols if c not in mc]
#|             if not loo: continue
#|             try:
#|                 lm = run_cv_on_splits(precompute_fold_splits(X_dev[loo], y_dev, g_dev, cfg),
#|                                       winner, abl_params, cfg, label=f"LOO_{mk}",
#|                                       years=yr_dev, groups=g_dev)[1]
#|                 rows.append({"ablation_type": "LEAVE_ONE_OUT", "modality": mk,
#|                              "n_features": len(loo), "R2_mean": lm.get("R2_mean"),
#|                              "RMSE_mean": lm.get("RMSE_mean")})
#|             except Exception: log_exc(f"LOO {mk} failed")
#|         adf = pd.DataFrame(rows).sort_values(["ablation_type", "R2_mean"],
#|                                              ascending=[True, False]).reset_index(drop=True)
#|         adf.to_csv(outdir / "ablation_study.csv", index=False)
#|         return {"ablation": adf}
#|     mgr.run("ablation", _ablation)
#|
#|     # ---- STAGE headline (SDOH vs Env vs Combined + smoothed-SDOH + incremental) ----
#|     def _headline():
#|         sdoh, env = sdoh_env_split(kept_cols)
#|         rows = []
#|         for scen, cols in [("SDOH_only", sdoh), ("Env_only", env), ("Combined", kept_cols)]:
#|             if not cols: continue
#|             _, summ, _, _, _ = run_cv_on_splits(
#|                 precompute_fold_splits(X_dev[cols], y_dev, g_dev, cfg),
#|                 winner, winner_params, cfg, label=scen, years=yr_dev, groups=g_dev)
#|             rows.append({"scenario": scen, "n_features": len(cols),
#|                          "r2": summ.get("R2_mean"), "rmse": summ.get("RMSE_mean"),
#|                          "mae": summ.get("MAE_mean")})
#|         pd.DataFrame(rows).to_csv(outdir / "modality_ablation.csv", index=False)
#|         # smoothed SDOH baseline (IHME comparable)
#|         base_rows = []
#|         if sdoh:
#|             m, oof = baseline_smoothed_sdoh(df.loc[~is_h].reset_index(drop=True), sdoh,
#|                                             winner, winner_params, cfg, g_dev, yr_dev)
#|             base_rows.append({"baseline": "smoothed_sdoh_IHME_comparable", **m})
#|         pd.DataFrame(base_rows).to_csv(outdir / "baselines.csv", index=False)
#|         # incremental leaderboard
#|         anchor_r2 = None
#|         inc_rows = []
#|         if sdoh:
#|             _, sa, _, _, _ = run_cv_on_splits(
#|                 precompute_fold_splits(X_dev[sdoh], y_dev, g_dev, cfg),
#|                 winner, winner_params, cfg, label="anchorSDOH", years=yr_dev, groups=g_dev)
#|             anchor_r2 = sa.get("R2_mean")
#|         for mk in sorted(ENV_MODALITIES):
#|             mc = [c for c in kept_cols if assign_modality(c) == mk]
#|             if not mc: continue
#|             combo = sdoh + mc
#|             _, sm, _, _, _ = run_cv_on_splits(
#|                 precompute_fold_splits(X_dev[combo], y_dev, g_dev, cfg),
#|                 winner, winner_params, cfg, label=f"+{mk}", years=yr_dev, groups=g_dev)
#|             inc_rows.append({"modality_added": mk, "n_features": len(mc),
#|                              "r2": sm.get("R2_mean"),
#|                              "gain_vs_sdoh": (sm.get("R2_mean") - anchor_r2) if anchor_r2 else np.nan,
#|                              "rmse": sm.get("RMSE_mean")})
#|         pd.DataFrame(inc_rows).sort_values("r2", ascending=False).to_csv(
#|             outdir / "incremental_modality.csv", index=False)
#|         return {"scenarios": rows}
#|     mgr.run("headline", _headline)
#|
#|     # ---- STAGE temporal ----
#|     def _temporal():
#|         # true temporal holdout: train < (year_end-1), test last 2 yrs
#|         split_year = cfg.year_end - 1
#|         tr = df.loc[~is_h & (df["year"] < split_year).values].reset_index(drop=True)
#|         te = df.loc[~is_h & (df["year"] >= split_year).values].reset_index(drop=True)
#|         thr = None
#|         if len(tr) > 100 and len(te) > 50:
#|             m, lo, hi = preprocess_fit(tr[kept_cols])
#|             mdl = make_estimator(winner, winner_params, seed=cfg.seed)
#|             mdl.fit(preprocess_transform(tr[kept_cols], m, lo, hi), tr[TARGET_COL].astype(float))
#|             pred = mdl.predict(preprocess_transform(te[kept_cols], m, lo, hi))
#|             thr = compute_all_metrics(te[TARGET_COL].astype(float), pred, "temporal_holdout")
#|             pd.DataFrame([thr]).to_csv(outdir / "temporal_holdout.csv", index=False)
#|             tdf = te[["GEOID", "year", TARGET_COL]].copy(); tdf["pred"] = pred
#|             tdf.to_csv(outdir / "temporal_holdout_predictions.csv", index=False)
#|             log(f"temporal holdout (train<{split_year}): R²={thr['R2']:.4f}")
#|         fwd = forward_chaining_temporal_cv(df.loc[~is_h].reset_index(drop=True),
#|                                            kept_cols, winner, winner_params, cfg)
#|         if not fwd.empty: fwd.to_csv(outdir / "forward_chaining_temporal_cv.csv", index=False)
#|         return {"temporal": thr, "forward": fwd}
#|     mgr.run("temporal", _temporal)
#|
#|     # ---- STAGE forecast (2020-2023 + OOD + intervals) ----
#|     def _forecast():
#|         yr = pd.to_numeric(df_raw["year"], errors="coerce")
#|         fc = df_raw[(yr >= cfg.forecast_start) & (yr <= cfg.forecast_end)].copy()
#|         if not len(fc): return {"ok": False}
#|         fca = fc.copy()
#|         for mod in MODALITY_YEAR_RANGES:
#|             if overlap_fraction(mod, (cfg.year_start, cfg.year_end)) >= cfg.static_modality_thresh:
#|                 continue
#|             for c in [c for c in kept_cols if assign_modality(c) == mod]:
#|                 if c in df.columns:
#|                     fca[c] = fca["GEOID"].map(df.groupby("GEOID")[c].mean())
#|         Xfc = preprocess_transform(fca.reindex(columns=kept_cols), medians_f, lo_f, hi_f)
#|         ood = fit_ood_detector(X_dev_imp, cfg)
#|         d, flag = score_ood(ood, Xfc)
#|         pred = final_model.predict(Xfc)
#|         out = fc[[c for c in ID_COLS if c in fc.columns]].copy()
#|         out["Predicted_LE"] = pred; out["OOD_Mahalanobis"] = d; out["OOD_flag"] = flag
#|         if qmodels:
#|             qi = predict_with_intervals(qmodels, Xfc[keep_after_imp])
#|             for c in qi.columns: out[c] = qi[c].values
#|         out.to_csv(outdir / "forecast_2020_2023.csv", index=False)
#|         if ood.get("available"):
#|             log(f"forecast OOD flagged: {flag.sum():,}/{len(flag):,} ({flag.mean()*100:.1f}%)")
#|         return {"ok": True, "n": len(out), "ood_threshold": ood.get("threshold")}
#|     mgr.run("forecast", _forecast)
#|
#|     # ---- STAGE stats (bootstrap, permutation, wilcoxon) ----
#|     def _stats():
#|         winner_oof = preds_by_model.get(winner)
#|         results = {}
#|         if winner_oof is not None:
#|             boot = bootstrap_r2_ci(winner_oof["actual"], winner_oof["predicted"], cfg)
#|             pd.DataFrame([boot]).to_csv(outdir / "bootstrap_ci.csv", index=False)
#|             results["bootstrap"] = boot
#|             real_r2 = r2_score(winner_oof["actual"], winner_oof["predicted"])
#|             perm, null = permutation_test(splits, winner, winner_params, cfg, real_r2)
#|             pd.DataFrame([perm]).to_csv(outdir / "permutation_test.csv", index=False)
#|             np.save(outdir / "permutation_null_r2.npy", null)
#|             results["permutation"] = perm
#|             log(f"bootstrap R²={boot['r2']:.4f} [{boot['ci_low']:.4f},{boot['ci_high']:.4f}] | "
#|                 f"perm p={perm['p_value']:.4f}")
#|         # paired wilcoxon across top models
#|         wrows = []
#|         models = [m for m in best_params_all if m in preds_by_model]
#|         for i in range(len(models)):
#|             for j in range(i + 1, len(models)):
#|                 wrows.append(paired_wilcoxon(preds_by_model[models[i]], preds_by_model[models[j]],
#|                                              models[i], models[j]))
#|         if wrows: pd.DataFrame(wrows).to_csv(outdir / "paired_wilcoxon.csv", index=False)
#|         return results
#|     mgr.run("stats", _stats)
#|
#|     # ---- STAGE tables ----
#|     def _tables():
#|         wp = preds_by_model.get(winner, preds_pruned)
#|         make_bracket_table(wp).to_csv(outdir / "table_bracket_error.csv", index=False)
#|         make_temporal_table(wp).to_csv(outdir / "table_temporal_performance.csv", index=False)
#|         bc, wc = make_county_error_table(wp)
#|         bc.to_csv(outdir / "table_best_counties.csv", index=False)
#|         wc.to_csv(outdir / "table_worst_counties.csv", index=False)
#|         make_residual_stats(wp).to_csv(outdir / "table_residual_stats.csv", index=False)
#|         # full in-sample predictions with holdout flag
#|         fx = preprocess_transform(X_full[kept_cols], medians_f, lo_f, hi_f)
#|         fp = final_model.predict(fx)
#|         ex = df[[c for c in ID_COLS if c in df.columns] + [TARGET_COL]].copy()
#|         ex["Predicted_LE"] = fp; ex["Residual"] = ex[TARGET_COL] - fp
#|         ex["was_in_holdout"] = is_h
#|         ex.to_csv(outdir / "full_dataset_predictions.csv", index=False)
#|         return {"ok": True}
#|     mgr.run("tables", _tables)
#|
#|     # ---- STAGE export (master manifest + metrics_master) ----
#|     def _export():
#|         # flat metrics_master.csv aggregating headline numbers
#|         rows = []
#|         for m in all_summaries:
#|             rows.append({"experiment": "cv_tournament", "condition": m["model"],
#|                          "r2": m.get("R2_mean"), "rmse": m.get("RMSE_mean"),
#|                          "mae": m.get("MAE_mean")})
#|         rows.append({"experiment": "holdout", "condition": winner,
#|                      "r2": holdout_metrics["R2"], "rmse": holdout_metrics["RMSE"],
#|                      "mae": holdout_metrics["MAE"]})
#|         for f in ["modality_ablation.csv", "baselines.csv", "temporal_holdout.csv"]:
#|             p = outdir / f
#|             if p.exists():
#|                 d = pd.read_csv(p)
#|                 for _, r in d.iterrows():
#|                     rows.append({"experiment": f.replace(".csv", ""),
#|                                  "condition": r.get("scenario") or r.get("baseline") or r.get("label", "-"),
#|                                  "r2": r.get("r2") or r.get("R2"),
#|                                  "rmse": r.get("rmse") or r.get("RMSE"),
#|                                  "mae": r.get("mae") or r.get("MAE")})
#|         pd.DataFrame(rows).to_csv(outdir / "metrics_master.csv", index=False)
#|         manifest = {
#|             "timestamp": datetime.now().isoformat(), "science_hash": mgr.hash,
#|             "config": asdict(cfg), "dataset_sha256": sha256,
#|             "git_hash": (subprocess.getoutput("git rev-parse HEAD") or "n/a")[:40],
#|             "winner": winner, "winner_params": winner_params,
#|             "holdout_metrics": holdout_metrics,
#|             "pruned_holdout_metrics": pruned_holdout_metrics,
#|             "n_features_initial": len(feats_initial), "n_features_kept": len(kept_cols),
#|             "n_features_pruned": len(keep_after_imp),
#|             "n_counties_dev": int(g_dev.nunique()), "n_counties_holdout": len(holdout_geoids),
#|             "leakage_dropped": leakage_cols, "static_aggregated": static_cols,
#|             "package_versions": {"numpy": np.__version__, "pandas": pd.__version__,
#|                                  "xgboost": getattr(xgb, "__version__", None),
#|                                  "lightgbm": getattr(lgb, "__version__", None),
#|                                  "catboost": getattr(cbst, "__version__", None),
#|                                  "shap": getattr(shap, "__version__", None)},
#|             "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
#|                          "xgb_device": XGB_DEVICE, "cb_task_type": CB_TASK_TYPE},
#|         }
#|         (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
#|         return {"ok": True}
#|     mgr.run("export", _export)
#|
#|     mgr.save_timing_report(outdir)
#|     log_sep("PIPELINE COMPLETE")
#|     log(f"Total wall time: {(time.time()-t_start)/60:.1f} min")
#|     log(f"All artifacts in: {outdir}")
#|     log(f"Resume/extend with: --outdir {outdir} [--rerun stage]")
#|
#|
#| if __name__ == "__main__":
#|     try:
#|         main()
#|     except SystemExit:
#|         raise
#|     except Exception:
#|         log_exc("PIPELINE CRASHED")
#|         sys.exit(1)
#|
#@@ END pipeline/run_ml_satallite_and_SDOH.py
