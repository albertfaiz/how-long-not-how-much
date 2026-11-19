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

@@STEPS@@

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
