"""
07_learner_and_hyperparameter_table.py
The settings the model was fitted under, and what each learner scored.

WHY THIS EXISTS
---------------
A reader who wants to know whether a result is an artefact of tuning has no
way to tell from the current manuscript. This builds the table that answers
it, from two sources you already have: the parameter block in the main runner,
and the tournament lines in the run log.

BE CAREFUL WHAT YOU CALL IT
---------------------------
This is NOT a hyperparameter search table. The main runner carries the comment
"gold-mine hyperparameters from prior Optuna runs" and then fits at those fixed
values; the search itself happened in an earlier run whose trial-by-trial
record is not in this log. Presenting this as a search would overclaim.

What it honestly is: the settings each learner was fitted under, and the
out-of-fold score each achieved on the identical county-grouped folds. That is
worth printing, because it shows the winner did not win by being tuned harder
than the others, and it lets a reader reproduce the fit exactly.

If you want a real search table, rerun the main pipeline in 'full' mode, which
re-tunes, and point --log at that run's log instead.

READS   run_ml_satallite_and_SDOH.py   (the parameter block)
        superset_311981.log            (the tournament results)
        studyA_model_tournament.csv    (the same results, as a table)
WRITES  derived/learner_settings.csv
        tables_v3/T5_learner_settings.tex
TIME    seconds

    python3 07_learner_and_hyperparameter_table.py
    python3 07_learner_and_hyperparameter_table.py --log ../../other_run.log
"""
from __future__ import annotations

# ===========================================================================
#  SETTINGS
# ===========================================================================
# Which learners to include, and how to name them in the table.
LEARNERS = {"XGB": "XGBoost", "CB": "CatBoost", "RF": "Random forest",
            "LGB": "LightGBM"}

# Hyperparameters worth printing. Anything else in the block is noise for a
# reader and stays in the released code.
SHOW = ["n_estimators", "iterations", "max_depth", "depth", "learning_rate",
        "subsample", "colsample_bytree", "min_child_weight", "num_leaves",
        "reg_alpha", "reg_lambda", "l2_leaf_reg", "random_strength",
        "min_samples_leaf", "max_features"]

CAPTION = (
    "Learner settings and out-of-fold performance. All four learners were "
    "fitted on the identical county-grouped folds of the same panel. The "
    "hyperparameter values were selected in an earlier tuning run and then "
    "held fixed for every learner, so the tournament measures the learners "
    "and not the effort spent tuning each one. LightGBM did not complete on "
    "the run platform and is reported as not evaluated rather than omitted.")
# ===========================================================================

import argparse
import ast
import json
import re

import pandas as pd

import _config as C


def parse_params(runner_path):
    """Pull BEST_PARAMS_GOLD out of the runner without executing it."""
    src = open(runner_path, encoding="utf-8", errors="replace").read()
    m = re.search(r"BEST_PARAMS_GOLD\s*=\s*\{", src)
    if not m:
        return {}
    i = src.index("{", m.start())
    depth, j = 0, i
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    blob = src[i:j + 1]
    # dict(...) calls are not literals, so rewrite them into dict displays
    blob = re.sub(r"\bdict\(", "{", blob)
    out = {}
    try:
        # convert "key=value" to "'key': value" inside each rewritten dict
        blob2 = re.sub(r"(\{|,)\s*([A-Za-z_]\w*)\s*=", r"\1 '\2':", blob)
        blob2 = blob2.replace(")", "}")
        out = ast.literal_eval(blob2)
    except Exception as e:
        print(f"  could not parse the parameter block automatically "
              f"({type(e).__name__}); falling back to a line scan")
        for key in LEARNERS:
            km = re.search(rf'"{key}"\s*:\s*dict\((.*?)\)\s*,?\s*\n', blob,
                           re.S)
            if km:
                kv = {}
                for pm in re.finditer(r"(\w+)\s*=\s*([^,\)]+)", km.group(1)):
                    try:
                        kv[pm.group(1)] = ast.literal_eval(pm.group(2).strip())
                    except Exception:
                        kv[pm.group(1)] = pm.group(2).strip()
                out[key] = kv
    return out


def parse_log(log_path, study="A"):
    """
    Tournament lines: R2, MAE, RMSE, seconds, and any failure.

    The log holds BOTH studies one after the other, and their tournaments look
    identical apart from the numbers. Reading the whole file and keeping the
    last match returns Study B's fused model, whose R-squared is 0.86 rather
    than 0.67. This paper is Study A, so the file is cut at the study headings
    first. The cross-check against the stored tournament table at the bottom
    of this script exists because that mistake is easy to make and silent.
    """
    res = {}
    if not log_path.exists():
        return res
    text = open(log_path, encoding="utf-8", errors="replace").read()
    marks = [(m.start(), m.group(1)) for m in
             re.finditer(r"###\s*STUDY\s+([AB])\b", text)]
    if marks:
        spans = {}
        for i, (pos, tag) in enumerate(marks):
            stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
            spans[tag] = text[pos:stop]
        if study not in spans:
            print(f"  the log has no 'STUDY {study}' section; "
                  f"found {sorted(spans)}")
            return res
        text = spans[study]
        print(f"  reading the STUDY {study} section only "
              f"({len(text.splitlines())} lines of {len(open(log_path).readlines())})")
    for line in text.splitlines():
        m = re.search(r"\[tournament\]\s+(\w[\w ]*?)\s+"
                      r"R2=([\d.]+)\s+MAE=([\d.]+)\s+RMSE=([\d.]+)\s+\((\d+)s\)",
                      line)
        if m:
            res[m.group(1).strip()] = dict(
                r2=float(m.group(2)), mae=float(m.group(3)),
                rmse=float(m.group(4)), seconds=int(m.group(5)))
            continue
        f = re.search(r"\[tournament\]\s+(\w[\w ]*?)\s+FAILED:\s*(.+?)\s*$",
                      line)
        if f:
            res[f.group(1).strip()] = dict(failed=f.group(2).strip())
        w = re.search(r"\[tournament\]\s+winner:\s*(\w+)", line)
        if w:
            res.setdefault("_winner", w.group(1))
    return res


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default=str(C.RUNNER))
    ap.add_argument("--log", default=str(C.RUN_LOG))
    ap.add_argument("--study", default="A", choices=["A", "B"],
                    help="which study's tournament to read from the log; "
                         "this paper is Study A")
    a = ap.parse_args()

    from pathlib import Path
    runner, log = Path(a.runner), Path(a.log)

    C.banner("07  learner settings and the tournament")
    if not runner.exists():
        print(f"  runner not found at {runner}")
    if not log.exists():
        print(f"  run log not found at {log}")

    params = parse_params(runner) if runner.exists() else {}
    scores = parse_log(log, a.study)
    winner = scores.pop("_winner", None)
    print(f"  parameter blocks found: {', '.join(sorted(params)) or 'none'}")
    print(f"  tournament rows found : {', '.join(sorted(scores)) or 'none'}")
    if winner:
        print(f"  winner recorded in the log: {winner}")

    rows = []
    for key, nice in LEARNERS.items():
        kv = params.get(key, {})
        shown = {k: kv[k] for k in SHOW if k in kv}
        sc = scores.get(nice, {})
        rows.append(dict(
            learner=nice,
            settings="; ".join(f"{k}={fmt(v)}" for k, v in shown.items())
                     or "not recorded",
            r2=sc.get("r2"), mae=sc.get("mae"), rmse=sc.get("rmse"),
            seconds=sc.get("seconds"), failed=sc.get("failed", ""),
            winner=(nice == LEARNERS.get(winner, winner))))
    tab = pd.DataFrame(rows)
    print()
    print(tab[["learner", "r2", "mae", "rmse", "seconds", "failed"]]
          .to_string(index=False))

    # cross-check against the stored tournament table
    if C.TOURNAMENT_A.exists():
        st = pd.read_csv(C.TOURNAMENT_A)
        print(f"\n  against {C.TOURNAMENT_A.name}:")
        for _, r in st.iterrows():
            mine = tab.loc[tab.learner == r.get("model"), "r2"]
            got = float(mine.iloc[0]) if len(mine) and pd.notna(mine.iloc[0]) \
                else None
            same = got is not None and abs(got - float(r["r2"])) < 5e-5
            print(f"    {'ok   ' if same else 'CHECK'} {r.get('model'):<12}"
                  f"log {got}   table {r['r2']}")

    # ---- LaTeX -------------------------------------------------------
    L = [r"\begin{table}[H]",
         r"\caption{" + CAPTION + r"}",
         r"\label{tab:learners}",
         r"\begin{adjustbox}{max width=\textwidth}",
         r"\begin{tabular}{llrrr}", r"\toprule",
         r"Learner & Settings & $R^{2}$ & MAE (yr) & RMSE (yr) \\",
         r"\midrule"]
    for _, r in tab.iterrows():
        s = (r.settings.replace("_", r"\_")
             .replace("colsample\\_bytree", "colsample")
             .replace("min\\_child\\_weight", "min child wt"))
        if r.failed:
            L.append(f"{r.learner} & {s} & \\multicolumn{{3}}{{l}}"
                     f"{{not evaluated: {r.failed}}} \\\\")
        else:
            star = r"\,$^{\ast}$" if r.winner else ""
            L.append(f"{r.learner}{star} & {s} & {r.r2:.4f} & {r.mae:.4f} & "
                     f"{r.rmse:.4f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
          r"\vspace{2pt}",
          r"\footnotesize $^{\ast}$ the learner carried forward. "
          r"All scores are out of fold on county-grouped splits.",
          r"\end{table}"]

    C.DERIVED.mkdir(parents=True, exist_ok=True)
    C.TABLES.mkdir(parents=True, exist_ok=True)
    p1 = C.DERIVED / "learner_settings.csv"
    tab.to_csv(p1, index=False)
    p2 = C.TABLES / "T5_learner_settings.tex"
    open(p2, "w", encoding="utf-8").write("\n".join(L) + "\n")
    p3 = C.DERIVED / "learner_settings.json"
    json.dump(dict(params=params, scores=scores, winner=winner),
              open(p3, "w"), indent=1, default=str)
    print(f"\n[saved] derived/{p1.name}, tables_v3/{p2.name}, "
          f"derived/{p3.name}")
    print("\n  To put it in the paper, add to the appendix:")
    print(r"      \input{tables_v3/T5_learner_settings}")
    C.write_manifest(__file__, [runner, log], [p1, p2, p3],
                     settings=dict(study=a.study),
                     notes="settings-and-tournament table, not a search table")


if __name__ == "__main__":
    main()
