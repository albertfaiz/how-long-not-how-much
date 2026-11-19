import ast, sys
from pathlib import Path
PF = Path(sys.argv[1])            # paper_final
OUT = Path(sys.argv[2])
A = PF / "analysis_final"
PURPOSE = {
 "01": "Check the Python environment and that every input file is present",
 "02": "Distribution of the outcome and model error by outcome bracket",
 "03": "Five reference baselines on the stored county-grouped folds",
 "04": "Drop-one refit of the two night-cooling constructions",
 "05": "Recompute SHAP values for the turning-point candidates",
 "06": "Bootstrap the turning points and apply the stability rule",
 "07": "The learner and its hyperparameters (Appendix Table A1)",
 "10": "Rerun a script and compare its outputs with the earlier run",
 "11": "Is the night-cooling pair one field described twice?",
 "12": "Fields with several admitted summaries and the spread of their attribution",
 "13": "State hold-out and the structure of attribution",
 "14": "Regional invariance of the turning points (median longitude and 100th meridian)",
 "16": "Frozen modelling panels and the coverage-gate audit",
 "17": "Refits without the formaldehyde durations, singly and together",
 "19": "Temporal validation: predicting a year not seen",
 "20": "Null skill and null turning points",
 "21": "The census under eight attribution instruments",
 "22": "Satellite-only and non-satellite modality arms",
 "23": "Influence of single counties and predictor perturbation",
 "24": "Sensitivity to where the admission rules were placed",
 "25": "Duration advantage against atmospheric lifetime",
 "26": "Rank stability and the learning curve",
 "27": "Permutation importance across five held-out splits",
 "28": "Residual audit against social determinants of health",
 "29": "The census against the exchangeable null",
 "30": "Sensitivity of the residual audit",
 "31": "Correlation of the two summaries in each census pair",
 "32": "Consistency of the data versions",
 "33": "The data behind Figure 8C",
 "34": "Sensitivity of the numbers to library versions",
 "35": "Check the statements about methods and data against the code",
 "constants": "Write every number and generated table the manuscript prints",
}
REFITS = {"01","03","04","05","06","13","16","17","19","20","22","23","26","28"}
files = [("analysis/_config.py", A / "_config.py")]
steps = []
for p in sorted(A.glob("[0-9][0-9]_*.py")):
    key = p.name[:2]
    src = p.read_text()
    kind = "refits" if key in REFITS else "reads"
    steps.append((key, p.name, PURPOSE[key], kind))
    files.append((f"analysis/{p.name}", p))
steps.append(("constants", "build_paper_constants.py", PURPOSE["constants"], "reads"))
files.append(("analysis/build_paper_constants.py", A / "build_paper_constants.py"))
# build_paper_constants imports the chemical-family rules from paper_figures
files.append(("analysis/paper_figures.py", A / "paper_figures.py"))
for p in sorted((PF / "pipeline").glob("*.py")):
    files.append((f"pipeline/{p.name}", p))

T = Path(sys.argv[3]).read_text()
steps_txt = "STEPS = [\n" + "".join(
    f"    ({k!r}, {n!r},\n     {d!r}, {c!r}),\n" for k, n, d, c in steps) + "]\n"
T = T.replace("@@STEPS@@", steps_txt)
store = ["", "# " + "=" * 75, "#  STORED SOURCES  (verbatim; every line below is '#|' plus the original)",
         "# " + "=" * 75]
for rel, p in files:
    src = p.read_text()
    store.append(f"#@@ FILE {rel}")
    for ln in src.split("\n"):
        store.append("#|" + (" " + ln if ln else ""))
    store.append(f"#@@ END {rel}")
out = T.rstrip("\n") + "\n" + "\n".join(store) + "\n"
ast.parse(out)
OUT.write_text(out)
print("written", OUT, len(out.splitlines()), "lines,", len(files), "files")
