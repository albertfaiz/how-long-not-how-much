#!/bin/bash
# Reproduce every number, table and figure of the paper from the result files.
# Needs Final_ML_Ready_Dataset.csv at the root and
# results_studyA_20260801_004313/studyA_reference_frame.parquet (both on Zenodo).
set -euo pipefail
cd "$(dirname "$0")/paper_final/analysis"
for script in [0-9][0-9]_*.py; do
    echo "=================================================="
    echo "  running $script"
    echo "=================================================="
    python3 "$script"
done
python3 build_paper_constants.py
python3 paper_figures.py
python3 figure01_methodology_and_graphical_abstract.py
echo "done: constants in paper_final/*.tex, figures in paper_final/Figures/"
