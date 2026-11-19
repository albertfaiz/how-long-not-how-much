#!/bin/bash
# Rerun every analysis step of the paper, then write the numbers, tables and
# figures. Needs Final_ML_Ready_Dataset.csv at the repository root and
# results_studyA_20260801_004313/studyA_reference_frame.parquet (both on Zenodo).
set -euo pipefail
cd "$(dirname "$0")/paper_final/analysis_final"
run () { echo "=== $*"; python3 "$@"; }
run 01_check_environment_and_inputs.py
run 02_outcome_distribution_and_error_brackets.py
run 03_baselines_five_reference_models.py
run 04_refit_drop_one_nce_construction.py
run 05_recompute_shap_for_threshold_candidates.py
run 06_bootstrap_turning_points_and_gate.py
run 07_learner_and_hyperparameter_table.py
run 11_verify_nce_identity_by_search.py
run 12_fields_with_several_summaries.py
run 13_spatial_holdout_and_attribution_structure.py
run 14_regional_invariance_of_turning_points.py --all-candidates
run 14_regional_invariance_of_turning_points.py --all-candidates --split meridian
run 16_export_frozen_panels_and_gate_audit.py
run 17_drop_formaldehyde_durations.py
run 19_temporal_validation.py
run 20_null_skill_and_null_crossing.py
run 22_modality_arms.py
run 23_influence_and_perturbation.py
run 25_lifetime_literature_test.py
run 26_rank_stability_and_learning_curve.py
run 28_residual_and_social_determinants.py
run 30_residual_audit_sensitivity.py
run 33_export_figure10c_data.py
run 34_library_version_sensitivity.py
run 35_check_method_statements.py
run build_paper_constants.py
run paper4_figures_all_in_one.py --outdir ../Figures
echo "done: numbers and tables in paper_final/*.tex, figures in paper_final/Figures/"
