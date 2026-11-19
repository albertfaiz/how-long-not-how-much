# Manifests

Each manifest records the settings, library versions and output hashes of the
run that last wrote a step's result files. Those of 11 to 13 October 2026 are
the runs behind the paper's numbers (macOS arm64, XGBoost 3.0.2, scikit-learn
1.7.1, SHAP 0.48.0).

Two result files were written on 11 November 2026 from the stored October
refits, with no new fit:

- `residual_spatial_structure.json` gained Moran's I for the production
  model's own residual and for the outcome (its earlier keys are unchanged, so
  the October manifest of step 13 no longer matches its hash);
- `formaldehyde_drop.json` (step 17) collects the four refits with and without
  the formaldehyde durations.

The fold of every row is in `../fold_maps/`.
