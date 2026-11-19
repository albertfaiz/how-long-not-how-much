#!/bin/bash
# =============================================================================
#  run_studies_v2.sh  ---  both studies on the superset dataset
#
#  UT Dallas Juno.
#  Usage:   sbatch run_studies_v2.sh
#      or:  bash run_studies_v2.sh          (interactive node)
# =============================================================================
#SBATCH --job-name=le_superset
#SBATCH --output=superset_%j.log
#SBATCH --error=superset_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=h200        # Explicitly route to the H200 partition
#SBATCH --cpus-per-task=32      # Reverted to known-good CPU count
#SBATCH --mem=256G              # Reverted to known-good maximum RAM limit
#SBATCH --gres=gpu:1            # Request 1 GPU (Partition flag handles the routing)

set -euo pipefail

# Initialize conda for non-interactive shells and activate your environment
eval "$(conda shell.bash hook)"
conda activate geo_ml

CSV="${CSV:-Final_ML_Ready_Dataset.csv}"
STAMP=$(date +%Y%m%d_%H%M%S)
GPU_FLAG="${GPU_FLAG:---gpu}"     # set GPU_FLAG="" to force CPU

echo "=============================================================="
echo " Life-expectancy superset run  ${STAMP}"
echo " dataset : ${CSV}"
echo " host    : $(hostname)"
echo " conda   : ${CONDA_DEFAULT_ENV}"
echo " python  : $(python3 --version)"
echo "=============================================================="

python3 - <<'PY'
import importlib
need = ["pandas","numpy","sklearn","shap","joblib","pyarrow"]
opt  = ["lightgbm","xgboost","catboost","statsmodels"]
for m in need:
    importlib.import_module(m); print(f"  required  {m:12s} ok")
for m in opt:
    try:
        importlib.import_module(m); print(f"  optional  {m:12s} ok")
    except ImportError:
        print(f"  optional  {m:12s} MISSING (learner or smoother will be skipped)")
PY

# -----------------------------------------------------------------------------
# STUDY A  ->  Papers 1 and 2.  Environment only, 2003-2019.
# -----------------------------------------------------------------------------
echo ""
echo "### STUDY A  (Papers 1 and 2) ###"
python3 pipeline_A_moisture_v2.py \
    --csv "${CSV}" \
    --outdir "results_studyA_${STAMP}" \
    --folds 8 \
    --coverage 0.95 \
    --n-boot 1000 \
    --forward \
    ${GPU_FLAG}

# -----------------------------------------------------------------------------
# STUDY B  ->  Paper 3.  Satellite + socioeconomic fusion, 2009-2019.
# -----------------------------------------------------------------------------
echo ""
echo "### STUDY B  (Paper 3) ###"
python3 pipeline_B_vanishing_v2.py \
    --csv "${CSV}" \
    --outdir "results_studyB_${STAMP}" \
    --folds 8 \
    --coverage 0.80 \
    ${GPU_FLAG}

# =============================================================================
#  AUTO-ORGANIZATION BLOCK 
# =============================================================================
echo ""
echo "=============================================================="
echo " Packaging all outputs into a single master folder..."
echo "=============================================================="

# Define the master folder using the timestamp and SLURM Job ID
JOB_ID=${SLURM_JOB_ID:-local}
MASTER_DIR="Master_Run_${STAMP}_Job${JOB_ID}"
mkdir -p "${MASTER_DIR}"

# 1. Move the model/SHAP artefact directories into the master folder
mv "results_studyA_${STAMP}" "${MASTER_DIR}/" 2>/dev/null || true
mv "results_studyB_${STAMP}" "${MASTER_DIR}/" 2>/dev/null || true

# 2. Sweep up ALL the stray gate, JSON, and parquet files from the root
mv studyA_* "${MASTER_DIR}/" 2>/dev/null || true
mv studyB_* "${MASTER_DIR}/" 2>/dev/null || true

# 3. Save a copy of the run script itself for 100% reproducibility
cp run_studies_v2.sh "${MASTER_DIR}/"

echo "✅ All files safely moved to: ${MASTER_DIR}/"

# 4. Safely capture the SLURM log/err files
# SLURM keeps these open until the bash script officially exits. We spawn a 
# background task that waits 5 seconds for the job to end, then moves the logs.
if [ "$JOB_ID" != "local" ]; then
    (sleep 5 && mv superset_${JOB_ID}.log superset_${JOB_ID}.err "${MASTER_DIR}/" 2>/dev/null) &
fi
