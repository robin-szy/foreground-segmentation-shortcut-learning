#!/bin/bash -l
#SBATCH --job-name=DL_final_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=batch
# SBATCH --partition=gpu,hopper,l40s
# SBATCH --gpus-per-task=1
#SBATCH --qos=besteffort
#SBATCH --time=0-3:00:00
#SBATCH --mail-user=robinszymanski@gmx.de
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=append

set -euo pipefail

echo "========== JOB START $(date) =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=${HOSTNAME}"

module --force purge
module load env/development/2025a-rc0
module load lang/Python/3.13.1-GCCcore-14.2.0

source "$HOME/deep_learning/DL_env_latest/bin/activate"

PROJECT_DIR="$HOME/deep_learning"
export TMP_BASE="/tmp/${USER}/dl_test_${SLURM_JOB_ID}"
export DATA_ROOT="${TMP_BASE}/inaturalist_12K"

trap 'rm -rf "$TMP_BASE"' EXIT

cd "$PROJECT_DIR"
mkdir -p logs runs_10class "$TMP_BASE"

echo "Extracting dataset to ${DATA_ROOT}"
tar -xzf "$HOME/deep_learning/inaturalist_12K.tar.gz" -I gzip -C "$TMP_BASE"

echo "Copying segmented dataset..."
cp -r "$HOME/deep_learning/inaturalist_12K_segmented" "$TMP_BASE/"

echo "Creating split"
cd "$TMP_BASE"
python "$PROJECT_DIR/split_image_dataset.py" \
  --root "$DATA_ROOT" \
  --split-dir "$DATA_ROOT/splits" \
  --seed 42

cd "$PROJECT_DIR"

echo "Checking final models..."
ls -lh "$PROJECT_DIR/final_models"

echo "========== TESTING START $(date) =========="

srun python -u train_10-class_classifier.py \
  --normal-root "$DATA_ROOT" \
  --normal-split-dir "$DATA_ROOT/splits/split_seed_42" \
  --segmented-root "$TMP_BASE/inaturalist_12K_segmented/segm_full_train_val_vitb_pps16" \
  --eval-domains normal \
  --batch-size 32 \
  --num-workers 7 \
  --out-dir runs_10class/final_test_CPU \
  --device cpu

echo "========== JOB END $(date) =========="
