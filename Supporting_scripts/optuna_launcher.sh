#!/bin/bash -l
#SBATCH --job-name=DL_optuna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
# SBATCH --partition=batch
#SBATCH --partition=gpu,hopper,l40s
#SBATCH --gpus-per-task=1
#SBATCH --qos=besteffort
#SBATCH --time=0-03:00:00 #DD-HH:MM:SS
#SBATCH --mail-user=robinszymanski@gmx.de
#SBATCH --mail-type=ALL
#SBATCH --array=1-1
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --open-mode=append
# SBATCH --gres=gpu:1


set -euo pipefail

echo "========== JOB START $(date) =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

module --force purge
module load env/development/2025a-rc0
module load lang/Python/3.13.1-GCCcore-14.2.0


source $HOME/deep_learning/DL_env_latest/bin/activate

# export DATA_ROOT=/tmp/inaturalist_12K
export TMP_BASE="/tmp/${USER}/dl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
trap 'rm -rf "$TMP_BASE"' EXIT

export DATA_ROOT="${TMP_BASE}/inaturalist_12K"
PROJECT_DIR="$HOME/deep_learning"
CONFIG_FILE="$PROJECT_DIR/configs_run5_4.csv"
cd "$PROJECT_DIR"
mkdir -p logs runs_10class


CONFIG_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_FILE")

IFS=',' read -r RUN_NAME SCRIPT MODEL_TYPE EPOCHS BATCH_SIZE LR WEIGHT_DECAY DROPOUT SEED AMP PATIENCE OPTIMIZER AUG USE_SEGMENTED SEGMENTED_PROB SEGMENTED_VAL <<< "$CONFIG_LINE"

AMP_FLAG=""
if [[ "$AMP" == "true" ]]; then
  AMP_FLAG="--amp"
fi

SEGMENTED_FLAGS=""
if [[ "$USE_SEGMENTED" == "true" ]]; then
  SEGMENTED_FLAGS="$SEGMENTED_FLAGS --use-segmented"
fi

if [[ "$SEGMENTED_VAL" == "true" ]]; then
  SEGMENTED_FLAGS="$SEGMENTED_FLAGS --segmented-val"
fi

SPLIT_DIR="$DATA_ROOT/splits/split_seed_${SEED}"


echo "Extracting dataset to ${DATA_ROOT}"
# mkdir -p /tmp
# tar -xzf $HOME/deep_learning/inaturalist_12K.tar.gz -I gzip -C /tmp
mkdir -p "$TMP_BASE"
tar -xzf "$HOME/deep_learning/inaturalist_12K.tar.gz" -I gzip -C "$TMP_BASE"

echo "Copying segmented dataset..."
cp -r "$HOME/deep_learning/inaturalist_12K_segmented" "$TMP_BASE/"

cd "$TMP_BASE"
python "$PROJECT_DIR/split_image_dataset.py" \
  --root "$DATA_ROOT" \
  --split-dir "$DATA_ROOT/splits" \
  --seed "$SEED"

cd "$PROJECT_DIR"

echo "========== TRAINING START $(date) =========="

srun python -u optuna_custom.py \
  --normal-root "$DATA_ROOT" \
  --normal-split-dir "$SPLIT_DIR" \
  --segmented-root "$TMP_BASE/inaturalist_12K_segmented/segm_full_train_val_vitb_pps16" \
  --n-trials 8

rm -rf "$TMP_BASE"
