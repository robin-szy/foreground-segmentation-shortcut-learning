#!/bin/bash -l
#SBATCH --job-name=DL_incept
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
# SBATCH --partition=batch
#SBATCH --partition=gpu,hopper,l40s
#SBATCH --gpus-per-task=1
#SBATCH --qos=besteffort
#SBATCH --time=0-04:00:00 #DD-HH:MM:SS
#SBATCH --mail-user=robinszymanski@gmx.de
#SBATCH --mail-type=ALL
#SBATCH --array=1-2
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
CONFIG_FILE="$PROJECT_DIR/configs_inception.csv"
cd "$PROJECT_DIR"
mkdir -p logs runs_10class


CONFIG_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_FILE")

IFS=',' read -r RUN_NAME SCRIPT MODEL_TYPE EPOCHS BATCH_SIZE LR WEIGHT_DECAY DROPOUT SEED AMP PATIENCE OPTIMIZER AUG <<< "$CONFIG_LINE"

AMP_FLAG=""
if [[ "$AMP" == "true" ]]; then
  AMP_FLAG="--amp"
fi

SPLIT_DIR="$DATA_ROOT/splits/split_seed_${SEED}"


echo "Extracting dataset to ${DATA_ROOT}"
# mkdir -p /tmp
# tar -xzf $HOME/deep_learning/inaturalist_12K.tar.gz -I gzip -C /tmp
mkdir -p "$TMP_BASE"
tar -xzf "$HOME/deep_learning/inaturalist_12K.tar.gz" -I gzip -C "$TMP_BASE"

# cd /tmp
cd "$TMP_BASE"
python "$PROJECT_DIR/split_image_dataset.py" \
  --root "$DATA_ROOT" \
  --split-dir "$DATA_ROOT/splits" \
  --seed "$SEED"

cd "$PROJECT_DIR"

echo "========== TRAINING START $(date) =========="

srun python -u "$SCRIPT" \
  --run-name "${RUN_NAME}_job${SLURM_JOB_ID}" \
  --normal-root "$DATA_ROOT" \
  --normal-split-dir "$SPLIT_DIR" \
  --seed "$SEED" \
  --mode train \
  --model-type "$MODEL_TYPE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --dropout "$DROPOUT" \
  --patience "$PATIENCE" \
  --optimizer "$OPTIMIZER" \
  --aug "$AUG" \
  --resume \
  $AMP_FLAG

rm -rf "$TMP_BASE"
