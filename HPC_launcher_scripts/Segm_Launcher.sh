#!/bin/bash -l
#SBATCH --job-name=DL_segm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
# SBATCH --partition=batch
#SBATCH --partition=gpu,hopper,l40s
#SBATCH --gpus-per-task=1
#SBATCH --qos=besteffort
#SBATCH --time=0-12:00:00 #DD-HH:MM:SS
#SBATCH --mail-user=robinszymanski@gmx.de
#SBATCH --mail-type=ALL
#SBATCH --array=1-1
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --open-mode=append


set -euo pipefail

echo "========== JOB START $(date) =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "HOSTNAME=${HOSTNAME}"

module --force purge
module load env/development/2025a-rc0
module load lang/Python/3.13.1-GCCcore-14.2.0

source "$HOME/deep_learning/DL_env_latest/bin/activate"

PROJECT_DIR="$HOME/deep_learning"
CONFIG_FILE="$PROJECT_DIR/configs_segmentation.csv"
DATA_ARCHIVE="$PROJECT_DIR/inaturalist_12K.tar.gz"
SAM_CHECKPOINT_DEFAULT="$PROJECT_DIR/sam_vit_b_01ec64.pth"

export TMP_BASE="/tmp/${USER}/segm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export DATA_ROOT="$TMP_BASE/inaturalist_12K"
trap 'rm -rf "$TMP_BASE"' EXIT

cd "$PROJECT_DIR"
mkdir -p logs inaturalist_12K_segmented

CONFIG_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_FILE")
if [[ -z "${CONFIG_LINE}" ]]; then
  echo "No config line found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

IFS=',' read -r \
  RUN_NAME SCRIPT SEED SPLITS MAX_IMAGES_PER_CLASS POINTS_PER_SIDE PROMPT_MODE \
  SCORE_THRESHOLD BACKGROUND SAM_MODEL_TYPE SAM_CHECKPOINT AMP OVERWRITE \
  <<< "$CONFIG_LINE"

AMP_FLAG=""
if [[ "${AMP}" == "true" ]]; then
  AMP_FLAG="--amp"
fi

OVERWRITE_FLAG=""
if [[ "${OVERWRITE}" == "true" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

MAX_IMAGES_FLAG=""
if [[ -n "${MAX_IMAGES_PER_CLASS}" && "${MAX_IMAGES_PER_CLASS}" != "none" && "${MAX_IMAGES_PER_CLASS}" != "None" ]]; then
  MAX_IMAGES_FLAG="--max-images-per-class ${MAX_IMAGES_PER_CLASS}"
fi

if [[ -z "${SAM_CHECKPOINT}" || "${SAM_CHECKPOINT}" == "default" ]]; then
  SAM_CHECKPOINT="$SAM_CHECKPOINT_DEFAULT"
fi

# Store final masks/debug output in HOME, not /tmp, because /tmp is deleted at the end.
OUT_DIR="$PROJECT_DIR/inaturalist_12K_segmented"

# Allow CSV field like train+val, because spaces are annoying in CSV.
SPLIT_ARGS=${SPLITS//+/ }

echo "========== SEGMENTATION CONFIG =========="
echo "RUN_NAME=${RUN_NAME}"
echo "SCRIPT=${SCRIPT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "SPLITS=${SPLIT_ARGS}"
echo "SEED=${SEED}"
echo "MAX_IMAGES_PER_CLASS=${MAX_IMAGES_PER_CLASS}"
echo "POINTS_PER_SIDE=${POINTS_PER_SIDE}"
echo "PROMPT_MODE=${PROMPT_MODE}"
echo "SCORE_THRESHOLD=${SCORE_THRESHOLD}"
echo "BACKGROUND=${BACKGROUND}"
echo "SAM_MODEL_TYPE=${SAM_MODEL_TYPE}"
echo "SAM_CHECKPOINT=${SAM_CHECKPOINT}"
echo "AMP=${AMP}"
echo "OVERWRITE=${OVERWRITE}"
echo "========================================="

echo "Extracting dataset to ${DATA_ROOT}"
mkdir -p "$TMP_BASE"
tar -xzf "$DATA_ARCHIVE" -I gzip -C "$TMP_BASE"

echo "========== SEGMENTATION START $(date) =========="

python -u "$PROJECT_DIR/$SCRIPT" \
  --run-name "${RUN_NAME}" \
  --data-dir "$DATA_ROOT" \
  --out-dir "$OUT_DIR" \
  --splits ${SPLIT_ARGS} \
  --seed "$SEED" \
  ${MAX_IMAGES_FLAG} \
  --points-per-side "$POINTS_PER_SIDE" \
  --prompt-mode "$PROMPT_MODE" \
  --score-threshold "$SCORE_THRESHOLD" \
  --background "$BACKGROUND" \
  --sam-model-type "$SAM_MODEL_TYPE" \
  --sam-checkpoint "$SAM_CHECKPOINT" \
  --save-masks \
  ${AMP_FLAG} \
  ${OVERWRITE_FLAG}

echo "========== SEGMENTATION END $(date) =========="
