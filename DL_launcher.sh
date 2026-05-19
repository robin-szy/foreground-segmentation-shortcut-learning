#!/bin/bash -l
#SBATCH --job-name=DL_model
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=0-01:00:00 #DD-HH:MM:SS
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#set -euo pipefail

module --force purge
module load env/development/2025a
module load lang/Python/3.13.1-GCCcore-14.2.0


source $HOME/deep_learning/DL_env_latest/bin/activate

export DATA_ROOT=/tmp/inaturalist_12K

echo "Extracting dataset to ${DATA_ROOT}"
mkdir -p /tmp
tar -xzf $HOME/deep_learning/inaturalist_12K.tar.gz -I gzip -C /tmp

srun python train_10-class_classifier.py \
  --mode train \
  --model-type resnet18 \
  --epochs 2 \
  --batch-size 32 \
  --max-train-samples 20 \
  --normal-root /tmp \
  --normal-split-dir /tmp/inaturalist_12K/splits/split_seed_42 \
  --num-workers 4 \
  --amp



