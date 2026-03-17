#!/bin/bash -l
#SBATCH --job-name=python_ex
#SBATCH --node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=0-02:00:00 #DD-HH:MM:SS
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

srun tar -xzf $HOME/deep_learning/inaturalist_12K.tar.gz -I gzip -C /tmp
srun python pymod.py

# Pytorch: GPU
# Start by simple script printing GPU parameters
