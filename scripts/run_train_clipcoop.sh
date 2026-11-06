#!/bin/bash
#SBATCH --job-name=clip_coop_benchmark
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

# Load modules on CINECA Leonardo
module purge
module load profile/deeplrn
module load autoload pytorch/1.12.1

# Navigate to CoOp root
cd $SLURM_SUBMIT_DIR

# Run CoOp on Caltech101 / EuroSAT across shot budgets
DATASET=$1  # caltech101 or eurosat
CFG=rn50_ep50
SHOTS=$2    # 1, 2, 4, 8, 16

bash scripts/coop/main.sh ${DATASET} ${CFG} end ${SHOTS}
