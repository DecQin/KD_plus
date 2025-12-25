#!/bin/bash
#SBATCH --job-name=REKD_plus
#SBATCH --partition=gpu_a100
#SBATCH --gpus=2
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --output=rekd7+++.out

export WANDB_API_KEY="9aa0c7ff9539a3ccd2eefb237d3a7e18d945c013"
export WANDB_ENTITY="18032343160-hefei-university-of-technology"

module load CUDA/12.8.0
source /home/npu/miniconda3/bin/activate py3121

python run_grid_rekd+++.py
