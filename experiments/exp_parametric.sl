#!/bin/bash

#SBATCH --job-name=array_job
#SBATCH --output=/scratch1/jaredcol/slurm-output/%A/output_%A_%a.out
#SBATCH --array=0-5000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Run computation, use SLURM_ARRAY_TASK_ID to differentiate tasks

module purge 
module load gcc/11.3.0
module load git
module load conda

export PYTHONUNBUFFERED=x

conda run -n saga python exp_parametric.py run --datadir "/scratch1/jaredcol/datasets/parametric_benchmarking" --resultsdir "/scratch1/jaredcol/results/parametric" --scheduler $SLURM_ARRAY_TASK_ID --batch
