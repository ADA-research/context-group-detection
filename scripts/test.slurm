#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="thomasmaliappis@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=1G

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

export PYTHONPATH="${PYTHONPATH}:${HOME}/context-group-detection/"

echo "User: ${SLURM_JOB_USER}, hostname: ${HOSTNAME}, job_id: ${SLURM_JOB_ID}"
echo "Current working directory: $(pwd)"

echo "Script starting"
cd "${HOME}/context-group-detection/models" || exit
python model.py -e 50
echo "Script finished"
