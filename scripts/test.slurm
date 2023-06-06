#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="thomasmaliappis@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=10M
#SBATCH --time=00:05:00
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

export PYTHONPATH="${PYTHONPATH}:${HOME}/context-group-detection/"

# Source the Python virtual environment
#source "${HOME}/context-group-detection/python_test_venv/bin/activate"

echo "[${SHELL}] #### Starting Python test"
echo "[${SHELL}] ## This is ${SLURM_JOB_USER} on ${HOSTNAME} and this job has the ID ${SLURM_JOB_ID}"
# get the current working directory
export CWD=$(pwd)
echo "[${SHELL}] ## current working directory: ${CWD}"

# Run the file
echo "[${SHELL}] ## Run script"
cd "${HOME}/context-group-detection/models" || exit
python model.py
echo "[${SHELL}] ## Script finished"

echo "[${SHELL}] #### Finished Python test. Have a nice day"
