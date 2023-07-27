#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="thomasmaliappis@gmail.com"
#SBATCH --mail-type="END"
#SBATCH --time=10:00:00
#SBATCH --partition=cpu-medium
#SBATCH --mem=5G

# Default values
n_balls=10
groups=3
K=3.0
b=0.02
sims=100

# Read flags and their corresponding values
while getopts "n:g:k:b:s:" flag; do
  case ${flag} in
  n) n_balls="${OPTARG}" ;;
  g) groups="${OPTARG}" ;;
  k) K="${OPTARG}" ;;
  b) b="${OPTARG}" ;;
  s) sims="${OPTARG}" ;;
  *)
    echo "Invalid flag: ${flag}" >&2
    exit 1
    ;;
  esac
done

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
pip install imageio

export PYTHONPATH="${PYTHONPATH}:${HOME}/data1/context-group-detection/"

echo "User: ${SLURM_JOB_USER}, hostname: ${HOSTNAME}, job_id: ${SLURM_JOB_ID}"
echo "Current working directory: $(pwd)"

echo "Running, sims: ${sims}, n_balls: ${n_balls}, groups: ${groups}, K: ${K}, b: ${b}"

echo "Script starting"
cd "${HOME}/data1/context-group-detection/datasets/simulation" || exit
python group_simulation.py --n-balls ${n_balls} --groups ${groups} --num-sim ${sims} --K ${K} --b ${b}

echo "Script finished"
