#!/bin/bash
#SBATCH --job-name=model_job
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="thomasmaliappis@gmail.com"
#SBATCH --mail-type="END"
#SBATCH --time=01:30:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=5G

# Default values
config="../config/model.json"
fold=0
epochs=1
dir_name=""

# Read flags and their corresponding values
while getopts "c:f:e:d:" flag; do
  case ${flag} in
  c) config="${OPTARG}" ;;
  f) fold="${OPTARG}" ;;
  e) epochs="${OPTARG}" ;;
  d) dir_name="${OPTARG}" ;;
  *)
    echo "Invalid flag: ${flag}" >&2
    exit 1
    ;;
  esac
done

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load PyYAML/6.0-GCCcore-11.3.0

export PYTHONPATH="${PYTHONPATH}:${HOME}/context-group-detection/"

echo "User: ${SLURM_JOB_USER}, hostname: ${HOSTNAME}, job_id: ${SLURM_JOB_ID}"
echo "Current working directory: $(pwd)"

echo "Using config file at: ${config}, fold: ${fold}, epochs: ${epochs}, dir_name: ${dir_name}"

echo "Script starting"
cd "${HOME}/context-group-detection/models" || exit
python model.py -c ${config} -f ${fold} -e ${epochs} -d ${dir_name}

echo "Script finished"
