#!/bin/bash
#SBATCH --job-name=dante_job
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="thomasmaliappis@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=5G

# Default values
fold="0"
epochs="50"
patience="50"
message=""

# Read flags and their corresponding values
while getopts "e:f:p:m:" flag; do
  case ${flag} in
  f) fold="${OPTARG}" ;;
  e) epochs="${OPTARG}" ;;
  p) patience="${OPTARG}" ;;
  m) message="${OPTARG}" ;;
  *)
    echo "Invalid flag: ${flag}" >&2
    exit 1
    ;;
  esac
done

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

export PYTHONPATH="${PYTHONPATH}:${HOME}/context-group-detection/"

echo "User: ${SLURM_JOB_USER}, hostname: ${HOSTNAME}, job_id: ${SLURM_JOB_ID}"
echo "Current working directory: $(pwd)"

echo "Script starting"
cd "${HOME}/context-group-detection/models" || exit
python dante.py -e ${epochs} -k ${fold} -p ${patience} -j ${SLURM_JOB_ID}

echo "Epochs: ${epochs}, fold: ${fold}, patience: ${patience}"
if [ ${message} != "" ]; then
  echo ${message}
fi
echo "Script finished"
