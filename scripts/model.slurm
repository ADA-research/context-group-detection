#!/bin/bash
#SBATCH --job-name=model_job
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="thomasmaliappis@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --time=01:30:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=5G

# Default values
fold="0"
epochs="50"
patience="50"
folder=""
eps_thres="1e-15"

# Read flags and their corresponding values
while getopts "e:f:p:m:t:" flag; do
  case ${flag} in
  f) fold="${OPTARG}" ;;
  e) epochs="${OPTARG}" ;;
  p) patience="${OPTARG}" ;;
  m) folder="${OPTARG}" ;;
  t) eps_thres="${OPTARG}" ;;
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

job_id=${SLURM_JOB_ID}
if [ ${folder} != "" ]; then
  job_id="${SLURM_JOB_ID}_${folder}"
fi
echo ${job_id}

echo "Script starting"
cd "${HOME}/context-group-detection/models" || exit
python model.py -e ${epochs} --fold ${fold} -p ${patience} -j ${job_id} -et ${eps_thres}

echo "Epochs: ${epochs}, fold: ${fold}, patience: ${patience}"
echo "Script finished"
