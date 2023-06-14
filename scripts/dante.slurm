#!/bin/bash
#SBATCH --job-name=dante_job
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="thomasmaliappis@gmail.com"
#SBATCH --mail-type="END"
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=5G

# Default values
fold="0"
epochs="50"
patience="50"
folder=""
eps_thres="1e-15"
dominant=false
dataset="eth"
dataset_path="../datasets/ETH/seq_eth"

# Read flags and their corresponding values
while getopts "de:f:p:m:t:r:s:" flag; do
  case ${flag} in
  d) dominant=true ;;
  f) fold="${OPTARG}" ;;
  e) epochs="${OPTARG}" ;;
  p) patience="${OPTARG}" ;;
  m) folder="${OPTARG}" ;;
  t) eps_thres="${OPTARG}" ;;
  r) dataset_path="${OPTARG}" ;;
  s) dataset="${OPTARG}" ;;
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

if [ ${folder} != "" ]; then
  dir_name="${folder}"
else
  dir_name=${SLURM_JOB_ID}
fi

if [ ${dominant} == true ]; then
  dominant_flag="-ds"
else
  dominant_flag=""
fi

echo "Dataset: ${dataset}, dataset_path: ${dataset_path}, epochs: ${epochs}, fold: ${fold}, patience: ${patience}, dir_name: ${dir_name}, eps_thres: ${eps_thres}, dominant: ${dominant_flag}"

echo "Script starting"
cd "${HOME}/context-group-detection/models" || exit
python dante.py --dataset ${dataset} --dataset_path ${dataset_path} -e ${epochs} -k ${fold} -p ${patience} --dir_name ${dir_name} -et ${eps_thres} ${dominant_flag}

echo "Script finished"
