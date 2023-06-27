#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="thomasmaliappis@gmail.com"
#SBATCH --mail-type="END"
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=5G

# Default values
config="./config/model.yml"
model="model"
fold=0
seed=14
epochs=1
dir_name=""

# Read flags and their corresponding values
while getopts "m:c:f:e:d:s:" flag; do
  case ${flag} in
  m) model="${OPTARG}" ;;
  c) config="${OPTARG}" ;;
  f) fold="${OPTARG}" ;;
  e) epochs="${OPTARG}" ;;
  d) dir_name="${OPTARG}" ;;
  s) seed="${OPTARG}" ;;
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

echo "Running ${model}"
echo "Using config file at: ${config}, fold: ${fold}, epochs: ${epochs}, dir_name: ${dir_name}, seed: ${seed}"

echo "Script starting"
cd "${HOME}/context-group-detection/models" || exit
python ${model}.py -c ${config} -f ${fold} -e ${epochs} -d ${dir_name} --seed ${seed}

echo "Script finished"
