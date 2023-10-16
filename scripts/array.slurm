#!/bin/bash
#SBATCH --array=0-24
#SBATCH --job-name=train_model
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="thomasmaliappis@gmail.com"
#SBATCH --mail-type="END"
#SBATCH --time=03:30:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=5G

# Default values
config="./config/model.yml"
model="model"
fold=0
epochs=1
dir_name=""
agents=10
frames=1
nc=false
sim=false
name=""

# Read flags and their corresponding values
while getopts "m:c:f:e:d:a:t:o:np" flag; do
  case ${flag} in
  m) model="${OPTARG}" ;;
  c) config="${OPTARG}" ;;
  f) fold="${OPTARG}" ;;
  e) epochs="${OPTARG}" ;;
  d) dir_name="${OPTARG}" ;;
  a) agents="${OPTARG}" ;;
  t) frames="${OPTARG}" ;;
  o) name="${OPTARG}" ;;
  n) nc=true ;;
  p) sim=true ;;
  *)
    echo "Invalid flag: ${flag}" >&2
    exit 1
    ;;
  esac
done

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load PyYAML/6.0-GCCcore-11.3.0

export PYTHONPATH="${PYTHONPATH}:${HOME}/data1/context-group-detection/"

echo "User: ${SLURM_JOB_USER}, hostname: ${HOSTNAME}, job_id: ${SLURM_JOB_ID}, array_task_id: ${SLURM_ARRAY_TASK_ID}"
echo "Current working directory: $(pwd)"

echo "Running ${model}, agents: ${agents}, frames: ${frames}, nc: ${nc}, name: ${name}"
echo "Using config file at: ${config}, fold: ${fold}, epochs: ${epochs}, dir_name: ${dir_name}, seed: ${SLURM_ARRAY_TASK_ID}"

nc_flag=""
if [ ${nc} = true ]; then
  nc_flag="-nc"
fi

sim_flag=""
if [ ${sim} = true ]; then
  sim_flag="--sim"
fi

frames_flag=""
if [ ${frames} -ne 1 ]; then
  frames_flag="-t ${frames}"
fi

name_flag=""
if [ ${name} != "" ]; then
  name_flag="-n ${name}"
fi

echo "Script starting"
cd "${HOME}/data1/context-group-detection/models" || exit
python ${model}.py -c ${config} -f ${fold} -e ${epochs} ${name_flag} -d ${dir_name} --seed ${SLURM_ARRAY_TASK_ID} -a ${agents} ${frames_flag} ${nc_flag} ${sim_flag}

echo "Script finished"
