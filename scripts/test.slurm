#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="thomasmaliappis@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --time=00:05:00
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100M
#SBATCH --gres=gpu:1

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

[s3249484@nodelogin01 ~]$ cd context-group-detection/
[s3249484@nodelogin01 context-group-detection]$ module load Python/3.7.4-GCCcore-8.3.0
[s3249484@nodelogin01 context-group-detection]$ module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
[s3249484@nodelogin01 context-group-detection]$ rm -r python_test_venv/
[s3249484@nodelogin01 context-group-detection]$ python -m venv python_test_venv
[s3249484@nodelogin01 context-group-detection]$ export PYTHONPATH="${PYTHONPATH}:${HOME}/context-group-detection/"
[s3249484@nodelogin01 context-group-detection]$ pip install seaborn
[s3249484@nodelogin01 context-group-detection]$ pip install xlsxwriter
[s3249484@nodelogin01 context-group-detection]$ pip install scikit-learn
[s3249484@nodelogin01 context-group-detection]$ cd models/
[s3249484@nodelogin01 models]$ python model.py
2023-06-06 20:47:58.240680: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1613] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13673 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:d8:00.0, compute capability: 7.5
Epoch 1/10
2023-06-06 20:48:19.588794: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:428] Loaded cuDNN version 8401
 5/90 [>.............................] - ETA: 2s - loss: 0.8144 - mse: 0.2894WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the b90/90 [==============================] - 46s 239ms/step - loss: 0.6399 - mse: 0.2207 - val_loss: 0.6548 - val_mse: 0.2309
Epoch 2/10
1346/1346 [==============================] - 6s 4ms/step 0.4059 - mse: 0.1288
[s3249484@nodelogin01 models]$
