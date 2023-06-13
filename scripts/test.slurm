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
verbose=false
output="output.txt"
myflag=""

# Parse flags and their corresponding values
while getopts "vo:" flag; do
  case $flag in
    v) verbose=true ;;
    o) output="$OPTARG" ;;
    -m) myflag="$OPTARG" ;;
    *) echo "Invalid flag: $flag" >&2
       exit 1 ;;
  esac
done

# Display the values of flags
echo "Verbose mode: $verbose"
echo "Output file: $output"
echo "My flag: $myflag"