#!/bin/bash
#SBATCH -J ab3_accuracy
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 01:00:00 
#SBATCH -o accuracy_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load python
module load cudatoolkit

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pycudaenv

mkdir -p models

DIR_PATH="setup_accuracy"
shopt -s nullglob
for FILE in "${DIR_PATH}"/*.json; do
    echo "Training ${FILE}..."
    srun -n 1 --cpu-bind=cores -G 1 --gpu-bind=none \
        python src/neuralnet.py "${FILE}" --train --gpu
done
