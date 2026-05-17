#!/bin/bash
#SBATCH -J roofline_profile
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 01:00:00 
#SBATCH -o profile_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load python
module load cudatoolkit

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pycudaenv

mkdir -p models

NCU_METRICS="smsp__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__flops_single.sum,dram__bytes_read.sum,dram__bytes_write.sum"

DIR_PATH="setup_profiling"
shopt -s nullglob
for FILE in "${DIR_PATH}"/*.json; do
    base="${FILE%.json}"
    echo "Profiling ${FILE}..."
    srun -n 1 --cpu-bind=cores -G 1 --gpu-bind=none \
        ncu --metrics "${NCU_METRICS}" -f -o "${base}" \
        python src/neuralnet.py "${FILE}" --train --gpu
done
