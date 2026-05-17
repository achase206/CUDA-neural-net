#!/bin/bash
# Run once on a Perlmutter login node before submitting jobs.
# CUDA toolkit must be visible when PyCUDA is installed/built.
set -euo pipefail

module load python
module load cudatoolkit

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n pycudaenv -c conda-forge \
    python=3.11 pip numpy matplotlib tqdm rdkit pycuda

conda activate pycudaenv
python -c "from rdkit import Chem; from rdkit.Chem import AllChem; import tqdm; import pycuda.driver as drv; drv.init(); print('env ok')"
