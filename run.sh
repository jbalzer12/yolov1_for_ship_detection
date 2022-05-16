#!/bin/sh
module load palma/2020b
module load GCCcore/10.2.0
module load GCC/10.2.0
module load CUDA/11.1.1
module load OpenMPI/4.0.5
module load Python/3.8.6
module load PyTorch/1.9.0
module load torchvision/0.10.0-PyTorch-1.9.0

python train2.py