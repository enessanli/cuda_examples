#!/bin/bash
#SBATCH --job-name=esanli
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
##SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

#SBATCH --mem=100G
#SBATCH --time=72:0:0
#SBATCH --output=output.log

module load cuda/11.8.0
module load binutils/2.38
module load gcc/11.2.0

nvcc -o exp1 exp1.cu
./exp1