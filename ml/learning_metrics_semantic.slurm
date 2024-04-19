#!/bin/bash

# job specifications
#SBATCH --job-name=learning_metrics5_semantic1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=cdawes@princeton.edu

# load modules and activate environment
module purge
module load anaconda3/2024.2
module load openmpi/gcc/4.1.2
conda activate torch-env

# execute carry_tables.py script
python learning_metrics_semantic.py -b 5 -u 1