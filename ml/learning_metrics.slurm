#!/bin/bash

# job specifications
#SBATCH --job-name=learning_metrics_32CPU_16workers_pinmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=cdawes@princeton.edu

# load modules and activate environment
module purge
module load anaconda3/2023.9
module load openmpi/gcc/4.1.2
conda activate env

# execute carry_tables.py script
python learning_metrics.py -b 3 -n 16