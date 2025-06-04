#!/bin/bash

# job specifications
#SBATCH --job-name=learning_metrics
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=cdawes@princeton.edu

# load modules and activate environment
module purge
module load anaconda3/2024.2
module load openmpi/gcc/4.1.2
conda activate addition

# change to project root
cd /home/cdawes/BaseAddition

# execute carry_tables.py script
mpirun -np 32 python -m ml.learning_metrics.basic_metrics -b 5 -m 'GRU' -t 10