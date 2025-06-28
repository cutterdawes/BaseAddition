#!/bin/bash

# job specifications
#SBATCH --job-name=learning_metrics_ood_semantic_b3
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=cdawes@princeton.edu

# load modules and activate environment
module purge
module load anaconda3/2024.10
module load openmpi/gcc/4.1.6
conda activate addition

# change to project root
cd /home/cdawes/BaseAddition

# execute carry_tables.py script
srun -n 10 python -m ml.learning_metrics.ood_semantic_metrics -b 3 -N 10 -m 'GRU' -t 10 -p