#!/bin/bash

# job specifications
#SBATCH --job-name=learning_metrics_semantic_u1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=12:00:00
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
python -m ml.learning_metrics.semantic_metrics -b 5 -u 1 -m 'RNN' -e 5000 -t 10