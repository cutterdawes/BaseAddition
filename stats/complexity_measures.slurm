#!/bin/bash

# job specifications
#SBATCH --job-name=complexity_measures
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=12:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=cdawes@princeton.edu

# load modules and activate environment
module purge
module load anaconda3/2023.9
conda activate env

# change to project root
cd /home/cdawes/BaseAddition

# execute carry_tables.py script
python -m stats.complexity_measures --depth 4