#!/bin/bash

# job specifications
#SBATCH --job-name=carry_tables
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=00:10:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=cdawes@princeton.edu

# load modules and activate environment
module purge
module load anaconda3/2023.9
module load openmpi/gcc/4.1.2
conda activate env

# change to project root
cd /home/cdawes/BaseAddition

# execute carry_tables.py script
srun -n 4 python stats/carry_tables.py -b 10 -p