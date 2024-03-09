#!/bin/bash

# set number of nodes and tasks if specified
if [ $# -eq 3 ]; then
    nodes = $2
    tasks = $3
else
    nodes = 1
    tasks = 1
fi

# job specifications
#SBATCH --job-name=carry_tables
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=00:10:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=cdawes@princeton.edu

# load modules and activate environment
module purge
module load anaconda3/2023.9
module load openmpi/gcc/4.1.2
conda activate carry_tables

# execute carry_tables.py script depending on provided arguments
if [ $# -eq 1 ]; then
    python carry_tables.py -b $1
else
    python carry_tables.py -b $1 -p
fi