#!/bin/bash

# set number of nodes and tasks if specified
if [ $# -eq 1 ]; then
    nodes=1
    tasks=1
else
    nodes=$2
    tasks=$3
fi

# job specifications
SBATCH="#SBATCH --job-name=carry_tables"
SBATCH="$SBATCH --nodes=$nodes"
SBATCH="$SBATCH --ntasks-per-node=$tasks"
SBATCH="$SBATCH --cpus-per-task=1"
SBATCH="$SBATCH --mem-per-cpu=12G"
SBATCH="$SBATCH --time=00:10:00"
SBATCH="$SBATCH --mail-type=begin,end,fail"
SBATCH="$SBATCH --mail-user=cdawes@princeton.edu"

echo "$SBATCH" >> carry_tables.slurm  # append the generated SBATCH options to script

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