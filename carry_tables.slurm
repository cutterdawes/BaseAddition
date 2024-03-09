#!/bin/bash
#SBATCH --job-name=carry_tables
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=72:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=cdawes@princeton.edu
#SBATCH --partition=mig

# load modules and activate environment
module purge
module load anaconda3/2023.9
module load openmpi/gcc/4.1.2
conda activate carry_tables

# execute carry_tables.py script depending on arguments given
if [ $# -eq 1 ]; then
    python carry_tables.py -b $1
else
    python carry_tables.py -b $1 -c $2
fi