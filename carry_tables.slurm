#!/bin/bash
#SBATCH --job-name=carry_tables
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:0
#SBATCH --time=72:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=cdawes@princeton.edu
#SBATCH --partition=mig

module purge
module load anaconda3/2023.9
conda activate carry_tables

python carry_tables.py $1