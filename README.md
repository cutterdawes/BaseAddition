# Adding with Alternative Carries
This repository contains all the code and data related to the senior thesis of Cutter Dawes, in partial fulfillment of the requirements for the degree of Bachelor of Arts in Mathematics at Princeton University. 

## Overview
This thesis project investigates the capacity for artificial neural networks to "think" abstractly, as demonstrated by learning how to add multi-digit numbers according to different carry tables.
Furthermore, the models' ability to learn a given carry table is compared to that table's complexity according to several measures.

## Repository Structure
The repository is structured as follows:

- ml/: contains scripts for training neural networks to add using different carry tables
- statistics/: contains notebooks, python scripts, and slurm jobs for finding coycles and computing complexity measures on their carry tables
- base_rep.py: classes for base representations
- utils.py: general utility functions including cocycle-finding functions, etc.
- env.yml: conda environment
- README.md: this file, providing an overview of the repository
