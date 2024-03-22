# Thesis Repository
This repository contains all the code and data related to the senior thesis of Cutter Dawes, in partial fulfillment of the requirements for the degree of Bachelor of Arts in Mathematics at Princeton University. 

## Overview
This thesis project investigates the capacity for artificial neural networks to "think" abstractly, as demonstrated by learning how to add multi-digit numbers according to different carry tables.
Furthermore, the models' ability to learn a given carry table is compared to that table's complexity according to several measures.

## Repository Structure
The repository is structured as follows:

* ml/: Contains scripts for training ML models to add using different carry tables.
* pickles/: Contains pickled carry tables and complexity measure data.
* statistical/: Contains notebooks, python scripts, and slurm jobs for finding coycles and computing complexity measures on their carry tables.
* fn.py: General utility functions including classes for base representations, cocycle-finding functions, etc.
* env.yml: YML file containing the conda environment used for running scripts and for slurm jobs.
* README.md: This file, providing an overview of the repository.
