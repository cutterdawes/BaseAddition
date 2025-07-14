# Training Neural Networks to Add With Different Carry Functions
This repository contains all the code and data related to the preprint "A Group Theoretic Analysis of the Symmetries Underlying Base Addition and Their Learnability by Neural Networks", submitted to the Journal of Mathematical Psychology.

## Repository Structure
Below is a tree of the repository structure including descriptions of each folder and the files therein.
```
.
├── base_rep.py
│   # base representation class
├── environment.yml
│   # conda environment file
├── figures
│   # folder to save figures to (created after first save)
├── ml
│   # folder containing ML scripts and notebook
│   ├── dataset.py
│   │   # base addition data classes
│   ├── learning_metrics
│   │   # folder containing NN simulation scripts
│   │   ├── basic_metrics.py
│   │   │   # generates basic learning metrics
│   │   ├── basic_metrics.slurm
│   │   │   # SLURM script for basic_metrics.py
│   │   ├── ood_metrics.py
│   │   │   # generates OOD generalization metrics
│   │   ├── ood_metrics.slurm
│   │   │   # SLURM script for ood_metrics.py
│   │   ├── ood_semantic_metrics.py
│   │   │   # generates semantic-encoded OOD generalization metrics
│   │   ├── ood_semantic_metrics.slurm
│   │   │   # SLURM script for ood_semantic_metrics.py
│   │   ├── semantic_metrics.py
│   │   │   # generates semantic-encoded learning metrics
│   │   └── semantic_metrics.slurm
│   │       # SLURM script for semantic_metrics.py
│   ├── ml.ipynb
│   │   # Jupyter notebook for ML simulations
│   ├── model.py
│   │   # recurrent model classes
│   └── training.py
│       # model training functions
├── pickles
│   # folder to save pickles to (created after first save)
├── stats
│   # folder containing statistics scripts and notebook
│   ├── associativity.py
│   │   # generates associativity fraction measure
│   ├── associativity.slurm
│   │   # SLURM script for associativity.py
│   ├── carry_tables.py
│   │   # generates carry tables
│   ├── carry_tables.slurm
│   │   # SLURM script for carry_tables.py
│   ├── complexity_measures.py
│   │   # generates complexity measures (fractal dimension, carry frequency)
│   ├── complexity_measures.slurm
│   │   # SLURM script for complexity_measures.py
│   └── statistics.ipynb
│       # Jupyter notebook for statistical analysis and figures
└── utils.py
    # helper functions
```

## Reproducing Results
First, to generate the carry tables, quantitative measures, and learnability results, either (a) run the following locally or (b) properly edit and run the relevant SLURM scripts.
```bash
# create and activate environment
conda env create -f "environment.yml"
conda activate addition

# get carry tables for bases 3, 4, 5
python -m stats.carry_tables --base 3
python -m stats.carry_tables --base 4
python -m stats.carry_tables --base 5

# compile carry tables into all_tables
...

# compute quantitative measures of carry tables
python -m stats.complexity_measures --depth 4
python -m stats.associativity --depth 4

# run ML experiments
# (i) basic learning metrics (default: GRU, 2500 epochs, 10 trials)
python -m ml.learning_metrics.basic_metrics --base 3
python -m ml.learning_metrics.basic_metrics --base 4
python -m ml.learning_metrics.basic_metrics --base 5
# (ii) semantic-encoded learning metrics (default: GRU, 2500 epochs, 10 trials)
python -m ml.learning_metrics.semantic_metrics --base 5 --unit 1
python -m ml.learning_metrics.semantic_metrics --base 5 --unit 2
# (iii) OOD generalization metrics (default: 10 digits max, GRU, 2500 epochs, 10 trials)
python -m ml.learning_metrics.ood_metrics --base 3
python -m ml.learning_metrics.ood_metrics --base 4
python -m ml.learning_metrics.ood_metrics --base 5
```

Then, run through the notebooks and create the relevant figures! Note that all of the above may be replicated with alternatively an RNN (though optimally with 5000 epochs) or an LSTM.
