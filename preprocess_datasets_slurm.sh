#!/bin/bash

## BEGIN SBATCH directives
#SBATCH --job-name=preprocess_datasets
#SBATCH --output=res_prep_ds.txt

#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_seq
#SBATCH --account=mlr-deep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=benjamin.riu@polytechnique.edu
## END SBATCH directives

## To clean and load modules defined at the compile and link phases
module purge
module load anaconda3/2020.11

## Execution
mkdir raw_files
mkdir unrefined_datasets
mkdir raw_matrices
mkdir preprocessed_datasets

GITPATH='https://github.com/benjaminriu/scratch/raw/main'
wget $GITPATH/description_task_target_merged.csv
wget $GITPATH/download_files_infos_merged.csv
wget $GITPATH/format_files_infos_curated.csv
wget $GITPATH/dataset_preprocessing_utilities.py
wget $GITPATH/preprocess_datasets.py

conda activate mlrnet_demo

PYPATH=$(which python)
srun $PYPATH ./preprocess_datasets.py

rm -rf raw_files
rm -rf unrefined_datasets
rm -rf raw_matrices