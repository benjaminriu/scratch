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
mkdir raw_files_kaggle
mkdir unrefined_datasets_kaggle
mkdir raw_matrices_kaggle
mkdir preprocessed_datasets_kaggle

GITPATH='https://github.com/benjaminriu/scratch/raw/main'
wget $GITPATH/description_task_target_kaggle.csv
wget $GITPATH/download_files_infos_kaggle.csv
wget $GITPATH/format_files_infos_kaggle.csv
wget $GITPATH/dataset_preprocessing_utilities.py
wget $GITPATH/preprocess_datasets_kaggle.py

conda activate mlrnet_demo

PYPATH=$(which python)
srun $PYPATH ./preprocess_datasets_kaggle.py

rm -rf raw_files_kaggle
rm -rf unrefined_datasets_kaggle
rm -rf raw_matrices_kaggle