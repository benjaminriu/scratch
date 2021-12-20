#!/bin/bash
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

python ./preprocess_datasets.py

rm -rf raw_files
rm -rf unrefined_datasets
rm -rf raw_matrices