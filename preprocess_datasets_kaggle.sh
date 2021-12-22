#!/bin/bash
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

python ./preprocess_datasets_kaggle.py

rm -rf raw_files_kaggle
rm -rf unrefined_datasets_kaggle
rm -rf raw_matrices_kaggle