#!/bin/bash

## BEGIN SBATCH directives
#SBATCH --job-name=noCAT_bench
#SBATCH --output=res_noCAT_default.txt

#SBATCH --ntasks=1
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=60G
#SBATCH --account=mlr-deep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=benjamin.riu@polytechnique.edu
## END SBATCH directives

## To clean and load modules defined at the compile and link phases
module purge
module load anaconda3/2020.11 cuda/10.2

## Execution
which python
conda activate mlrnet_benchmark
which python
nvidia-smi

NDATAREG=21
NDATACLF=24
NDATAKAG=10
NDATAMUL=13
SEEDS=10

OUTPATH='_strat.csv'
OUTPATHKAG='_strat_kaggle.csv'
METHODS='noCAT' #choices=['RF', 'sklearn', 'mlrnet', "regularnet", "fast_mlrnet", "fast_regularnet", "catboost","xgboost", "lgbm", "mars", "all", "best", "fast", "noCAT", "cpu"]
INTER='interrupt_noCAT_default.txt'

DATAPATH=$WORKDIR/datasets/preprocessed_datasets/
DATAPATHKAG=$WORKDIR/datasets/preprocessed_datasets_kaggle/
PYPATH=$(which python)

for (( i=0; i<$NDATAREG; i++ )); do
    srun $PYPATH ./temp_run_all.py --method $METHODS --dataset_id $i --task_name regression --benchmark_output_file $OUTPATH --dataset_seeds $SEEDS --input_repository $DATAPATH --interrupt_file_path $INTER
for (( i=0; i<$NDATACLF; i++ )); do
    srun $PYPATH ./temp_run_all.py --method $METHODS --dataset_id $i --task_name classification --benchmark_output_file $OUTPATH --dataset_seeds $SEEDS --input_repository $DATAPATH --interrupt_file_path $INTER
for (( i=0; i<$NDATAKAG; i++ )); do
    srun $PYPATH ./temp_run_all.py --method $METHODS --dataset_id $i --task_name regression --benchmark_output_file $OUTPATHKAG --dataset_seeds $SEEDS --input_repository $DATAPATHKAG --interrupt_file_path $INTER
done
for (( i=0; i<$NDATAMUL; i++ )); do
    srun $PYPATH ./temp_run_all.py --method $METHODS --dataset_id $i --task_name multiclass --benchmark_output_file $OUTPATH --dataset_seeds $SEEDS --input_repository $DATAPATH --interrupt_file_path $INTER
