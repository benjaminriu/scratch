#!/bin/bash

## BEGIN SBATCH directives
#SBATCH --job-name=cpu_train
#SBATCH --output=res_cpu_train.txt

#SBATCH --ntasks=1
#SBATCH --time=23:59:59
#SBATCH --partition=cpu_seq
#SBATCH --mem=60G
#SBATCH --account=mlr-deep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=benjamin.riu@polytechnique.edu
## END SBATCH directives

## To clean and load modules defined at the compile and link phases
module purge
module load anaconda3/2020.11

## Execution
which python
conda activate mlrnet_benchmark
which python

NDATAREG=21
NDATACLF=24
NDATAKAG=10
NDATAMUL=13
SEEDS=10

OUTPATH='_strat.csv'
OUTPATHKAG='_strat_kaggle.csv'
METHODS='cpu' #choices=['RF', 'sklearn', "mars", 'cpu']
INTER='interrupt_cpu_trainsize.txt'

DATAPATH=$WORKDIR/datasets/preprocessed_datasets/
DATAPATHKAG=$WORKDIR/datasets/preprocessed_datasets_kaggle/
PYPATH=$(which python)

TRAINPOWER=9
for (( i=0; i<$NDATAREG; i++ )); do
    for (( j=0; j<$TRAINPOWER; j++ )); do
        TRAINSIZE=$((50 * (2 ** $j)))
        srun $PYPATH ./temp_run_all.py --method $METHODS --dataset_id $i --task_name regression --benchmark_output_file $OUTPATH --dataset_seeds $SEEDS --input_repository $DATAPATH --interrupt_file_path $INTER --train_size $TRAINSIZE --should_stratify TRUE
    done
done
for (( i=0; i<$NDATACLF; i++ )); do
    for (( j=0; j<$TRAINPOWER; j++ )); do
        TRAINSIZE=$((50 * (2 ** $j)))
        srun $PYPATH ./temp_run_all.py --method $METHODS --dataset_id $i --task_name classification --benchmark_output_file $OUTPATH --dataset_seeds $SEEDS --input_repository $DATAPATH --interrupt_file_path $INTER --train_size $TRAINSIZE --should_stratify TRUE
    done
done
for (( i=0; i<$NDATAKAG; i++ )); do
    for (( j=0; j<$TRAINPOWER; j++ )); do
        TRAINSIZE=$((50 * (2 ** $j)))
        srun $PYPATH ./temp_run_all.py --method $METHODS --dataset_id $i --task_name regression --benchmark_output_file $OUTPATHKAG --dataset_seeds $SEEDS --input_repository $DATAPATHKAG --interrupt_file_path $INTER --train_size $TRAINSIZE --should_stratify TRUE
    done
done
for (( i=0; i<$NDATAMUL; i++ )); do
    for (( j=0; j<$TRAINPOWER; j++ )); do
        TRAINSIZE=$((50 * (2 ** $j)))
        srun $PYPATH ./temp_run_all.py --method $METHODS --dataset_id $i --task_name multiclass --benchmark_output_file $OUTPATH --dataset_seeds $SEEDS --input_repository $DATAPATH --interrupt_file_path $INTER --train_size $TRAINSIZE --should_stratify TRUE
    done
done