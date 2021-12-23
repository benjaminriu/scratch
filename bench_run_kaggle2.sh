#!/bin/bash

## BEGIN SBATCH directives
#SBATCH --job-name=run_bench_gpu
#SBATCH --output=res_py11.txt

#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
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
NDATAREG=11
OUTPATH='_kaggle_datasets.csv'
SEEDS=10
METHODS='all'
DATAPATH=$WORKDIR/datasets/preprocessed_datasets_kaggle/
PYPATH=$(which python)
for (( i=0; i<$NDATAREG; i++ )); do
    srun $PYPATH ./temp_run_all.py --method $METHODS --dataset_id $i --task_name regression --benchmark_output_file $OUTPATH --dataset_seeds $SEEDS --input_repository $DATAPATH
done

