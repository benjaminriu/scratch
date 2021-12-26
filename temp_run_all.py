import numpy as np
from run_experiment import *
from get_benchmarked_methods import *

import argparse
parser = argparse.ArgumentParser(
    description='test run script',
)
parser.add_argument(
    '--input_repository',
    type=str,
    default="../preprocessed_datasets/",
)
parser.add_argument(
    '--output_repository',
    type=str,
    default="outputs/",
)
parser.add_argument(
    '--benchmark_output_file',
    type=str,
    default="_benchmark_results.csv",
)
parser.add_argument(
    '--prediction_output_file',
    type=str,
    default=None,
)
parser.add_argument(
    '--interrupt_repository',
    type=str,
    default="./",
)
parser.add_argument(
    '--interrupt_file_path',
    type=str,
    default="interrupt.txt",
)
parser.add_argument(
    '--task_name',
    type=str,
    choices=['regression', 'classification'],
    default='regression',
)
parser.add_argument(
    '--method',
    type=str,
    choices=['RF', 
             'sklearn', 
             'mlrnet', 
             "regularnet",
             "fast_mlrnet",
             "fast_regularnet", 
             "catboost",
             "xgboost", 
             "lgbm", 
             "mars", 
             "all", 
             "best",
             "fast",
             "noCAT",
             "cpu"],
    default="RF",
)
parser.add_argument(
    '--hpo',
    type=bool,
    default=False,
)
parser.add_argument(
    '--method_seeds',
    type=int,
    default=1,
)
parser.add_argument(
    '--avoid_duplicates',
    type=str,
    default='True',
)
parser.add_argument(
    '--retry_failed_exp',
    type=str,
    default='True',
)
parser.add_argument(
    '--dataset_id',
    type=int,
    default=0,
)
parser.add_argument(
    '--dataset_seeds',
    type=int,
    default=1,
)
parser.add_argument(
    '--train_size',
    type=str,
    default="0.8",
)
parser.add_argument(
    '--n_features',
    type=str,
    default="None",
)
parser.add_argument(
    '--should_stratify',
    type=bool,
    default=False,
)


args = parser.parse_args()
options = vars(args)

def fma(arg_val, numeric = True):#format_multitype_arg
    if type(arg_val) == type('string'):
        if arg_val in ["None","NONE"]:
            return None
        elif arg_val in ["False","FALSE"]:
            return False
        elif arg_val in ["True","TRUE"]:
            return True
        elif not numeric:
            return arg_val
        elif "." in arg_val:
            return float(arg_val)
        else:
            return int(arg_val)
    return arg_val

if __name__ == '__main__':
    
    input_repository = args.input_repository
    output_repository = args.output_repository
    benchmark_output_file = args.benchmark_output_file
    prediction_output_file = fma(args.prediction_output_file)
    if args.method == "all": 
        methods = ['sklearn', 'mlrnet', "catboost","xgboost", "lgbm", "mars","regularnet"]
    elif args.method == "best":
        methods = ["RF",'mlrnet',"catboost","xgboost","regularnet"]
    elif args.method == "fast":
        methods = ['sklearn', "xgboost", "lgbm", "mars", 'fast_mlrnet',"fast_regularnet"]
    elif args.method == "cpu":
        methods = ['sklearn', "mars"]
    elif args.method == "noCAT": 
        methods = ['sklearn', 'mlrnet', "xgboost", "lgbm", "mars", "regularnet"]
    else:
        methods = [args.method]
    methods += ["HPO"] * args.hpo
    avoid_duplicates = fma(args.avoid_duplicates, numeric = False)
    retry_failed_exp= fma(args.retry_failed_exp, numeric = False)

    datasets = [args.dataset_id]
    dataset_seeds = np.arange(fma(args.dataset_seeds))
    method_seeds = np.arange(fma(args.method_seeds))
    
    train_size = fma(args.train_size)
    n_features = fma(args.n_features)
    should_stratify = arg.should_stratify
    task_name = args.task_name
    
    interrupt_repository = fma(args.interrupt_repository, numeric = False)
    interrupt_file_path = fma(args.interrupt_file_path, numeric = False)
    
    regression = task_name=="regression"
    run_experiment(get_benchmarked_methods(methods, regression = regression), 
                    datasets, 
                    task_name, 
                    input_repository, 
                    task_name+benchmark_output_file, 
                    output_repository,
                    regression = regression, 
                    dataset_seeds = dataset_seeds,
                    method_seeds = method_seeds,
                    train_size = train_size,
                    n_features = n_features,
                    should_stratify = should_stratify,
                    prediction_repository = prediction_output_file,
                    interrupt_file_path = interrupt_file_path,
                    interrupt_repository = interrupt_repository,
                    avoid_duplicates = avoid_duplicates,
                    retry_failed_exp= retry_failed_exp)
 