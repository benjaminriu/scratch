from get_dataset import get_dataset
from write_results import write_results, check_exp_exists, check_file_exists
import time
import numpy as np
from copy import deepcopy
from get_benchmarked_methods import get_run_
def run_experiment(methods, 
                   datasets, 
                   input_name, 
                   input_repository, 
                   output_file, 
                   output_repository, 
                   dataset_seeds = [0], 
                   method_seeds = [0], 
                   regression = True, 
                   train_size = 0.8,
                   n_features = None,
                   should_stratify = False,
                   prediction_repository = False,
                   interrupt_file_path = False,
                   interrupt_repository = "./",
                   avoid_duplicates = True, 
                   retry_failed_exp= False):
    save_prediction = type(prediction_repository) in [type("./")]
    task_tag = "reg" if regression else "clf"
    header = ["id","task","dataset","n", "p", "dataset_seed","strat","category", "method","method_seed","HPs","time","success"]
    if regression: header += ["R2","R2train"]
    else: header += ["ACC","AUC","ACCtrain","AUCtrain"]
    for dataset_id in datasets:
        for dataset_seed in dataset_seeds:
            X_train, X_test, y_train, y_test = get_dataset(dataset_id, 
                                                           input_name, 
                                                           input_repository, 
                                                           train_size = train_size,
                                                           n_features = n_features,
                                                           should_stratify = should_stratify,
                                                           seed = dataset_seed)
            n, p = X_train.shape
            for method_category, method_name, function, hp_name, hps in methods:
                for method_seed in method_seeds:
                    exp_description = [task_tag, 
                                        dataset_id, 
                                        n, 
                                        p, 
                                        dataset_seed,
                                        should_stratify,
                                        method_category, 
                                        method_name,
                                        method_seed, 
                                        hp_name]
                    exp_id = "_".join(list(map(str, exp_description)))
                    if avoid_duplicates and check_exp_exists(exp_id, output_file, output_repository, check_success = retry_failed_exp):
                        continue
                    start_time = time.time()
                    success, results, prediction = get_run_(function)(X_train, 
                                                      X_test, 
                                                      y_train, 
                                                      y_test, 
                                                      method_name, 
                                                      method_seed, 
                                                      regression = regression,
                                                      hyper_parameters = deepcopy(hps))
                    end_time = time.time() - start_time

                    result_line = [exp_id] + exp_description + [end_time, success] + results
                    
                    if avoid_duplicates and check_exp_exists(exp_id, output_file, output_repository, check_success = retry_failed_exp):
                        continue
                    write_results(result_line, output_file, output_repository, header = header)

                    if save_prediction and success: np.save(prediction_repository+exp_id, prediction)
                        
                    if should_interrupt(interrupt_file_path, interrupt_repository):
                        return False
    return True 

           
def should_interrupt(interrupt_file_path, interrupt_repository):
    if type(interrupt_file_path) == type("interrupt.txt") and type(interrupt_repository) in [type("./")]:
        return check_file_exists(interrupt_file_path, interrupt_repository)
    else:
        return False