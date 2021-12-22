#!/bin/python

import numpy as np
import pandas as pd
from copy import deepcopy
import os
import zipfile
import urllib.request as req
from dataset_preprocessing_utilities import *
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

#args
raw_files_infos = "download_files_infos_kaggle.csv"
raw_description_file ="format_files_infos_kaggle.csv"
description_task_target = "description_task_target_kaggle.csv"

infos_rep = "./"
raw_files_rep = "./raw_files_kaggle/"
ds_repository = "./unrefined_datasets_kaggle/"
np_repository = "./raw_matrices_kaggle/"
preprocessed_datasets_repository = "./preprocessed_datasets_kaggle/"

verbose = True
rerun = True

#run
#download raw files
df_raw_infos = pd.read_csv(infos_rep+raw_files_infos, index_col = 0)
for row_name, row in df_raw_infos.iterrows():
    kaggle_path = row["kaggle_api_path"]
    kaggle_name = row["kaggle_api_name"]
    new_name = row["new_name"]
    download_name = row["original_file_name"]
    unzipped_name = row["unzipped_name"]
    should_unzip = row["zipped"] 
    if not check_file_exists(new_name, raw_files_rep) or rerun:
        if not check_file_exists(download_name, raw_files_rep):
            print('downloading')
            if row["origine"] == 'dataset':
                new = api.dataset_download_file(kaggle_path,kaggle_name, path = raw_files_rep)
            else:
                new =  api.competition_download_file(kaggle_path,kaggle_name, path = raw_files_rep)
        if should_unzip:
            print("unzipping")
            with zipfile.ZipFile(raw_files_rep+download_name, 'r') as zip_ref:
                zip_ref.extractall(raw_files_rep)
        if check_file_exists(download_name, raw_files_rep):
            os.remove(raw_files_rep+download_name)
        if check_file_exists(download_name, raw_files_rep):
        os.rename(raw_files_rep+unzipped_name, raw_files_rep+new_name)
        
#"raw_file_size"
df_raw_infos["raw_file_size"] = np.zeros(len(df_raw_infos))
for row_name, row in df_raw_infos.iterrows():
    df_raw_infos["raw_file_size"][row_name] = os.path.getsize(raw_files_rep + row["new_name"])
df_raw_infos.sort_values("raw_file_size", inplace = True)
df_raw_infos.to_csv(infos_rep+raw_files_infos)

#populate infos for each task and target combination of every datasets
raw_datasets_infos = pd.read_csv(infos_rep+raw_files_infos, index_col = 0).set_index("pretty_name")#input file_description
raw_description = pd.read_csv(infos_rep+raw_description_file, index_col = 0)
row_names, row_list = [], []
for row_name, row in raw_description.iterrows():
    tasks = list(row["task"])
    targets = str(row["y"]).split(";")
    if len(tasks) == len(targets):
        targetntasks = list(zip(tasks,targets))
    else:
        targetntasks = [(task, target) for target in targets for task in tasks]
    for task, target in targetntasks:
        new_row = row.copy()
        new_row["y"] = target
        new_row["task"] = task
        new_row["excluded_columns"] = ";".join(targets)
        new_row["name_task_target"] = str(new_row["new_name"]) + "_task_"+ str(task)+ "_target_" + str(target)
        row_list.append(new_row)
        row_names.append(row_name)
dataset_by_task_target = pd.DataFrame(row_list)
dataset_by_task_target.sort_values("raw_file_size", inplace = True)
dataset_by_task_target.to_csv(infos_rep+description_task_target)

#manually reformat for unusual separators
file_name = 'Yacht Hydrodynamicsyacht_hydrodynamics.data'
with open(raw_files_rep+file_name, "r") as file:
    data = file.readlines()
for index,line in enumerate(data):
    data[index] = line.replace(" \n", "\n").replace("  ", " ")
with open(raw_files_rep+file_name, 'w') as file:
    file.writelines( data )
del data
for row_name, ds_infos in dataset_by_task_target.iterrows():
    if ds_infos["sep"] == ";":
        with open(raw_files_rep+ds_infos["new_name"], "r") as file:
            data = file.readlines()
        for index,line in enumerate(data):
            data[index] = line.replace(",", ".")
        with open(raw_files_rep+ds_infos["new_name"], 'w') as file:
            file.writelines( data )
        del data
        
#put in standard csv format while keeping only relevant rows
dataset_by_task_target = dataset_by_task_target.where(pd.notnull(dataset_by_task_target), None)
for row_name, ds_infos in dataset_by_task_target.iterrows():
    if verbose: print(row_name,ds_infos["pretty_name"], ds_infos["raw_file_size"])
    if not check_file_exists(ds_infos["name_task_target"],np_repository) or rerun:
        raw_np_mat = panda_read_raw_file(raw_files_rep, ds_infos)
        raw_np_mat.to_csv(np_repository+ds_infos["name_task_target"])
        if verbose: print(row_name, raw_np_mat.shape)
    else:
        print("Already There")
        if verbose: print()
dataset_by_task_target["target_reformated"] = np.zeros(len(dataset_by_task_target)).astype(bool)

#generate .npy file and preprocess features
regressions, classifications, multiclasses = 0, 0, 0
ds_full_name = []
ds_new_name = []
has_failed = []
y_means = []
y_stds = []
for row_name, ds_infos in dataset_by_task_target.iterrows():
    ds_full_name.append(ds_infos["name_task_target"])
    if ds_infos["task"] == "R":
        task_name = "regression"
        index = regressions
        regressions += 1
    if ds_infos["task"] == "C":
        task_name = "classification"
        index = classifications
        classifications += 1

    if ds_infos["task"] == "M":
        task_name = "multiclass"
        index = multiclasses
        multiclasses += 1
    dataset_name = task_name+str(index)
    if not check_file_exists(dataset_name+".npy",preprocessed_datasets_repository) or rerun:
        matrix = pd.read_csv(np_repository+ds_infos["name_task_target"], index_col = 0).values

        data = process_matrix(matrix, ds_infos)

        np.save(preprocessed_datasets_repository+dataset_name, data["data"])

        matrix = data.pop("data")

        n, p = matrix.shape
        if data["info"]["y_info"]["type"] == "numeric" and ds_infos["task"] == "R":
            y_means.append(data["info"]["y_info"]["mean"])
            y_stds.append(data["info"]["y_info"]["std"])
        else:
            y_means.append(None)
            y_stds.append(None)
        ds_infos["info"] = data["info"]
        #jsonify(ds_repository+ds_infos["name_task_target"], ds_infos) TODO: JSONify
        ds_new_name.append(preprocessed_datasets_repository+dataset_name+".npy")
        has_failed.append(False)

    print(ds_infos["name_task_target"],dataset_name)
del data
del matrix

dataset_by_task_target["np_matrix_name"] = ds_new_name
dataset_by_task_target["processing_failed"] = has_failed
dataset_by_task_target["y_mean"] = y_means
dataset_by_task_target["y_std"] = y_stds

# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'restaurant-revenue-predictiontrain.csv_task_R_target_-1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True

# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'mirichoi0218_insuranceinsurance.csv_task_R_target_-1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True

    
# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'snooptosh_bangalore-real-estate-priceblr_real_estate_prices.csv_task_R_target_-1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True

# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'arashnic_marketing-seris-customer-lifetime-valuesquark_automotive_CLV_training_data.csv_task_R_target_2'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True

# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'harlfoxem_housesalespredictionkc_house_data.csv_task_R_target_2'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True

# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'mercedes-benz-greener-manufacturingtrain.csv_task_R_target_1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True

# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'mcarujo_portugal-proprieties-rent-buy-and-vacationportugal_ads_proprieties.csv_task_R_target_2'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True

# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'tabular-playground-series-jan-2021train.csv_task_R_target_-1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(1 - y + np.max(y))
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True

# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'tabular-playground-series-feb-2021train.csv_task_R_target_-1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(1 - y + np.max(y))
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True
