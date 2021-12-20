import os
import numpy as np
import pandas as pd
def check_file_exists(file, repository):
    import os
    return file in os.listdir(repository)
def panda_read_raw_file(repository, ds_infos, always_download = False):
    read_args = {}
    read_args['header']= None
    read_args['index_col']= False
    if ds_infos['header']:
        read_args['skiprows']= ds_infos['header']
    if ds_infos['foot'] != -1:
        read_args['skipfooter'] = int(ds_infos['foot']*(-1)-1)
        read_args['engine'] = "python"
    if ds_infos['encoding']:
        read_args['encoding'] = ds_infos['encoding']
    if ds_infos['sep']:
        read_args['sep'] = ds_infos['sep']
    destination = repository+ds_infos["new_name"]
    if always_download or not check_file_exists(ds_infos["new_name"], repository):
        import urllib.request as req
        url = ds_infos["repository"] + ds_infos['original_file_name']
        req.urlretrieve(url, destination)
    return pd.read_csv(destination,**read_args)
def renorma(vec):
    return (vec - np.mean(vec))/np.std(vec)
import numpy as np
def process_matrix(matrix, filtr):
    n, p = matrix.shape
    extracted_cols = np.zeros(p)

    _NUMERIC_KINDS = set('buif')
    filtr['col_end'] = int(filtr['col_end'])
    filtr['col_start'] = int(filtr['col_start'])
    filtr['y'] = int(filtr['y'])
    
    if filtr['col_end'] == -1:
        extracted_cols[filtr['col_start']:] = 1
    else:
        extracted_cols[filtr['col_start']:filtr['col_end']+1] = 1
    extracted_cols[filtr['y']] = 0
    if "excluded" in filtr.keys():
        for col_index in filtr["excluded"]:
            extracted_cols[col_index] = 0

    y = matrix[:,filtr['y']]
    X = matrix[:,extracted_cols.astype(bool)]
    y_info = {}
    if filtr["task"] == "R":
        y = safe_cast_col(y)
        included_rows = np.logical_not(np.isnan(y))
        X, y = X[included_rows], y[included_rows]
        m, s = y.mean(), y.std()
        y = (y - m) / s
        y_info = {"type":"numeric", "mean":m, "std":(s), "Nan": n - np.sum(included_rows)}

    if filtr["task"] == "C":
        if y.dtype.kind in _NUMERIC_KINDS:
            included_rows = np.logical_not(np.isnan(y))
            X, y = X[included_rows], y[included_rows]
            new_target = np.zeros(n)
            new_target[y == np.max(y)] = 1.
            y_info= {"type" : "numeric", "0" : np.min(y), "1" : np.max(y), "Nan": n - np.sum(included_rows)}
            y = new_target

        else:
            (ens,counts) = np.unique(y.astype(str),return_counts=True)
            most_freq = ens[np.argmax(counts)]
            y_info= {
                "type":"text",
                "0":most_freq,
                "1":ens[np.argmin(counts)]
            }
            y = (y != most_freq).astype(float)
            
    if filtr["task"] == "M":
        if y.dtype.kind in _NUMERIC_KINDS:
            ens = np.unique(y)
            new_target = np.zeros(n)
            y_info= {
            "type":"text"
            }
            for j,val in enumerate(np.sort(ens)):
                new_target[y== val] = j
                y_info[j] = val
            y = new_target.astype(float)
        else:
            (ens,counts) = np.unique(y.astype(str),return_counts=True)
            new_target = np.zeros(n)
            y_info= {
                "type":"text"
            }
            for j,val in enumerate(ens[np.argsort(counts)][::-1]):
                new_target[y== val] = j
                y_info[j] = val
            y = new_target.astype(float)

    new_cols = []
    X_info = {}
    for index,col in enumerate(X.T):
        i = np.arange(p)[extracted_cols.astype(bool)][index]
        try:
            (ens,counts) = np.unique(col,return_counts=True)
        except:
            col = col.astype(str)
            (ens,counts) = np.unique(col,return_counts=True)
        card = len(ens)
        if card == 1:
            continue
        if card == 2:
            most_freq = ens[np.argmax(counts)]
            X_info[len(new_cols)] = {
                "type":"binary",
                "orig":i,
                "0":most_freq,
                "1":ens[np.argmin(counts)]
            }
            new_cols.append((col != most_freq).astype(float))
        else:
            if card <= 12:
                for val in ens:
                    X_info[len(new_cols)] = {
                    "type":"qualitative",
                    "orig":i,
                    "1":val
                    }
                    new_cols.append((col == val).astype(float))
            if type(col[0]) in [float,int]:
                col = cast_col(col)
                    
            if col.dtype.kind in _NUMERIC_KINDS:
                X_info[len(new_cols)] = {
                    "type":"quantitative",
                    "orig":i
                }
                is_nan = np.isnan(col)
                col[is_nan] = 0
                valid_cols = col[np.logical_not(is_nan)]
                m,s = valid_cols.mean(),valid_cols.std()
                col[np.logical_not(is_nan)] = (valid_cols - m)/s
                col = (col-m)/s
                X_info[len(new_cols)] = {
                    "type":"quantitative",
                    "orig":i,
                    "mean":m,
                    "std":s,
                    "is_nan":np.sum(is_nan)
                }
                new_cols.append(col.astype(float))
    Xy = np.concatenate([col.reshape(-1,1) for col in new_cols+[y]], axis = 1)
    return {"data":Xy,"info":{"y_info":y_info,"X_info":X_info}}
def unpack_dict(dico):
    new_dico = {}
    for key, value in dico.items():
        if type(value)== dict:
            for new_key, new_value in unpack_dict(value).items():
                new_dico[str(key)+"/"+str(new_key)] = new_value
        else:
             new_dico[key] = value 
    return new_dico
def jsonify(file_name,dico):
    dico = unpack_dict(dico)
    new_dico = {}
    for key,value in dico.items():
        if type(value)!= str:
            if hasattr(value, "__len__"):
                value = [str(val)for val in value]
            else: 
                value = float(value)
        if type(key)!= str:
            key= float(key)
        new_dico[key] = value
    with open(file_name, 'w') as outfile:
        json.dump(new_dico, outfile)
def cast_col(col):
    new_col = []
    for val in col:
        if type(val) in [float,int]:
            new_col.append(float(val))
        else:
            new_col.append(np.nan)
    return np.array(new_col).astype(float)
def safe_cast_col(col):
    new_col = []
    for val in col:
        try:
            new_val = float(val)
        except:
            new_val = np.nan
        new_col.append(new_val)
    return np.array(new_col).astype(float)