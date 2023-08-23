import numpy as np
import pandas as pd
import csv

def _get_x_y_survival(dataset, os_col_event, os_col_time, dfs_col_event, dfs_col_time, val_outcome):

    y_os = np.empty(dtype=[('event', bool), ('time', np.float64)],
                    shape=dataset.shape[0])
    y_os['event'] = (dataset[os_col_event] == val_outcome).values
    y_os['time'] = dataset[os_col_time].values

    y_dfs = np.empty(dtype=[('event', bool), ('time', np.float64)],
                    shape=dataset.shape[0])
    y_dfs['event'] = (dataset[dfs_col_event] == val_outcome).values
    y_dfs['time'] = dataset[dfs_col_time].values
        
    x_frame = dataset.drop([os_col_event, os_col_time, dfs_col_event, dfs_col_time], axis=1)
    return x_frame, y_os, y_dfs

def append_pathology_pred(tab, path):
    path_pred = pd.read_csv(path + ".csv", index_col=0)
    tab = tab.merge(path_pred, left_index=True,right_index=True, how='left')
    return tab

def obtain_data(pathology_path, table_path):
    pathology_pred_data = pd.read_csv(pathology_path+"/path.csv")
    id_column = 'ID'
    path_ID = pathology_pred_data[id_column]

    tab = pd.read_excel(table_path, index_col=0)   
    tab = tab.loc[path_ID]
    pathology_pred = pd.read_csv(pathology_path + "/path.csv", index_col=0)
    tab = tab.merge(pathology_pred, left_index=True, right_index=True, how='left')
    X, os, dfs = _get_x_y_survival(tab, 'Death（no=0；yes=1）', 'OS', 'Distant metastasis（no=0；yes=1）', 'DFS', 1)
    
    X.replace(' ', float('nan'), inplace=True)
    X.replace(float('nan'), -1, inplace=True)
    X.replace('NaN', '-1', inplace=True)
    
    return X, os, dfs

import pandas as pd
import numpy as np

def find_columns(columns, s):
    for column in columns:
        if s in column:
            return column

def normalize(tab, column):
    normalizer = lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))
    columns = tab.columns.tolist()
    s = find_columns(columns, column)
    value = tab[[s]].values[:, 0]
    value = value[value != -1]
    value_mean = value.mean()
    value = tab[[s]].copy()
    value.replace(-1, value_mean, inplace=True)
    value_ = value.values
    value = value.apply(normalizer)
    tab = tab.drop([s], axis=1)
    tab.insert(len(tab.columns), column, value)
    return tab
    
def onehot(tab, column):
    columns = tab.columns.tolist()
    s = find_columns(columns, column)
    onehot = pd.get_dummies(tab[s]).values
    shapes = onehot.shape[1]
    tab = tab.drop([s], axis=1)
    for i in range(shapes):
        tab.insert(len(tab.columns), column + "_" + str(i), onehot[:, i])
    return tab

def concat_tab(tab1, tab2, tab3, tab4):
    tab = pd.concat([tab1, tab2, tab3, tab4], axis=0)
    return tab

def split_tab(tab, train_len, val_len, test_tab_1_len, test_tab_2_len):
    train_x = tab.iloc[:train_len]
    val_x = tab.iloc[train_len:train_len + val_len]
    test_tab_1 = tab.iloc[train_len + val_len:train_len + val_len + test_tab_1_len]
    test_tab_2 = tab.iloc[train_len + val_len + test_tab_1_len:train_len + val_len + test_tab_1_len + test_tab_2_len]
    return train_x, val_x, test_tab_1, test_tab_2

def preprocess(tab):
    onehot_feature = ["Laterality", "Menopausal status", "cT", "cN", "TNM", "Nuclear grade", "ER before NAT", 
        "PR before NAT", "HER2 status before NAT", "Ki67 before NAT", "WHO grading", "sTils", "iTILs", "LVI", "Surgery type", "NAC regimens", 
        "Radiotherapy", "Endocrine therapy", "Anti-HER2 therapy", "pT", "pN", "pCR", "Molecular classification 1"
    ]
    for feature in onehot_feature:
        try:
           tab = onehot(tab, feature)
        except:
            tab = tab
    # normalize_feature = ["Age", "Primary tumor bed :d1", "Primary tumor bed :d2", "Overall cancer cellularity", "Number of positive lymph nodes", "Diameter of largest metastasis", "cancer_area_ratio", "cancer_area_all", "path"]
    # for feature in normalize_feature:
    #     try:
    #         tab = normalize(tab, feature)
    #     except:
    #         tab = tab
    return tab

def preprocess_all_data(tab):
    tab = preprocess(tab)
    return tab