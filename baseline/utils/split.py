# 划分训练集和验证集
import os
import os.path as osp
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
from glob import glob
import numpy as np
import csv


def get_filepaths():
    mp1_4_train = glob('/data3/louwei/MedComm/data/train/*')
    mp5_train = glob('/data3/louwei/MedComm/data/train/*')

    train = mp1_4_train + mp5_train

    train_ID = []
    for ID in train:
        train_ID.append(ID.split('/')[-1].split('.')[0])
    train_ID = list(set(train_ID))
    print(len(train_ID))
    # 读取 Excel 表格
    train_tab = pd.read_excel("/data3/louwei/MedComm/data/table/hx-1.1.xlsx", index_col=0)

    # 获取表格中存在的所有 ID
    valid_ids_in_table = set(train_tab.index.astype(str))  # 假设 ID 是字符串类型

    # 过滤掉不在表格中的 ID
    filtered_train_ID = [id for id in train_ID if id in valid_ids_in_table]
    print(len(filtered_train_ID))
    # 提取对应的标签
    train_DFS = train_tab.loc[filtered_train_ID, 'DFS'].values
    train_OS = train_tab.loc[filtered_train_ID, 'OS'].values
    train_DEvent = train_tab.loc[filtered_train_ID, 'Distant metastasis（no=0；yes=1）'].values
    train_OSEvent = train_tab.loc[filtered_train_ID, 'Death（no=0；yes=1）'].values

    return {
        'ID': filtered_train_ID,
        'DFS': train_DFS,
        'OS': train_OS,
        'DEvent': train_DEvent,
        'OSEvent': train_OSEvent
    }


if __name__ == '__main__':
    train_data = get_filepaths()
    for seed in range(0, 10):
        tr_list, val_list = train_test_split(train_data['ID'], test_size=0.2, random_state=seed, shuffle=True)
        print(len(tr_list),len(val_list))
        # 保存
        os.makedirs("splits_purehx/seed_"+str(seed), exist_ok=True)
        tr_csv_file = "splits_purehx/seed_"+str(seed)+"/train.csv"
        val_csv_file = "splits_purehx/seed_"+str(seed)+"/val.csv"

        with open(tr_csv_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(tr_list)

        with open(val_csv_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(val_list)