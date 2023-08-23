import sys
sys.path.append(".")
import argparse
from sksurv.ensemble import RandomSurvivalForest
from utils.utils import *
import json
from data import obtain_data, preprocess_all_data

def get_args(parser):
    parser.add_argument("--savedir", type=str, default="./multimodal/haparameter/os")
    parser.add_argument("--tabledata_path", type=str, default="./data/table/table.xlsx")
    parser.add_argument("--pathology_pred_path", type=str, default="./multimodal/pathology/os/")
    parser.add_argument("--n_estimators", type=float, default=1000)
    parser.add_argument("--max_depth", type=float, default=1000)
    parser.add_argument("--min_samples_split", type=float, default=10)
    parser.add_argument("--min_samples_leaf", type=float, default=10)
    parser.add_argument("--seed", type=float, default=42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Models")
    opt = get_args(parser)
    opt, remaining_args = parser.parse_known_args()
    set_seed(opt.seed)
    file = open(opt.savedir+'/parameter.cfg', 'r')
    para = file.read()
    para = json.loads(para)['parameters']
    # Update opt with the parameters from the para dictionary
    for key, value in para.items():
        setattr(opt, key, value)    
    set_seed(opt.seed)
    
    X, os, dfs = obtain_data(opt.pathology_pred_path, opt.tabledata_path)
    X = preprocess_all_data(X)
    estimator = RandomSurvivalForest(
        n_estimators = int(opt.n_estimators),
        max_depth = int(opt.max_depth),
        min_samples_split = int(opt.min_samples_split),
        min_samples_leaf = int(opt.min_samples_leaf)
    ).fit(X, os)

    train_prediction = estimator.predict(X)


