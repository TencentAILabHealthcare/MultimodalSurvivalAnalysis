#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import contextlib
import numpy as np
import random
import shutil
import os
import torch
# Numerical / Array
import lifelines
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_auc_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, checkpoint_path, filename="checkpoint.pt"):
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(state, filename)


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint["state_dict"])


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copied from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def store_preds_to_disk(tgts, preds, args):
    if args.task_type == "multilabel":
        with open(os.path.join(args.savedir, "test_labels_pred.txt"), "w") as fw:
            fw.write(
                "\n".join([" ".join(["1" if x else "0" for x in p]) for p in preds])
            )
        with open(os.path.join(args.savedir, "test_labels_gold.txt"), "w") as fw:
            fw.write(
                "\n".join([" ".join(["1" if x else "0" for x in t]) for t in tgts])
            )
        with open(os.path.join(args.savedir, "test_labels.txt"), "w") as fw:
            fw.write(" ".join([l for l in args.labels]))

    else:
        with open(os.path.join(args.savedir, "test_labels_pred.txt"), "w") as fw:
            fw.write("\n".join([str(x) for x in preds]))
        with open(os.path.join(args.savedir, "test_labels_gold.txt"), "w") as fw:
            fw.write("\n".join([str(x) for x in tgts]))
        with open(os.path.join(args.savedir, "test_labels.txt"), "w") as fw:
            fw.write(" ".join([str(l) for l in args.labels]))


def log_metrics(set_name, metrics, logger):
    logger.info(
        "{}: Loss: {:.5f} | depth_acc: {:.5f}, rgb_acc: {:.5f}".format(
            set_name, metrics["loss"], metrics["depth_acc"], metrics["rgb_acc"]
        )
    )


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)



################
# Survival Utils
################
def CoxLoss(survtime, censor, hazard_pred):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).cuda()
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox


# def CoxLoss(survtime, censor, hazard_pred, device):
#     # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
#     # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
#     current_batch_len = len(survtime)
#     R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
#     for i in range(current_batch_len):
#         for j in range(current_batch_len):
#             R_mat[i,j] = survtime[j] >= survtime[i]

#     R_mat = torch.FloatTensor(R_mat).to(device)
#     theta = hazard_pred.reshape(-1)
#     exp_theta = torch.exp(theta)
#     loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
#     return loss_cox

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_cox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)


def CIndex(hazards, labels, survtime_all):
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total += 1
                    if hazards[j] < hazards[i]: concord += 1
                    elif hazards[j] < hazards[i]: concord += 0.5

    return(concord/total)

def bootstrap_cindex(prediction, event, time, n_bootstrap=1000, ci=0.95):
    n = len(prediction)
    cindices = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        pred_bs = prediction[idx]
        time_bs = time[idx]
        event_bs = event[idx]
        cindex = CIndex_lifeline(pred_bs, event_bs, time_bs)
        cindices.append(cindex)
    # 计算置信区间
    alpha = (1 - ci) * 100
    lower = np.percentile(cindices, alpha / 2)
    upper = np.percentile(cindices, 100 - alpha / 2)
    return lower, upper
def CIndex_lifeline(hazards, labels, survtime_all):
    if np.sum(labels) == 0:
        print("警告：所有样本都未发生事件，无法计算 C-index")
        return np.nan
    if np.all(survtime_all == survtime_all[0]) and np.sum(labels) == len(labels):
        print("警告：所有事件样本的生存时间相同，无法计算 C-index")
        return np.nan
    return(concordance_index(survtime_all, -hazards, labels))

def eval(prediction, test, data):
    cindex_test = CIndex_lifeline(prediction, test['event'], test['time'])
    pvalue_test = cox_log_rank(prediction,  test['event'], test['time'])
    surv_acc_test = accuracy_cox(prediction, test['event'])
    cindex_lower, cindex_upper = bootstrap_cindex(prediction, test['event'], test['time'], n_bootstrap=1000)
    # 一年存活auc
    # 过滤掉没有存活一年（time < 12）且还在进行中的病人（event = false）
    # 获得所有目标样本
    sampel_one_year = np.where(np.logical_and(test['time'] <12, test['event'] == False)==False)
    auc_one_year = roc_auc_score_FIXED(test['time'][sampel_one_year]<12, prediction[sampel_one_year])
    sampel_two_year = np.where(np.logical_and(test['time'] <24, test['event'] == False)==False)
    auc_two_year = roc_auc_score_FIXED(test['time'][sampel_two_year]<24, prediction[sampel_two_year])
    sampel_three_year = np.where(np.logical_and(test['time'] <36, test['event'] == False)==False)
    auc_three_year = roc_auc_score_FIXED(test['time'][sampel_three_year]<36, prediction[sampel_three_year])
    
    # auc, mean_auc = cumulative_dynamic_auc(train, test, prediction, )
    return {data+'_cindex': cindex_test, data+'_cindex_lower': cindex_lower,data+'_cindex_upper': cindex_upper,data+'_pvalue': pvalue_test, data +'_surv_acc': surv_acc_test, \
            data+'_1_auc': auc_one_year, data+'_2_auc': auc_two_year, data+'_3_auc': auc_three_year
        }


def roc_auc_score_FIXED(y_true, y_pred):
    if len(np.unique(y_true)) == 1: # bug in roc_auc_score
        return 0 # accuracy_score(y_true, np.rint(y_pred))
    return roc_auc_score(y_true, y_pred)


