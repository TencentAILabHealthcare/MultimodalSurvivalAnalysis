import sys
sys.path.append(".")
import argparse
from sksurv.ensemble import RandomSurvivalForest
from utils.utils import *
import json
from data import obtain_data, preprocess_all_data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
import matplotlib.pyplot as plt

def get_args(parser):
    parser.add_argument("--savedir", type=str, default="/data3/louwei/MedComm/codes/multimodal/haparameter/dfs")
    parser.add_argument("--tabledata_path", type=str, default="/data3/louwei/MedComm/data/table/sx - 5.0-更新后.xlsx")
    parser.add_argument("--pathology_pred_path", type=str, default="/data3/louwei/MedComm/codes/multimodal/pathology/os/seed0/")
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
    
    X, OS, dfs = obtain_data(opt.pathology_pred_path, opt.tabledata_path)
    X = preprocess_all_data(X)
    
    estimator = RandomSurvivalForest(
        n_estimators=int(opt.n_estimators),
        max_depth=int(opt.max_depth),
        min_samples_split=int(opt.min_samples_split),
        min_samples_leaf=int(opt.min_samples_leaf)
    ).fit(X, dfs)

    # 替换原来的 predict 行
    train_prediction = estimator.predict(X)  # 转为风险分数：越大风险越高
    scaler = MinMaxScaler()
    risk_score_normalized = scaler.fit_transform(train_prediction.reshape(-1, 1)).flatten()
    event = dfs['event']
    time = dfs['time']
    # 保存时改名
    results_df = pd.DataFrame({
        'ID': X.index,
        'event': event,
        'time': time,
        'risk_score': risk_score_normalized
    })



    # 保存预测结果
    output_path = os.path.join(opt.savedir, "multimodal_predictions.csv")
    results_df.to_csv(output_path, index=False)
    print(f"预测结果已保存至: {output_path}")

    # ====== 开始生存分析逻辑 ======
    print("开始生存分析...")

    working_dir = opt.savedir
    random_seed = opt.seed
    np.random.seed(random_seed)
    os.chdir(working_dir)

    task = "DFS"
    result_path_list = ["multimodal"]
    data_split = "multimodal_predictions.csv"

    def create_save_path(task, result_path, data_split, suffix):
        base_name = os.path.splitext(data_split)[0]
        save_dir = os.path.join(task, result_path)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{suffix}_{base_name}.pdf")
        return save_path

    # 自定义最佳切点函数
    def determine_best_cutoff(df, time_col, event_col, risk_col, thresholds=None):
        if thresholds is None:
            thresholds = np.linspace(df[risk_col].min(), df[risk_col].max(), 100)

        best_p = 1.0
        best_threshold = df[risk_col].median()

        for threshold in thresholds:
            high_risk = df[df[risk_col] > threshold]    
            low_risk = df[df[risk_col] <= threshold]

            if len(high_risk) < 1 or len(low_risk) < 1:
                continue  # 跳过样本太少的情况

            try:
                results = logrank_test(
                    high_risk[time_col], low_risk[time_col],
                    event_observed_A=high_risk[event_col],
                    event_observed_B=low_risk[event_col]
                )
                p_value = results.p_value
                if p_value < best_p:
                    best_p = p_value
                    best_threshold = threshold
            except:
                continue

        return best_threshold

    for result_path in result_path_list:
        file_path = os.path.join(working_dir, data_split)
        print(f"Processing file: {file_path}")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        data = pd.read_csv(file_path)

        # 提取并清洗数据
        data_clean = data[['event', 'time', 'risk_score']].copy()
        data_clean['event'] = data_clean['event'].astype(str).str.contains('True|1').astype(int)
        data_clean['time'] = pd.to_numeric(data_clean['time'])
        data_clean['risk'] = data_clean['risk_score']

        # 找最佳切点
        best_cutoff = determine_best_cutoff(
            data_clean,
            "time",
            "event",
            "risk",
            thresholds=np.linspace(data_clean['risk'].min(), data_clean['risk'].max(), 100)
        )

        # 保存最佳切点到 CSV
        cutoff_summary = pd.DataFrame({"best_cutoff": [best_cutoff]})
        cutoff_file = create_save_path(task, result_path, data_split, "cutoff")
        cutoff_summary.to_csv(cutoff_file.replace(".pdf", ".csv"), index=False)

        # 划分高低风险组
        data_clean['group'] = data_clean['risk'].apply(lambda x: 'High' if x > best_cutoff else 'Low')

        # ---- 插入 C-index 和 95% CI 的计算 ----
        event_indicator = data_clean['event'].astype(bool)
        event_time = data_clean['time'].values
        predicted_risk = data_clean['risk'].values
        print(event_indicator,predicted_risk)
        c_index = concordance_index_censored(event_indicator, event_time, predicted_risk)[0]
        print("C-index:", c_index)

        n_bootstraps = 1000
        np.random.seed(random_seed)
        c_indices = []

        for _ in range(n_bootstraps):
            indices = np.random.choice(len(data_clean), len(data_clean), replace=True)
            boot_data = data_clean.iloc[indices]
            event_indicator_boot = boot_data['event'].astype(bool)
            event_time_boot = boot_data['time'].values
            predicted_risk_boot = boot_data['risk'].values
            try:
                c_idx = concordance_index_censored(event_indicator_boot, event_time_boot, predicted_risk_boot)[0]
                if np.isnan(c_idx):
                    continue
                c_indices.append(c_idx)
            except:
                continue

        ci_lower, ci_upper = np.percentile(c_indices, [2.5, 97.5])
        print(f"C-index: {c_index:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")

        # 保存 C-index 到 CSV
        cindex_df = pd.DataFrame({
            "C_index": [c_index],
            "CI_lower": [ci_lower],
            "CI_upper": [ci_upper]
        })
        cindex_path = create_save_path(task, result_path, data_split, "cindex").replace(".pdf", ".csv")
        cindex_df.to_csv(cindex_path, index=False)

        # ---- 插入结束 ----

        # 拟合 Kaplan-Meier 曲线
        kmf = KaplanMeierFitter()

        groups = data_clean['group']
        strata = groups.unique()

        color_map = {'Low': 'blue', 'High': 'red'}

        plt.figure(figsize=(10, 6))

        for group in ['Low', 'High']:
            idx = (data_clean['group'] == group)
            if idx.sum() == 0:
                continue
            kmf.fit(data_clean.loc[idx, 'time'], event_observed=data_clean.loc[idx, 'event'], label=group)
            kmf.plot_survival_function(ci_show=False, color=color_map[group])

        plt.title(f"Kaplan-Meier Curve by Risk Group (Cutoff={best_cutoff:.4f})")
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.grid(True)

        # Log-rank test
        high_risk = data_clean[data_clean['group'] == 'High']
        low_risk = data_clean[data_clean['group'] == 'Low']

        results = logrank_test(
            high_risk['time'], low_risk['time'],
            event_observed_A=high_risk['event'],
            event_observed_B=low_risk['event']
        )
        plt.text(0.6, 0.15, f"p = {results.p_value:.4f}", transform=plt.gca().transAxes)

        # 保存为 PDF
        save_path = create_save_path(task, result_path, data_split, "surv")
        plt.tight_layout()
        plt.savefig(save_path, format='pdf')
        plt.close()

    print("已完成 sx_merge_update.csv 的生存分析，结果已保存。")