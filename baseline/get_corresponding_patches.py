import os
import pandas as pd
from glob import glob
import shutil

# 配置
ATTENTION_CSV = "predictions_with_attention_sx_os.xlsx"  # 你的 attention 结果文件
PATCH_ROOT = "/data3/louwei/MedComm/data/validation_1_sx/patch/1024"
OUTPUT_ROOT = "./top_attention_patches_sx_top5_OS"  # 输出目录

# 创建输出目录
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 读取 attention 结果
df = pd.read_excel(ATTENTION_CSV)

for _, row in df.iterrows():
    patient_id = row['ID']
    top_indices_str = row['Top_Attention_Indices']
    
    # 跳过空值
    if pd.isna(top_indices_str):
        print(f"Skip {patient_id}: no attention indices")
        continue

    # 获取 patch 路径列表（⚠️ 必须和特征提取时完全一致！）
    patch_pattern = os.path.join(PATCH_ROOT, f"{patient_id}*.ndpi", "*.png")
    patch_paths = glob(patch_pattern)
    
    if not patch_paths:
        print(f"⚠️ No patches found for {patient_id}")
        continue

    # ⚠️ 关键：不要排序！保持 glob 原始顺序（和特征提取时一致）
    # patch_paths = sorted(patch_paths)  # ❌ 不要加这行！

    # 解析 top indices
    try:
        top_indices = [int(x) for x in top_indices_str.split(',')]
    except:
        print(f"⚠️ Invalid indices for {patient_id}: {top_indices_str}")
        continue

    # 创建 patient 输出目录
    patient_out_dir = os.path.join(OUTPUT_ROOT, str(patient_id))
    os.makedirs(patient_out_dir, exist_ok=True)

    # 复制 top patch
    for i, idx in enumerate(top_indices):
        if idx >= len(patch_paths):
            print(f"⚠️ Index {idx} out of range for {patient_id} (total: {len(patch_paths)})")
            continue
        src = patch_paths[idx]
        dst = os.path.join(patient_out_dir, f"top{i+1}_idx{idx}_{os.path.basename(src)}")
        shutil.copy(src, dst)
        print(f"Copied: {patient_id} -> {dst}")

print("✅ Done! Top patches saved to:", OUTPUT_ROOT)