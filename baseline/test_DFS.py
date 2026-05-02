import sys
from sklearn import metrics

from torch._C import device 
sys.path.append(".") 
import argparse
from nni.utils import merge_parameter
import nni
from utils.utils import *
from glob import glob
import pandas as pd
from DatasetLoader import MILDatasetLoader, MILFeatureLoader
from model import AttentionNetMoreLayer
import torch.optim as optim
from utils.logger import create_logger
import csv
from tqdm import tqdm

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_factor", type=float, default=0.2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--name", type=str, default="OS_seed0")
    parser.add_argument("--savedir", type=str, default="/data3/louwei/MedComm/codes/baseline/checkpoints/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--jizhi", type=int, default=0)
    parser.add_argument("--lr_patience", type=int, default=100)
    parser.add_argument("--instance_attention_layers", type=int, default=5)
    parser.add_argument("--instance_attention_dim", type=int, default=256)
    parser.add_argument("--feature_attention_layers", type=int, default=5)
    parser.add_argument("--feature_attention_dim", type=int, default=128)
    parser.add_argument("--instance_lr", type=float, default=1e-3)
    parser.add_argument("--feat_lr", type=float, default=1e-3)
    parser.add_argument("--classifier_lr", type=float, default=1e-3)
    parser.add_argument("--split_seed", type=int, default=2)
    parser.add_argument("--aug", type=int, default=1)

def get_filepaths(opt):
    # 1. 提取 test 文件夹中的 ID（转为 int）
    test = glob('/data3/louwei/MedComm/data/efficientnet/test/*')
    test_ID = [ID.split('/')[-1].split('.')[0] for ID in test]
    test_ID = list(set(test_ID))  # 去重

    # 2. 读取 Excel 文件，index_col=0 是 Patient code，是 int 类型
    test_tab = pd.read_excel("/data3/louwei/MedComm/data/table/zs-V17-primary.xlsx", index_col=0)

    # 3. 找出存在于 test_tab 中的 ID
    valid_ids = test_tab.index.intersection(test_ID).tolist()

    print("test_ID:", test_ID)
    print("test_tab.index:", test_tab.index.tolist())
    print("valid_ids:", valid_ids)
    if not valid_ids:
        print("⚠️ 没有找到匹配的 ID！请检查 ID 是否匹配")
        return {
            'ID': [],
            'DFS': np.array([]),
            'OS': np.array([]),
            'DEvent': np.array([]),
            'OSEvent': np.array([])
        }

    # 4. 提取数据
    test_DFS = test_tab.loc[valid_ids, 'DFS'].tolist()
    test_OS = test_tab.loc[valid_ids, 'OS'].tolist()
    test_DEvent = test_tab.loc[valid_ids, 'Distant metastasis（no=0；yes=1）'].tolist()

    # 5. 提取 OSEvent，并处理缺失值
    test_OSEvent = []
    for ID in valid_ids:
        value = test_tab.loc[ID, 'Death（no=0；yes=1）']
        if pd.isna(value):
            print(f"Warning: ID {ID} has missing value in Death column")
            test_OSEvent.append(-1)  # 缺失值用 -1 填充
        else:
            test_OSEvent.append(int(value))

    # 6. 转为 numpy 数组
    test_DFS, test_OS = np.array(test_DFS), np.array(test_OS)
    test_DEvent, test_OSEvent = np.array(test_DEvent), np.array(test_OSEvent)

    print('test_OSEvent', test_OSEvent)

    return {
        'ID': valid_ids,
        'DFS': test_DFS,
        'OS': test_OS,
        'DEvent': test_DEvent,
        'OSEvent': test_OSEvent
    }



def model_eval(test_loader, model, logger):
    model.eval()
    pred_all = None
    OS_all = None
    OSEvent_all = None

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            data = batch['data'].cuda().float()
            if data.dim() == 3 and data.size(0) == 1:
                data = data.squeeze(0)
            data = data.unsqueeze(0)  # Add batch dimension

            pred = model(data)
            if i == 0:
                pred_all = pred
                OS_all = batch['DFS'].float().cpu()
                OSEvent_all = batch['DEvent'].float().cpu()
            else:
                pred_all = torch.cat((pred_all, pred), dim=0)
                OS_all = torch.cat((OS_all, batch['DFS'].float().cpu()), dim=0)
                OSEvent_all = torch.cat((OSEvent_all, batch['DEvent'].float().cpu()), dim=0)

    loss = CoxLoss(OS_all.cuda(), OSEvent_all.cuda(), pred_all.cuda()).item()

    OS_all = OS_all.numpy()
    OSEvent_all = OSEvent_all.numpy()
    pred_all = pred_all.cpu().numpy()

    label = {"time": OS_all, "event": OSEvent_all}
    metrics = eval(pred_all, label, "test")
    print(metrics)
    logger.info(f"C-Index: {metrics['test_cindex']:.4f}")
    logger.info(f"P-Value: {metrics['test_pvalue']:.4f}")

    return pred_all, OS_all, OSEvent_all, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Models")
    opt = get_args(parser)
    opt, remaining_args = parser.parse_known_args()
    nniparams = nni.get_next_parameter()
    opt = merge_parameter(opt, nniparams)
    set_seed(opt.seed)
    print(opt)

    savedir = os.path.join(opt.savedir, opt.name)
    os.makedirs(savedir, exist_ok=True)
    logger = create_logger("%s/test_logfile.txt" % savedir)
    print(savedir)
    # 1. 加载模型
    model = AttentionNetMoreLayer(attentionL=2560, attentionD=512, attentionK=3,
                                  dropout_p=opt.dropout,
                                  instance_attention_layers=opt.instance_attention_layers,
                                  instance_attention_dim=opt.instance_attention_dim,
                                  feature_attention_layers=opt.feature_attention_layers,
                                  feature_attention_dim=opt.feature_attention_dim)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    # 加载权重
    ckpt_path = savedir + '/checkpoint/checkpoint.pt'
    print(ckpt_path)
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Loaded weights from {ckpt_path}")
    else:
        logger.error(f"Checkpoint not found at {ckpt_path}")
        return

    # 2. 加载测试数据
    test_fp = get_filepaths(opt)
    print(len(test_fp['ID']))
    test_dataset = MILFeatureLoader(opt, test_fp, path='test', aug=False, shuffle_bag=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_sz,
        shuffle=False,
        num_workers=opt.n_workers,
        drop_last=False,
        pin_memory=True
    )

    # 3. 推理
    pred_all, OS_all, OSEvent_all, metrics = model_eval(test_loader, model, logger)

    # 4. 保存预测结果
    # 保存为 Excel 文件
    results = {
        "ID": test_fp['ID'],
        "Predictions": pred_all.flatten(),
        #"OS_all": OS_all,
        #"OSEvent_all": OSEvent_all
    }
    df = pd.DataFrame(results)
    output_path = os.path.join(savedir, "predictions.xlsx")
    df.to_excel(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")



if __name__ == '__main__':
    main()