import sys
sys.path.append(".")
import argparse
from DatasetLoader import MILFeatureLoader, obtain_data
from model import AttentionNetMoreLayer
from utils.utils import *
import json


def get_args(parser):
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--savedir", type=str, default="./pathology/checkpoint/os")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--instance_attention_layers", type=int, default=5)
    parser.add_argument("--instance_attention_dim", type=int, default=256)
    parser.add_argument("--feature_attention_layers", type=int, default=5)
    parser.add_argument("--feature_attention_dim", type=int, default=128)
    parser.add_argument("--aug", type=int, default=0)
    parser.add_argument("--table_path", type=str, default="./data/table/table.xlsx")
    parser.add_argument("--pathology_path", type=str, default="./data/pathology_feature")


def model_eval(test_loader, model, dataname):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            data = batch['data'].cuda().float()
            data = data.squeeze(0)
            pred = model(data)
            if i!=0:
                pred_all = torch.cat((pred_all, pred), 0)
                OS_all = torch.cat((OS_all, batch['OS'].cuda().float()), 0)
                OSEvent_all = torch.cat((OSEvent_all, batch['OSEvent'].cuda().float()), 0)
            else:
                pred_all = pred
                OS_all, OSEvent_all = batch['OS'].cuda().float(), batch['OSEvent'].cuda().float()

    OS_all =  OS_all.detach().cpu().numpy()
    OSEvent_all = OSEvent_all.detach().cpu().numpy()
    pred_all = pred_all.detach().cpu().numpy()
    # label = {"time": OS_all, "event": OSEvent_all}
    # metrics = eval(pred_all, label, dataname)
    # return metrics, pred_all
    return pred_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Models")
    opt = get_args(parser)
    opt, remaining_args = parser.parse_known_args()
    set_seed(opt.seed)
    file = open(opt.savedir+'/parameter.cfg', 'r')
    para = file.read()
    para = json.loads(para)['parameters']
    # Update opt with the parameters from the para dictionary
    for key, value in para.items():
        setattr(opt, key, value)

    model = AttentionNetMoreLayer(attentionL=2560, attentionD=512, attentionK=3, \
        dropout_p=opt.dropout, instance_attention_layers=opt.instance_attention_layers, instance_attention_dim=opt.instance_attention_dim, \
            feature_attention_layers=opt.feature_attention_layers, feature_attention_dim=opt.feature_attention_dim)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    load_checkpoint(model, opt.savedir+"/checkpoint.pt")
    data_path = obtain_data(opt.table_path, opt.pathology_path)

    val_data = MILFeatureLoader(opt, data_path, aug=False, shuffle_bag=False)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers = opt.n_workers,
        drop_last = False,
        pin_memory = True
    )

    model.eval()
    val_pred = model_eval(val_loader, model, 'val')
    print(val_pred)



 


    

